#!/usr/bin/env python3
"""
RetinaFace + SAM3 + GazeAnywhere Multi-Person Gaze Estimation Pipeline

This pipeline uses:
1. RetinaFace (via insightface) for accurate face detection on the first frame
2. SAM3 Video Tracker with bounding box prompts for tracking faces through the video
3. GazeAnywhere for per-person gaze estimation

Usage:
    python retinaface_gaze_pipeline.py --video_path /path/to/video.mp4
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class FaceDetection:
    """Face detection result from RetinaFace."""
    face_id: int
    bbox: Tuple[float, float, float, float]  # (x1, y1, x2, y2) normalized
    bbox_px: Tuple[int, int, int, int]  # (x1, y1, x2, y2) pixels
    score: float
    landmarks: Optional[np.ndarray] = None  # 5-point facial landmarks


@dataclass
class TrackedFace:
    """Tracked face result from SAM3."""
    face_id: int
    bbox: Tuple[float, float, float, float]  # (x1, y1, x2, y2) normalized
    bbox_px: Tuple[int, int, int, int]  # (x1, y1, x2, y2) pixels
    score: float
    mask: Optional[np.ndarray] = None


@dataclass
class GazeResult:
    """Gaze estimation result for a tracked face."""
    face_id: int
    face_bbox: Tuple[float, float, float, float]
    face_bbox_px: Tuple[int, int, int, int]
    head_bbox: Tuple[float, float, float, float]
    head_bbox_px: Tuple[float, float, float, float]
    gaze_point: Tuple[float, float]
    gaze_point_px: Tuple[float, float]
    inout: bool
    detection_score: float


# ============================================================================
# RetinaFace Face Detector
# ============================================================================

class RetinaFaceDetector:
    """
    Face detector using InsightFace's RetinaFace model.
    
    RetinaFace provides high-accuracy face detection with 5-point landmarks.
    """
    
    def __init__(self, device: str = "cuda", det_thresh: float = 0.5):
        """
        Initialize RetinaFace detector.
        
        Args:
            device: "cuda" or "cpu"
            det_thresh: Detection confidence threshold
        """
        self.device = device
        self.det_thresh = det_thresh
        self._app = None
    
    def _ensure_loaded(self):
        """Lazy-load the InsightFace model."""
        if self._app is None:
            try:
                from insightface.app import FaceAnalysis
                
                # Use CUDA if available
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if self.device == "cuda" else ['CPUExecutionProvider']
                
                self._app = FaceAnalysis(
                    name='buffalo_l',  # Default model with RetinaFace
                    providers=providers
                )
                self._app.prepare(ctx_id=0 if self.device == "cuda" else -1, det_thresh=self.det_thresh)
                print("RetinaFace detector loaded successfully")
                
            except ImportError as e:
                raise ImportError(
                    "InsightFace not installed. Please install: pip install insightface onnxruntime-gpu"
                ) from e
    
    def detect_faces(
        self, 
        frame_bgr: np.ndarray
    ) -> List[FaceDetection]:
        """
        Detect faces in a frame.
        
        Args:
            frame_bgr: BGR image (OpenCV format)
            
        Returns:
            List of FaceDetection objects
        """
        self._ensure_loaded()
        
        height, width = frame_bgr.shape[:2]
        
        # Run detection
        faces = self._app.get(frame_bgr)
        
        detections = []
        for i, face in enumerate(faces):
            # face.bbox is [x1, y1, x2, y2] in pixels
            x1, y1, x2, y2 = face.bbox
            
            # Normalize
            bbox_norm = (
                float(x1) / width,
                float(y1) / height,
                float(x2) / width,
                float(y2) / height,
            )
            
            bbox_px = (int(x1), int(y1), int(x2), int(y2))
            
            detections.append(FaceDetection(
                face_id=i,
                bbox=bbox_norm,
                bbox_px=bbox_px,
                score=float(face.det_score),
                landmarks=face.kps if hasattr(face, 'kps') else None,
            ))
        
        # Sort by x-coordinate for consistent ordering
        detections.sort(key=lambda d: d.bbox_px[0])
        
        # Reassign IDs after sorting
        for i, det in enumerate(detections):
            det.face_id = i
        
        return detections


# ============================================================================
# SAM3 Bounding Box Tracker
# ============================================================================

class Sam3BboxTracker:
    """
    SAM3 video tracker using bounding box prompts.
    
    Uses face bboxes from RetinaFace as prompts to track each person
    through the video.
    """
    
    def __init__(self, device: str = "cuda"):
        """
        Initialize SAM3 video tracker.
        
        Args:
            device: Device for inference
        """
        self.device = device
        self._predictor = None
        self._session_id = None
    
    def _ensure_loaded(self):
        """Lazy-load SAM3 video predictor."""
        if self._predictor is None:
            try:
                from sam3.model_builder import build_sam3_video_predictor
                self._predictor = build_sam3_video_predictor()
                print("SAM3 video predictor loaded successfully")
            except ImportError as e:
                raise ImportError(
                    "SAM3 not installed. Please install from: "
                    "git clone https://github.com/facebookresearch/sam3.git && "
                    "cd sam3 && pip install -e '.[notebooks]'"
                ) from e
    
    def start_session(self, video_path: str) -> str:
        """Start a video tracking session."""
        self._ensure_loaded()
        
        response = self._predictor.start_session(
            resource_path=video_path,
            session_id=None,
        )
        self._session_id = response["session_id"]
        return self._session_id
    
    def add_face_prompts(
        self, 
        face_detections: List[FaceDetection], 
        frame_idx: int = 0,
        video_size: Tuple[int, int] = None,
    ) -> None:
        """
        Add face bounding boxes as tracking prompts.
        
        Args:
            face_detections: List of face detections from RetinaFace
            frame_idx: Frame index where faces were detected
            video_size: (width, height) of the video
        """
        if self._session_id is None:
            raise RuntimeError("No active session. Call start_session() first.")
        
        for face in face_detections:
            # Convert normalized xyxy to normalized xywh for SAM3
            x1, y1, x2, y2 = face.bbox
            x = x1
            y = y1
            w = x2 - x1
            h = y2 - y1
            
            # SAM3 expects boxes_xywh in normalized format
            bbox_xywh = [[x, y, w, h]]
            
            # Add prompt for this face
            self._predictor.add_prompt(
                session_id=self._session_id,
                frame_idx=frame_idx,
                bounding_boxes=bbox_xywh,
                bounding_box_labels=[1],  # Positive label
                obj_id=face.face_id,
            )
    
    def propagate_tracking(
        self, 
        max_frames: Optional[int] = None
    ) -> Dict[int, List[TrackedFace]]:
        """
        Propagate tracking through the video.
        
        Returns:
            Dict mapping frame_idx to list of TrackedFace
        """
        if self._session_id is None:
            raise RuntimeError("No active session. Call start_session() first.")
        
        results = {}
        
        # Use handle_stream_request for propagate_in_video
        request = dict(
            type="propagate_in_video",
            session_id=self._session_id,
            propagation_direction="forward",
            start_frame_index=0,
            max_frame_num_to_track=max_frames if max_frames else None,
        )
        
        propagation = self._predictor.handle_stream_request(request=request)
        
        for frame_output in propagation:
            frame_idx = frame_output["frame_index"]
            outputs = frame_output["outputs"]
            
            if outputs is None:
                continue
            
            tracked_faces = []
            
            # SAM3 video predictor returns:
            # - out_obj_ids: numpy array of object IDs
            # - out_probs: numpy array of detection scores  
            # - out_boxes_xywh: numpy array of boxes in (x, y, w, h) normalized format
            # - out_binary_masks: numpy array of binary masks
            obj_ids = outputs.get("out_obj_ids", [])
            scores = outputs.get("out_probs", [])
            boxes_xywh = outputs.get("out_boxes_xywh", [])
            masks = outputs.get("out_binary_masks", None)
            
            for i, obj_id in enumerate(obj_ids):
                score = scores[i] if i < len(scores) else 1.0
                
                # Convert xywh (normalized) to xyxy (normalized)
                # SAM3 returns (x, y, w, h) where x,y is TOP-LEFT corner
                if i < len(boxes_xywh):
                    x, y, w, h = boxes_xywh[i]
                    x1, y1 = x, y
                    x2, y2 = x + w, y + h
                    bbox_norm = (x1, y1, x2, y2)
                else:
                    bbox_norm = (0, 0, 0, 0)
                
                tracked_faces.append(TrackedFace(
                    face_id=int(obj_id),
                    bbox=bbox_norm,
                    bbox_px=(0, 0, 0, 0),  # Will compute later
                    score=float(score),
                    mask=masks[i] if masks is not None and i < len(masks) else None,
                ))
            
            results[frame_idx] = tracked_faces
        
        return results
    
    def close_session(self):
        """Close the current session."""
        if self._session_id is not None and self._predictor is not None:
            self._predictor.close_session(session_id=self._session_id)
            self._session_id = None
    
    def process_video(
        self,
        video_path: str,
        face_detections: List[FaceDetection],
        max_frames: Optional[int] = None,
    ) -> Tuple[Dict[int, List[TrackedFace]], Tuple[int, int], float]:
        """
        Full video processing: track faces through video.
        
        Args:
            video_path: Path to video file
            face_detections: Face detections from first frame
            max_frames: Maximum frames to process
            
        Returns:
            Tuple of:
                - Dict mapping frame_idx to list of TrackedFace
                - Video size (width, height)
                - Video FPS
        """
        # Get video metadata
        cap = cv2.VideoCapture(video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        try:
            # Start session
            self.start_session(video_path)
            
            # Add face prompts from first frame
            print(f"Adding {len(face_detections)} face prompts...")
            self.add_face_prompts(face_detections, frame_idx=0, video_size=(width, height))
            
            # Propagate through video
            print(f"Tracking faces through video ({total_frames} frames)...")
            tracked_faces = self.propagate_tracking(max_frames=max_frames)
            
            # Convert normalized bboxes to pixel coordinates
            for frame_idx, faces in tracked_faces.items():
                for face in faces:
                    x1, y1, x2, y2 = face.bbox
                    face.bbox_px = (
                        int(x1 * width),
                        int(y1 * height),
                        int(x2 * width),
                        int(y2 * height),
                    )
            
            return tracked_faces, (width, height), fps
            
        finally:
            self.close_session()


# ============================================================================
# GazeAnywhere Integration (reuse from existing code)
# ============================================================================

class GazePredictor:
    """Wrapper for GazeAnywhere model."""
    
    def __init__(
        self, 
        checkpoint_path: str = "/projects/illinois/eng/cs/jrehg/users/xucao2/ChildGaze/checkpoints/GazeAnywhere/gazeanywhere.pth",
        device: str = "cuda"
    ):
        self.device = device
        self.checkpoint_path = checkpoint_path
        self._predictor = None
    
    def _ensure_loaded(self):
        if self._predictor is None:
            # Import from existing gaze_anywhere module
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            from gaze_anywhere.inference import GazePredictor as _GazePredictor
            self._predictor = _GazePredictor(
                checkpoint_path=self.checkpoint_path,
                device=self.device,
            )
            print("GazeAnywhere model loaded successfully")
    
    def predict(self, frame_rgb: np.ndarray, position_prompt: str) -> dict:
        """
        Predict gaze for a person given their position.
        
        Args:
            frame_rgb: RGB image
            position_prompt: Position prompt like "position: [0.5, 0.3]"
            
        Returns:
            Dict with gaze_point, inout, head_bbox, etc.
        """
        self._ensure_loaded()
        return self._predictor.predict(frame_rgb, position_prompt)
    
    @staticmethod
    def format_position_prompt(bbox_norm: Tuple[float, float, float, float]) -> str:
        """
        Format a bounding box as a position prompt for GazeAnywhere.
        
        Args:
            bbox_norm: (x1, y1, x2, y2) normalized coordinates
            
        Returns:
            Position prompt string like "position: [0.5, 0.7]"
        """
        x1, y1, x2, y2 = bbox_norm
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        return f"position: [{cx:.4f}, {cy:.4f}]"


# ============================================================================
# Full Pipeline Annotator
# ============================================================================

class RetinaFaceGazeAnnotator:
    """
    Full pipeline: RetinaFace → SAM3 → GazeAnywhere
    """
    
    def __init__(
        self,
        device: str = "cuda",
        gaze_checkpoint: str = "/projects/illinois/eng/cs/jrehg/users/xucao2/ChildGaze/checkpoints/GazeAnywhere/gazeanywhere.pth",
        det_thresh: float = 0.5,
    ):
        self.device = device
        self.face_detector = RetinaFaceDetector(device=device, det_thresh=det_thresh)
        self.tracker = Sam3BboxTracker(device=device)
        self.gaze_predictor = GazePredictor(checkpoint_path=gaze_checkpoint, device=device)
    
    def annotate_video(
        self,
        video_path: str,
        max_frames: Optional[int] = None,
        progress_bar: bool = True,
    ) -> Dict:
        """
        Annotate a video with face detection, tracking, and gaze estimation.
        
        Returns:
            Dict with video metadata and per-frame annotations
        """
        # Get video info
        cap = cv2.VideoCapture(video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Read first frame for face detection
        ret, first_frame = cap.read()
        if not ret:
            raise ValueError(f"Cannot read first frame from {video_path}")
        cap.release()
        
        # Step 1: Detect faces on first frame with RetinaFace
        print("Detecting faces on first frame with RetinaFace...")
        face_detections = self.face_detector.detect_faces(first_frame)
        print(f"  Found {len(face_detections)} faces")
        
        if len(face_detections) == 0:
            print("Warning: No faces detected. Returning empty results.")
            return self._create_empty_results(video_path, width, height, fps, total_frames)
        
        # Step 2: Track faces through video with SAM3
        tracked_faces, video_size, video_fps = self.tracker.process_video(
            video_path=video_path,
            face_detections=face_detections,
            max_frames=max_frames,
        )
        
        # Step 3: Estimate gaze for each tracked face in each frame
        print("Estimating gaze for each face...")
        cap = cv2.VideoCapture(video_path)
        
        results = {
            "video_path": video_path,
            "video_fps": video_fps,
            "video_size": list(video_size),
            "total_frames": total_frames,
            "processed_frames": len(tracked_faces),
            "num_faces_detected": len(face_detections),
            "persons_summary": {},
            "frames": [],
        }
        
        person_frames = {}  # Track which frames each person appears in
        
        frame_indices = sorted(tracked_faces.keys())
        pbar = tqdm(frame_indices, desc="Estimating gaze", disable=not progress_bar)
        
        for frame_idx in pbar:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame_bgr = cap.read()
            if not ret:
                continue
            
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            
            frame_data = {
                "frame_idx": frame_idx,
                "timestamp": frame_idx / video_fps if video_fps > 0 else 0,
                "persons": [],
            }
            
            for face in tracked_faces[frame_idx]:
                # Create position prompt from face bbox center
                position_prompt = GazePredictor.format_position_prompt(face.bbox)
                
                # Estimate gaze
                gaze_result = self.gaze_predictor.predict(frame_rgb, position_prompt)
                
                # Store result
                person_data = {
                    "person_id": face.face_id,
                    "face_bbox": face.bbox,
                    "face_bbox_px": face.bbox_px,
                    "detection_score": float(face.score),
                    "head_bbox": gaze_result["head_bbox"],
                    "head_bbox_px": gaze_result["head_bbox_px"],
                    "gaze_point": gaze_result["gaze_point"],
                    "gaze_point_px": gaze_result["gaze_point_px"],
                    "inout": gaze_result["inout"],
                }
                
                frame_data["persons"].append(person_data)
                
                # Track person frames
                if face.face_id not in person_frames:
                    person_frames[face.face_id] = []
                person_frames[face.face_id].append(frame_idx)
            
            results["frames"].append(frame_data)
        
        cap.release()
        
        # Add person summary
        for person_id, frames in person_frames.items():
            # Calculate in-frame gaze percentage
            inframe_count = sum(
                1 for fd in results["frames"]
                for p in fd["persons"]
                if p["person_id"] == person_id and p["inout"]
            )
            
            results["persons_summary"][str(person_id)] = {
                "first_frame": min(frames),
                "last_frame": max(frames),
                "track_length": len(frames),
                "inframe_gaze_pct": round(100 * inframe_count / len(frames), 1) if frames else 0,
            }
        
        return results
    
    def _create_empty_results(self, video_path, width, height, fps, total_frames):
        """Create empty results when no faces detected."""
        return {
            "video_path": video_path,
            "video_fps": fps,
            "video_size": [width, height],
            "total_frames": total_frames,
            "processed_frames": 0,
            "num_faces_detected": 0,
            "persons_summary": {},
            "frames": [],
        }
    
    def save_results(self, results: Dict, output_path: str) -> None:
        """Save annotation results to JSON file."""
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.floating):
                    return float(obj)
                if isinstance(obj, np.integer):
                    return int(obj)
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super().default(obj)
        
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, cls=NumpyEncoder)
        print(f"Saved results to: {output_path}")


# ============================================================================
# Visualization
# ============================================================================

def visualize_annotations(
    video_path: str,
    results: Dict,
    output_path: str,
    show_face_bbox: bool = True,
    show_head_bbox: bool = True,
    show_gaze: bool = True,
    progress_bar: bool = True,
) -> None:
    """
    Create a visualization video with annotations overlaid.
    """
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Color palette
    colors = [
        (0, 255, 0),    # Green
        (255, 128, 0),  # Blue-ish
        (0, 128, 255),  # Orange
        (255, 255, 0),  # Cyan
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Yellow
        (128, 0, 255),  # Purple
        (255, 0, 128),  # Pink
    ]
    out_color = (0, 0, 255)  # Red for "OUT"
    
    pbar = tqdm(results["frames"], desc="Creating visualization", disable=not progress_bar)
    
    for frame_data in pbar:
        frame_idx = frame_data["frame_idx"]
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if not ret:
            continue
        
        for person in frame_data["persons"]:
            person_id = person["person_id"]
            color = colors[person_id % len(colors)]
            inout = person["inout"]
            
            # Draw face bbox
            if show_face_bbox:
                x1, y1, x2, y2 = person["face_bbox_px"]
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
                
                # Label with colored background
                label = f"Face {person_id}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
                cv2.putText(frame, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
            
            # Draw head bbox
            if show_head_bbox:
                hx1, hy1, hx2, hy2 = [int(v) for v in person["head_bbox_px"]]
                cv2.rectangle(frame, (hx1, hy1), (hx2, hy2), color, 3, cv2.LINE_AA)
            
            # Draw gaze
            if show_gaze:
                hx1, hy1, hx2, hy2 = person["head_bbox_px"]
                hcx, hcy = int((hx1 + hx2) / 2), int((hy1 + hy2) / 2)
                
                if inout:
                    gx, gy = int(person["gaze_point_px"][0]), int(person["gaze_point_px"][1])
                    cv2.line(frame, (hcx, hcy), (gx, gy), color, 2, cv2.LINE_AA)
                    cv2.circle(frame, (gx, gy), 12, color, -1, cv2.LINE_AA)
                    cv2.circle(frame, (gx, gy), 12, (255, 255, 255), 2, cv2.LINE_AA)
                else:
                    cv2.putText(frame, "OUT", (hcx - 20, hcy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, out_color, 2, cv2.LINE_AA)
        
        # Frame info
        info_text = f"Frame: {frame_idx} | t={frame_data['timestamp']:.2f}s | {len(frame_data['persons'])} face(s)"
        (tw, th), _ = cv2.getTextSize(info_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(frame, (5, 5), (15 + tw, 35), (0, 0, 0), -1)
        cv2.putText(frame, info_text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        
        out.write(frame)
    
    cap.release()
    out.release()
    
    # Re-encode with H.264 for compatibility
    _reencode_video_h264(output_path)
    
    print(f"Saved visualization to: {output_path}")


def _reencode_video_h264(video_path: str) -> None:
    """Re-encode video with H.264 codec for better compatibility."""
    import subprocess
    import shutil
    
    if shutil.which("ffmpeg") is None:
        print("Warning: ffmpeg not found, skipping H.264 re-encoding.")
        return
    
    temp_path = video_path + ".temp.mp4"
    
    try:
        os.rename(video_path, temp_path)
        
        cmd = [
            "ffmpeg", "-y",
            "-i", temp_path,
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "23",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            "-loglevel", "error",
            video_path
        ]
        
        subprocess.run(cmd, check=True)
        os.remove(temp_path)
        
    except Exception as e:
        print(f"Warning: H.264 re-encoding failed: {e}")
        if os.path.exists(temp_path) and not os.path.exists(video_path):
            os.rename(temp_path, video_path)


# ============================================================================
# Main
# ============================================================================

DEFAULT_OUTPUT_DIR = "/u/arkimjh/code/ECCV-jh/data_results"


def main():
    parser = argparse.ArgumentParser(description="RetinaFace + SAM3 + GazeAnywhere Pipeline")
    parser.add_argument("--video_path", type=str, required=True, help="Path to input video")
    parser.add_argument("--output_json", type=str, default=None, help="Output JSON path")
    parser.add_argument("--output_video", type=str, default=None, help="Output visualization video path")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Output directory")
    parser.add_argument("--max_frames", type=int, default=None, help="Max frames to process")
    parser.add_argument("--det_thresh", type=float, default=0.5, help="Face detection threshold")
    parser.add_argument("--no_visualization", action="store_true", help="Skip visualization")
    parser.add_argument("--gaze_checkpoint", type=str, 
                        default="/projects/illinois/eng/cs/jrehg/users/xucao2/ChildGaze/checkpoints/GazeAnywhere/gazeanywhere.pth",
                        help="Path to GazeAnywhere checkpoint")
    
    args = parser.parse_args()
    
    # Auto-generate output paths
    video_name = os.path.splitext(os.path.basename(args.video_path))[0]
    
    if args.output_json is None:
        args.output_json = os.path.join(args.output_dir, f"{video_name}_retinaface_gaze.json")
    
    if args.output_video is None:
        args.output_video = os.path.join(args.output_dir, f"{video_name}_retinaface_viz.mp4")
    
    # Initialize annotator
    print("Initializing models...")
    annotator = RetinaFaceGazeAnnotator(
        device="cuda" if torch.cuda.is_available() else "cpu",
        gaze_checkpoint=args.gaze_checkpoint,
        det_thresh=args.det_thresh,
    )
    
    # Process video
    print(f"Processing: {args.video_path}")
    results = annotator.annotate_video(
        video_path=args.video_path,
        max_frames=args.max_frames,
    )
    
    # Save results
    annotator.save_results(results, args.output_json)
    
    # Print summary
    print("\n" + "=" * 50)
    print("Annotation Summary")
    print("=" * 50)
    print(f"  Video: {args.video_path}")
    print(f"  Size: {results['video_size'][0]}x{results['video_size'][1]}")
    print(f"  FPS: {results['video_fps']:.2f}")
    print(f"  Total frames: {results['total_frames']}")
    print(f"  Processed frames: {results['processed_frames']}")
    print(f"  Faces detected: {results['num_faces_detected']}")
    
    for person_id, summary in results["persons_summary"].items():
        print(f"    Face {person_id}: frames {summary['first_frame']}-{summary['last_frame']} "
              f"({summary['track_length']} frames, {summary['inframe_gaze_pct']}% in-frame gaze)")
    
    print(f"\nOutput JSON: {args.output_json}")
    
    # Create visualization
    if not args.no_visualization and results["processed_frames"] > 0:
        print("\nCreating visualization...")
        visualize_annotations(
            video_path=args.video_path,
            results=results,
            output_path=args.output_video,
        )
        print(f"Output video: {args.output_video}")


if __name__ == "__main__":
    main()
