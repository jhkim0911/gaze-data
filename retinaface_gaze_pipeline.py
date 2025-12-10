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
    bbox: Tuple[float, float, float, float]  # (x1, y1, x2, y2) normalized - from SAM3 mask
    bbox_px: Tuple[int, int, int, int]  # (x1, y1, x2, y2) pixels - from SAM3 mask
    score: float
    retinaface_bbox_px: Optional[Tuple[int, int, int, int]] = None  # Original RetinaFace bbox
    retinaface_bbox: Optional[Tuple[float, float, float, float]] = None  # Original RetinaFace bbox normalized
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
# SAM3 Bounding Box Tracker (HuggingFace Transformers API)
# ============================================================================

class Sam3BboxTracker:
    """
    SAM3 video tracker using bounding box prompts.
    
    Uses HuggingFace Transformers Sam3TrackerVideoModel/Processor which supports
    multi-object tracking with points or boxes.
    """
    
    def __init__(self, device: str = "cuda"):
        """
        Initialize SAM3 tracker.
        
        Args:
            device: Device for inference
        """
        self.device = device
        self._model = None
        self._processor = None
        self._inference_session = None
    
    def _ensure_loaded(self):
        """Lazy-load SAM3 tracker via HuggingFace Transformers."""
        if self._model is None:
            try:
                from transformers import Sam3TrackerVideoModel, Sam3TrackerVideoProcessor
                import torch
                
                self._model = Sam3TrackerVideoModel.from_pretrained("facebook/sam3")
                self._model = self._model.to(self.device, dtype=torch.bfloat16)
                self._processor = Sam3TrackerVideoProcessor.from_pretrained("facebook/sam3")
                
                print("SAM3 tracker (HuggingFace Transformers) loaded successfully")
            except ImportError as e:
                raise ImportError(
                    "transformers package doesn't have Sam3TrackerVideoModel. "
                    "Try: pip install --upgrade transformers"
                ) from e
    
    def init_session(self, video_frames: List[np.ndarray]) -> None:
        """
        Initialize inference session with video frames.
        
        Args:
            video_frames: List of RGB frames as numpy arrays
        """
        self._ensure_loaded()
        import torch
        
        self._inference_session = self._processor.init_video_session(
            video=video_frames,
            inference_device=self.device,
            dtype=torch.bfloat16,
        )
    
    def add_face_prompts(
        self, 
        face_detections: List[FaceDetection], 
        frame_idx: int = 0,
        video_size: Tuple[int, int] = None,
    ) -> None:
        """
        Add face bounding boxes as tracking prompts.
        
        Uses actual bounding boxes from RetinaFace to prompt SAM3 to track faces
        (not whole persons).
        
        Args:
            face_detections: List of face detections from RetinaFace
            frame_idx: Frame index where faces were detected
            video_size: (width, height) of the video
        """
        if self._inference_session is None:
            raise RuntimeError("No active session. Call init_session() first.")
        
        if len(face_detections) == 0:
            return
        
        # Store face detections for later retrieval during tracking
        self._face_detections = {face.face_id: face for face in face_detections}
        
        # Build bounding boxes for all faces
        obj_ids = [face.face_id for face in face_detections]
        
        # Input boxes format: [[[x1, y1, x2, y2], [x1, y1, x2, y2], ...]] - 3 levels: image, box, coords
        input_boxes = [[]]
        
        for face in face_detections:
            x1, y1, x2, y2 = face.bbox_px
            input_boxes[0].append([x1, y1, x2, y2])
        
        # Add all object prompts using bounding boxes
        self._processor.add_inputs_to_inference_session(
            inference_session=self._inference_session,
            frame_idx=frame_idx,
            obj_ids=obj_ids,
            input_boxes=input_boxes,
        )
        
        # Run inference on the first frame (required before propagation)
        outputs = self._model(
            inference_session=self._inference_session,
            frame_idx=frame_idx,
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
        if self._inference_session is None:
            raise RuntimeError("No active session. Call init_session() first.")
        
        results = {}
        
        video_height = self._inference_session.video_height
        video_width = self._inference_session.video_width
        
        # Propagate through video
        for sam3_output in self._model.propagate_in_video_iterator(self._inference_session):
            frame_idx = sam3_output.frame_idx
            
            # Get masks at original resolution
            video_res_masks = self._processor.post_process_masks(
                [sam3_output.pred_masks], 
                original_sizes=[[video_height, video_width]], 
                binarize=True
            )[0]
            
            tracked_faces = []
            obj_ids = self._inference_session.obj_ids
            
            for i, obj_id in enumerate(obj_ids):
                mask = video_res_masks[i].cpu().numpy().squeeze()
                
                # Get bbox from mask
                if mask.any():
                    ys, xs = np.where(mask > 0.5)
                    if len(xs) > 0 and len(ys) > 0:
                        x1, x2 = xs.min(), xs.max()
                        y1, y2 = ys.min(), ys.max()
                        
                        bbox_px = (int(x1), int(y1), int(x2), int(y2))
                        bbox_norm = (
                            x1 / video_width,
                            y1 / video_height,
                            x2 / video_width,
                            y2 / video_height,
                        )
                        
                        # Get original RetinaFace bbox if available
                        orig_face = self._face_detections.get(int(obj_id)) if hasattr(self, '_face_detections') else None
                        
                        tracked_faces.append(TrackedFace(
                            face_id=int(obj_id),
                            bbox=bbox_norm,
                            bbox_px=bbox_px,
                            score=1.0,
                            retinaface_bbox_px=orig_face.bbox_px if orig_face else None,
                            retinaface_bbox=orig_face.bbox if orig_face else None,
                            mask=mask,
                        ))
            
            results[frame_idx] = tracked_faces
            
            if max_frames and len(results) >= max_frames:
                break
        
        return results
    
    def reset_state(self):
        """Reset the inference session."""
        if self._inference_session is not None:
            self._inference_session.reset_inference_session()
        self._inference_session = None
    
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
        # Load video frames
        cap = cv2.VideoCapture(video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Read all frames
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
            if max_frames and len(frames) >= max_frames:
                break
        cap.release()
        
        print(f"Loaded {len(frames)} frames")
        
        try:
            # Initialize session with video frames
            self.init_session(frames)
            
            # Add face prompts from first frame
            print(f"Adding {len(face_detections)} face prompts...")
            self.add_face_prompts(face_detections, frame_idx=0, video_size=(width, height))
            
            # Propagate through video
            print(f"Tracking faces through video ({len(frames)} frames)...")
            tracked_faces = self.propagate_tracking(max_frames=max_frames)
            
            return tracked_faces, (width, height), fps
            
        finally:
            self.reset_state()



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
                    "face_bbox": face.bbox,  # SAM3 tracked bbox (changes per frame)
                    "face_bbox_px": face.bbox_px,  # SAM3 tracked bbox in pixels
                    "retinaface_bbox": face.retinaface_bbox,  # Original RetinaFace detection (constant)
                    "retinaface_bbox_px": face.retinaface_bbox_px,  # Original RetinaFace detection in pixels
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
                import torch
                if isinstance(obj, torch.Tensor):
                    return obj.cpu().numpy().tolist()
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
    show_face_bbox: bool = True,  # SAM3 tracked bbox
    show_retinaface_bbox: bool = True,  # Original RetinaFace detection bbox
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
    
    # Color palette for per-person elements
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
    retinaface_color = (255, 0, 200)  # Magenta/Hot Pink for RetinaFace bbox
    
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
            
            # Draw RetinaFace detection bbox (magenta, constant across frames)
            if show_retinaface_bbox and person.get("retinaface_bbox_px"):
                rx1, ry1, rx2, ry2 = person["retinaface_bbox_px"]
                cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), retinaface_color, 2, cv2.LINE_AA)
                
                # Small label
                label = f"RF{person_id}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(frame, (rx1, ry2), (rx1 + tw + 4, ry2 + th + 6), retinaface_color, -1)
                cv2.putText(frame, label, (rx1 + 2, ry2 + th + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            
            # Draw SAM3 tracked face bbox (per-person color, changes per frame)
            if show_face_bbox:
                x1, y1, x2, y2 = person["face_bbox_px"]
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
                
                # Label with colored background
                label = f"SAM{person_id}"
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
