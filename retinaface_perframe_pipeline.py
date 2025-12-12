#!/usr/bin/env python3
"""
RetinaFace Per-Frame Detection + GazeAnywhere Multi-Person Gaze Estimation Pipeline

This pipeline uses:
1. RetinaFace (via insightface) for face detection on EVERY frame
2. Hungarian algorithm for face ID association across frames
3. GazeAnywhere for per-person gaze estimation

This approach provides more reliable face tracking than SAM3 for gaze estimation
since RetinaFace is specifically optimized for face detection.

Usage:
    python retinaface_perframe_pipeline.py --video_path /path/to/video.mp4
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment


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
    embedding: Optional[np.ndarray] = None  # Face embedding for re-identification


@dataclass
class TrackedFace:
    """Tracked face result with persistent ID."""
    face_id: int  # Persistent ID across frames
    bbox: Tuple[float, float, float, float]  # (x1, y1, x2, y2) normalized
    bbox_px: Tuple[int, int, int, int]  # (x1, y1, x2, y2) pixels
    score: float
    landmarks: Optional[np.ndarray] = None
    embedding: Optional[np.ndarray] = None
    frames_since_seen: int = 0  # For track management


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
    
    RetinaFace provides high-accuracy face detection with 5-point landmarks
    and optional face embeddings for re-identification.
    """
    
    def __init__(
        self, 
        device: str = "cuda", 
        det_thresh: float = 0.5,
        use_embeddings: bool = True,
    ):
        """
        Initialize RetinaFace detector.
        
        Args:
            device: "cuda" or "cpu"
            det_thresh: Detection confidence threshold
            use_embeddings: Whether to extract face embeddings for better association
        """
        self.device = device
        self.det_thresh = det_thresh
        self.use_embeddings = use_embeddings
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
            
            # Get embedding if available and requested
            embedding = None
            if self.use_embeddings and hasattr(face, 'embedding') and face.embedding is not None:
                embedding = face.embedding
            
            detections.append(FaceDetection(
                face_id=i,  # Temporary ID, will be reassigned by tracker
                bbox=bbox_norm,
                bbox_px=bbox_px,
                score=float(face.det_score),
                landmarks=face.kps if hasattr(face, 'kps') else None,
                embedding=embedding,
            ))
        
        return detections


# ============================================================================
# Face Tracker with Hungarian Algorithm Association
# ============================================================================

class FaceTracker:
    """
    Multi-face tracker using Hungarian algorithm for ID association.
    
    Uses a combination of:
    1. IoU (Intersection over Union) for spatial matching
    2. Face embeddings (when available) for appearance matching
    """
    
    def __init__(
        self,
        iou_threshold: float = 0.3,
        embedding_threshold: float = 0.7,
        max_frames_missing: int = 999999,  # Effectively infinite - tracks never deleted
        embedding_weight: float = 0.4,
    ):
        """
        Initialize tracker.
        
        Args:
            iou_threshold: Minimum IoU for matching
            embedding_threshold: Maximum embedding distance for matching
            max_frames_missing: Maximum frames a track can be missing before deletion
            embedding_weight: Weight for embedding similarity (0-1), IoU weight = 1 - embedding_weight
        """
        self.iou_threshold = iou_threshold
        self.embedding_threshold = embedding_threshold
        self.max_frames_missing = max_frames_missing
        self.embedding_weight = embedding_weight
        
        self.tracks: Dict[int, TrackedFace] = {}  # Active tracks
        self.graveyard: Dict[int, np.ndarray] = {}  # Deleted tracks' embeddings for re-identification
        self.next_id = 0
        self.frame_count = 0
    
    def reset(self):
        """Reset tracker state."""
        self.tracks = {}
        self.graveyard = {}
        self.next_id = 0
        self.frame_count = 0
    
    @staticmethod
    def compute_iou(box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> float:
        """Compute IoU between two bounding boxes (pixel coordinates)."""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Intersection
        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)
        
        if xi2 <= xi1 or yi2 <= yi1:
            return 0.0
        
        inter_area = (xi2 - xi1) * (yi2 - yi1)
        
        # Union
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = box1_area + box2_area - inter_area
        
        if union_area <= 0:
            return 0.0
        
        return inter_area / union_area
    
    @staticmethod
    def compute_embedding_distance(emb1: Optional[np.ndarray], emb2: Optional[np.ndarray]) -> float:
        """Compute cosine distance between two embeddings."""
        if emb1 is None or emb2 is None:
            return 1.0  # Maximum distance if embeddings not available
        
        # Normalize
        emb1_norm = emb1 / (np.linalg.norm(emb1) + 1e-8)
        emb2_norm = emb2 / (np.linalg.norm(emb2) + 1e-8)
        
        # Cosine similarity -> distance
        similarity = np.dot(emb1_norm, emb2_norm)
        distance = 1.0 - similarity
        
        return float(distance)
    
    def update(self, detections: List[FaceDetection]) -> List[TrackedFace]:
        """
        Update tracks with new detections.
        
        Args:
            detections: List of face detections from current frame
            
        Returns:
            List of tracked faces with persistent IDs
        """
        self.frame_count += 1
        
        # Increment frames_since_seen for all tracks
        for track_id in self.tracks:
            self.tracks[track_id].frames_since_seen += 1
        
        if len(detections) == 0:
            # No detections - remove stale tracks and return empty
            self._remove_stale_tracks()
            return []
        
        if len(self.tracks) == 0:
            # No existing tracks - create new ones for all detections
            tracked_faces = []
            for det in detections:
                track = self._create_track(det)
                tracked_faces.append(track)
            return tracked_faces
        
        # Build cost matrix for Hungarian algorithm
        track_ids = list(self.tracks.keys())
        cost_matrix = np.zeros((len(track_ids), len(detections)))
        
        for i, track_id in enumerate(track_ids):
            track = self.tracks[track_id]
            for j, det in enumerate(detections):
                # Compute IoU cost (1 - IoU so lower is better)
                iou = self.compute_iou(track.bbox_px, det.bbox_px)
                iou_cost = 1.0 - iou
                
                # Compute embedding cost
                emb_dist = self.compute_embedding_distance(track.embedding, det.embedding)
                
                # Combined cost
                if track.embedding is not None and det.embedding is not None:
                    cost = (1 - self.embedding_weight) * iou_cost + self.embedding_weight * emb_dist
                else:
                    cost = iou_cost
                
                cost_matrix[i, j] = cost
        
        # Hungarian algorithm for optimal assignment
        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        
        matched_tracks: Set[int] = set()
        matched_detections: Set[int] = set()
        tracked_faces = []
        
        # Process matches
        for row_idx, col_idx in zip(row_indices, col_indices):
            track_id = track_ids[row_idx]
            det = detections[col_idx]
            track = self.tracks[track_id]
            
            # Check if match is valid (IoU above threshold)
            iou = self.compute_iou(track.bbox_px, det.bbox_px)
            emb_dist = self.compute_embedding_distance(track.embedding, det.embedding)
            
            # Accept match if IoU is good OR embedding is very similar
            if iou >= self.iou_threshold or (det.embedding is not None and emb_dist < self.embedding_threshold):
                # Update track
                track.bbox = det.bbox
                track.bbox_px = det.bbox_px
                track.score = det.score
                track.landmarks = det.landmarks
                track.frames_since_seen = 0
                
                # Update embedding with exponential moving average
                if det.embedding is not None:
                    if track.embedding is not None:
                        track.embedding = 0.7 * track.embedding + 0.3 * det.embedding
                    else:
                        track.embedding = det.embedding
                
                matched_tracks.add(track_id)
                matched_detections.add(col_idx)
                tracked_faces.append(track)
        
        # Create new tracks for unmatched detections
        for j, det in enumerate(detections):
            if j not in matched_detections:
                track = self._create_track(det)
                tracked_faces.append(track)
        
        # Remove stale tracks
        self._remove_stale_tracks()
        
        # Sort by ID for consistent ordering
        tracked_faces.sort(key=lambda t: t.face_id)
        
        return tracked_faces
    
    def _create_track(self, detection: FaceDetection) -> TrackedFace:
        """Create a new track from a detection, or reuse old ID if embedding matches graveyard."""
        
        # Check if this face matches any in the graveyard (previously deleted tracks)
        reuse_id = None
        if detection.embedding is not None and len(self.graveyard) > 0:
            best_match_id = None
            best_match_dist = float('inf')
            
            for old_id, old_embedding in self.graveyard.items():
                dist = self.compute_embedding_distance(detection.embedding, old_embedding)
                if dist < best_match_dist:
                    best_match_dist = dist
                    best_match_id = old_id
            
            # If best match is below threshold, reuse that ID
            if best_match_dist < self.embedding_threshold:
                reuse_id = best_match_id
                # Remove from graveyard since it's active again
                del self.graveyard[best_match_id]
        
        # Use reused ID or create new one
        track_id = reuse_id if reuse_id is not None else self.next_id
        if reuse_id is None:
            self.next_id += 1
        
        track = TrackedFace(
            face_id=track_id,
            bbox=detection.bbox,
            bbox_px=detection.bbox_px,
            score=detection.score,
            landmarks=detection.landmarks,
            embedding=detection.embedding,
            frames_since_seen=0,
        )
        self.tracks[track_id] = track
        return track
    
    def _remove_stale_tracks(self):
        """Remove tracks that haven't been seen for too long and save to graveyard."""
        stale_ids = [
            track_id for track_id, track in self.tracks.items()
            if track.frames_since_seen > self.max_frames_missing
        ]
        for track_id in stale_ids:
            track = self.tracks[track_id]
            # Save embedding to graveyard for potential re-identification
            if track.embedding is not None:
                self.graveyard[track_id] = track.embedding.copy()
            del self.tracks[track_id]


# ============================================================================
# SAM3 Per-Frame Segmenter
# ============================================================================

class Sam3PerFrameSegmenter:
    """
    SAM3 segmenter for per-frame head segmentation.
    
    Uses SAM3 to segment heads given face bounding box prompts on each frame.
    This gives more accurate head bboxes than simple face bbox expansion.
    """
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        self._model = None
        self._processor = None
    
    def _ensure_loaded(self):
        """Lazy-load SAM3 tracker via HuggingFace Transformers."""
        if self._model is None:
            try:
                from transformers import Sam3TrackerProcessor, Sam3TrackerModel
                
                self._model = Sam3TrackerModel.from_pretrained("facebook/sam3")
                self._model = self._model.to(self.device, dtype=torch.bfloat16)
                self._processor = Sam3TrackerProcessor.from_pretrained("facebook/sam3")
                
                print("SAM3 per-frame segmenter loaded successfully")
            except ImportError as e:
                raise ImportError(
                    "transformers package doesn't have Sam3TrackerModel. "
                    "Try: pip install --upgrade transformers"
                ) from e
    
    def segment_faces(
        self,
        frame_rgb: np.ndarray,
        face_bboxes_px: List[Tuple[int, int, int, int]],
    ) -> List[Tuple[Tuple[int, int, int, int], np.ndarray]]:
        """
        Segment faces/heads on a single frame using SAM3 with bbox prompts.
        
        Args:
            frame_rgb: RGB frame as numpy array
            face_bboxes_px: List of face bounding boxes in pixel coords [(x1,y1,x2,y2), ...]
            
        Returns:
            List of (head_bbox_px, mask) tuples for each input face
        """
        self._ensure_loaded()
        
        if len(face_bboxes_px) == 0:
            return []
        
        height, width = frame_rgb.shape[:2]
        
        # Format input boxes: [[[x1, y1, x2, y2], ...]] - 3 levels for batch/object/coords
        input_boxes = [[list(bbox) for bbox in face_bboxes_px]]
        
        # Process with SAM3 Tracker (single image mode)
        inputs = self._processor(
            images=Image.fromarray(frame_rgb),
            input_boxes=input_boxes,
            return_tensors="pt"
        ).to(self._model.device, dtype=torch.bfloat16)
        
        with torch.no_grad():
            outputs = self._model(**inputs, multimask_output=False)
        
        # Post-process masks
        masks = self._processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"]
        )[0]  # [num_objects, 1, H, W]
        
        results = []
        for i in range(len(face_bboxes_px)):
            mask = masks[i].squeeze().numpy()
            
            # Get bbox from mask
            if mask.any():
                ys, xs = np.where(mask > 0.5)
                if len(xs) > 0 and len(ys) > 0:
                    x1, x2 = int(xs.min()), int(xs.max())
                    y1, y2 = int(ys.min()), int(ys.max())
                    head_bbox_px = (x1, y1, x2, y2)
                else:
                    head_bbox_px = face_bboxes_px[i]
            else:
                head_bbox_px = face_bboxes_px[i]
            
            results.append((head_bbox_px, mask))
        
        return results


# ============================================================================
# GazeAnywhere Integration
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

class RetinaFacePerFrameAnnotator:
    """
    Full pipeline: RetinaFace per-frame detection → SAM3 per-frame segmentation → GazeAnywhere
    
    On every frame:
    1. RetinaFace detects faces
    2. SAM3 segments heads using face bboxes as prompts
    3. GazeAnywhere estimates gaze
    """
    
    def __init__(
        self,
        device: str = "cuda",
        gaze_checkpoint: str = "/projects/illinois/eng/cs/jrehg/users/xucao2/ChildGaze/checkpoints/GazeAnywhere/gazeanywhere.pth",
        det_thresh: float = 0.5,
        use_embeddings: bool = True,
        iou_threshold: float = 0.3,
        max_frames_missing: int = 30,
    ):
        self.device = device
        self.face_detector = RetinaFaceDetector(
            device=device, 
            det_thresh=det_thresh,
            use_embeddings=use_embeddings,
        )
        self.tracker = FaceTracker(
            iou_threshold=iou_threshold,
            max_frames_missing=max_frames_missing,
        )
        self.sam3_segmenter = Sam3PerFrameSegmenter(device=device)
        self.gaze_predictor = GazePredictor(checkpoint_path=gaze_checkpoint, device=device)
    
    def reset_tracker(self):
        """Reset tracker state between videos for batch processing."""
        self.tracker.reset()
    
    def annotate_video(
        self,
        video_path: str,
        max_frames: Optional[int] = None,
        progress_bar: bool = True,
        sample_fps: Optional[float] = None,  # Process at this fps (e.g., 2.0 = 2fps)
    ) -> Dict:
        """
        Annotate a video with face detection, tracking, and gaze estimation.
        
        Args:
            video_path: Path to input video
            max_frames: Maximum frames to process (None = all)
            progress_bar: Show progress bar
            sample_fps: Sample video at this fps (e.g., 2.0 = process 2 frames per second)
                       If None, uses original video fps
            
        Returns:
            Dict with video metadata and per-frame annotations
        """
        # Get video info
        cap = cv2.VideoCapture(video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_duration = total_frames / original_fps if original_fps > 0 else 0
        
        # Calculate frames to process using time-based sampling
        if sample_fps and sample_fps < original_fps:
            # Sample at exact fps using time-based calculation
            sample_interval_sec = 1.0 / sample_fps  # e.g., 0.5 sec for 2fps
            frames_to_process = []
            t = 0.0
            while t < video_duration:
                frame_idx = int(t * original_fps)
                if frame_idx < total_frames:
                    frames_to_process.append(frame_idx)
                t += sample_interval_sec
            effective_fps = sample_fps
        else:
            frames_to_process = list(range(total_frames))
            effective_fps = original_fps
        
        if max_frames:
            frames_to_process = frames_to_process[:max_frames]
        
        results = {
            "video_path": video_path,
            "video_fps": original_fps,
            "sample_fps": effective_fps,
            "video_size": [width, height],
            "total_frames": total_frames,
            "processed_frames": 0,
            "num_unique_faces": 0,
            "tracking_method": "retinaface_per_frame",
            "persons_summary": {},
            "frames": [],
        }
        
        # Reset tracker
        self.tracker.reset()
        
        person_frames = {}  # Track which frames each person appears in
        all_face_ids: Set[int] = set()
        
        print(f"Processing video: {video_path}")
        print(f"  Resolution: {width}x{height}, FPS: {original_fps:.2f}, Frames: {total_frames}")
        if effective_fps != original_fps:
            print(f"  Sampling at {effective_fps:.1f}fps, processing {len(frames_to_process)} frames")
        
        pbar = tqdm(frames_to_process, desc="Processing frames", disable=not progress_bar)
        
        for frame_idx in pbar:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame_bgr = cap.read()
            if not ret:
                break
            
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            
            # Step 1: Detect faces with RetinaFace
            detections = self.face_detector.detect_faces(frame_bgr)
            
            # Update tracker for ID association
            tracked_faces = self.tracker.update(detections)
            
            # Step 2: Segment all faces with SAM3
            sam3_results = {}
            if len(tracked_faces) > 0:
                face_bboxes = [face.bbox_px for face in tracked_faces]
                sam3_outputs = self.sam3_segmenter.segment_faces(frame_rgb, face_bboxes)
                for i, face in enumerate(tracked_faces):
                    sam3_results[face.face_id] = sam3_outputs[i]  # (head_bbox_px, mask)
            
            # Prepare frame data
            frame_data = {
                "frame_idx": frame_idx,
                "timestamp": frame_idx / original_fps if original_fps > 0 else 0,
                "persons": [],
            }
            
            # Step 3: Estimate gaze for each tracked face
            for face in tracked_faces:
                all_face_ids.add(face.face_id)
                
                # Create position prompt from face bbox center
                position_prompt = GazePredictor.format_position_prompt(face.bbox)
                
                # Estimate gaze
                gaze_result = self.gaze_predictor.predict(frame_rgb, position_prompt)
                
                # Get SAM3 head bbox
                sam3_head_bbox_px, sam3_mask = sam3_results[face.face_id]
                final_head_bbox_px = tuple(float(v) for v in sam3_head_bbox_px)
                final_head_bbox = (
                    sam3_head_bbox_px[0] / width,
                    sam3_head_bbox_px[1] / height,
                    sam3_head_bbox_px[2] / width,
                    sam3_head_bbox_px[3] / height,
                )
                
                # Store result
                person_data = {
                    "person_id": face.face_id,
                    "face_bbox": face.bbox,
                    "face_bbox_px": face.bbox_px,
                    "detection_score": float(face.score),
                    "head_bbox": final_head_bbox,
                    "head_bbox_px": final_head_bbox_px,
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
            
            # Update progress bar description
            pbar.set_postfix({"faces": len(tracked_faces), "total_ids": len(all_face_ids)})
        
        cap.release()
        
        # Finalize results
        results["processed_frames"] = len(results["frames"])
        results["num_unique_faces"] = len(all_face_ids)
        
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
    show_face_bbox: bool = True,
    show_head_bbox: bool = True,
    show_gaze: bool = True,
    show_track_id: bool = True,
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
                
                if show_track_id:
                    # Label with colored background
                    label = f"ID:{person_id}"
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
    parser = argparse.ArgumentParser(description="RetinaFace Per-Frame + GazeAnywhere Pipeline")
    parser.add_argument("--video_path", type=str, required=True, help="Path to input video")
    parser.add_argument("--output_json", type=str, default=None, help="Output JSON path")
    parser.add_argument("--output_video", type=str, default=None, help="Output visualization video path")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Output directory")
    parser.add_argument("--max_frames", type=int, default=None, help="Max frames to process")
    parser.add_argument("--det_thresh", type=float, default=0.6, help="Face detection threshold")
    parser.add_argument("--iou_threshold", type=float, default=0.3, help="IoU threshold for tracking")
    parser.add_argument("--max_frames_missing", type=int, default=30, help="Max frames a track can be missing")
    parser.add_argument("--no_embeddings", action="store_true", help="Disable face embeddings for tracking")
    parser.add_argument("--no_visualization", action="store_true", help="Skip visualization")

    parser.add_argument("--gaze_checkpoint", type=str, 
                        default="/projects/illinois/eng/cs/jrehg/users/xucao2/ChildGaze/checkpoints/GazeAnywhere/gazeanywhere.pth",
                        help="Path to GazeAnywhere checkpoint")
    
    args = parser.parse_args()
    
    # Auto-generate output paths
    video_name = os.path.splitext(os.path.basename(args.video_path))[0]
    
    if args.output_json is None:
        args.output_json = os.path.join(args.output_dir, f"{video_name}_perframe_gaze.json")
    
    if args.output_video is None:
        args.output_video = os.path.join(args.output_dir, f"{video_name}_perframe_viz.mp4")
    
    # Initialize annotator
    print("Initializing models...")
    annotator = RetinaFacePerFrameAnnotator(
        device="cuda" if torch.cuda.is_available() else "cpu",
        gaze_checkpoint=args.gaze_checkpoint,
        det_thresh=args.det_thresh,
        use_embeddings=not args.no_embeddings,
        iou_threshold=args.iou_threshold,
        max_frames_missing=args.max_frames_missing,

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
    print(f"  Unique faces tracked: {results['num_unique_faces']}")
    print(f"  Tracking method: {results['tracking_method']}")
    
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
