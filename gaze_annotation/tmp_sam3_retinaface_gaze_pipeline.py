#!/usr/bin/env python3
"""
SAM3 Person Tracking + RetinaFace Per-Frame + GazeAnywhere Pipeline

This hybrid pipeline combines:
1. SAM3 with text prompt "people" for consistent person ID tracking
2. RetinaFace per-frame for accurate face detection
3. GazeAnywhere per-frame for high-quality gaze estimation

This gives the best of both worlds:
- Consistent person IDs from SAM3 tracking (even when faces are temporarily occluded)
- High-quality gaze estimation from per-frame face detection

Usage:
    python sam3_retinaface_gaze_pipeline.py --video_path /path/to/video.mp4
"""

import argparse
import json
import os
import sys
import tempfile
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment

# Try to import decord for efficient video reading
try:
    import decord
    decord.bridge.set_bridge("torch")
    DECORD_AVAILABLE = True
except ImportError:
    DECORD_AVAILABLE = False


def extract_sampled_frames_to_dir(
    video_path: str,
    output_dir: str,
    sample_fps: float = 2.0,
    max_frames: Optional[int] = None,
) -> Tuple[List[int], float, int]:
    """
    Extract sampled frames from video to a directory.
    
    Uses decord for efficient reading - only loads frames we need.
    
    Args:
        video_path: Path to video file
        output_dir: Directory to save frame images
        sample_fps: Target frames per second to sample
        max_frames: Maximum number of frames to extract
        
    Returns:
        Tuple of (frame_indices, original_fps, total_original_frames)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if DECORD_AVAILABLE:
        # Use decord for efficient frame extraction
        vr = decord.VideoReader(video_path, num_threads=1)
        total_frames = len(vr)
        fps = vr.get_avg_fps()
        
        # Calculate which frames to sample
        frame_interval = fps / sample_fps
        frame_indices = []
        idx = 0.0
        while int(idx) < total_frames:
            frame_indices.append(int(idx))
            idx += frame_interval
            if max_frames and len(frame_indices) >= max_frames:
                break
        
        # Extract only the sampled frames (efficient!)
        print(f"  Extracting {len(frame_indices)} sampled frames (from {total_frames} total)...")
        frames = vr.get_batch(frame_indices)  # Only reads needed frames
        
        # Save as images
        for i, (frame_idx, frame) in enumerate(zip(frame_indices, frames)):
            # frame is (H, W, C) torch tensor
            frame_np = frame.numpy()
            img = Image.fromarray(frame_np)
            img.save(os.path.join(output_dir, f"{i:06d}.jpg"))
        
        del frames, vr
        
    else:
        # Fallback to cv2
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        frame_interval = fps / sample_fps
        frame_indices = []
        idx = 0.0
        while int(idx) < total_frames:
            frame_indices.append(int(idx))
            idx += frame_interval
            if max_frames and len(frame_indices) >= max_frames:
                break
        
        print(f"  Extracting {len(frame_indices)} sampled frames (from {total_frames} total)...")
        for i, frame_idx in enumerate(frame_indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                img.save(os.path.join(output_dir, f"{i:06d}.jpg"))
        
        cap.release()
    
    return frame_indices, fps, total_frames

# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class PersonTrack:
    """Person tracking result from SAM3."""
    person_id: int
    bbox: Tuple[float, float, float, float]  # (x1, y1, x2, y2) normalized
    bbox_px: Tuple[int, int, int, int]  # (x1, y1, x2, y2) pixels
    score: float
    mask: Optional[np.ndarray] = None


@dataclass
class FaceDetection:
    """Face detection result from RetinaFace."""
    bbox: Tuple[float, float, float, float]  # (x1, y1, x2, y2) normalized
    bbox_px: Tuple[int, int, int, int]  # (x1, y1, x2, y2) pixels
    score: float
    landmarks: Optional[np.ndarray] = None
    embedding: Optional[np.ndarray] = None  # 512-dim face embedding for re-ID


@dataclass
class PersonGazeResult:
    """Combined person tracking + gaze result."""
    person_id: int
    # From SAM3
    body_bbox: Tuple[float, float, float, float]
    body_bbox_px: Tuple[int, int, int, int]
    # From RetinaFace
    face_bbox: Optional[Tuple[float, float, float, float]]
    face_bbox_px: Optional[Tuple[int, int, int, int]]
    # From GazeAnywhere
    head_bbox: Optional[Tuple[float, float, float, float]]
    head_bbox_px: Optional[Tuple[float, float, float, float]]
    gaze_point: Optional[Tuple[float, float]]
    gaze_point_px: Optional[Tuple[float, float]]
    inout: Optional[bool]
    # Metadata
    face_detected: bool  # Whether face was found within person bbox


# ============================================================================
# SAM3 Person Tracker (Video Predictor)
# ============================================================================

class Sam3PersonTracker:
    """
    SAM3 video predictor for person tracking.
    
    Uses text prompt "people" to detect and track all persons with consistent IDs.
    """
    
    def __init__(self, device: str = "cuda", confidence_threshold: float = 0.5):
        self.device = device
        self.confidence_threshold = confidence_threshold
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
                    "SAM3 not installed. Install from: "
                    "git clone https://github.com/facebookresearch/sam3.git && "
                    "cd sam3 && pip install -e '.[notebooks]'"
                ) from e
    
    def start_session(self, video_path: str) -> str:
        """Start a video processing session."""
        self._ensure_loaded()
        response = self._predictor.start_session(
            resource_path=video_path,
            session_id=None,
        )
        self._session_id = response["session_id"]
        return self._session_id
    
    def detect_persons(self, frame_index: int = 0) -> Dict:
        """Add 'people' text prompt for initial detection."""
        if self._session_id is None:
            raise RuntimeError("No active session. Call start_session() first.")
        
        response = self._predictor.add_prompt(
            session_id=self._session_id,
            frame_idx=frame_index,
            text="people",
        )
        return response.get("outputs", {})
    
    def propagate_tracking(
        self, 
        max_frames: Optional[int] = None
    ) -> Dict[int, List[PersonTrack]]:
        """Propagate person tracking through video."""
        if self._session_id is None:
            raise RuntimeError("No active session. Call start_session() first.")
        
        results = {}
        
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
            
            tracks = []
            obj_ids = outputs.get("out_obj_ids", [])
            scores = outputs.get("out_probs", [])
            boxes_xywh = outputs.get("out_boxes_xywh", [])
            masks = outputs.get("out_binary_masks", None)
            
            for i, obj_id in enumerate(obj_ids):
                score = scores[i] if i < len(scores) else 1.0
                if score >= self.confidence_threshold:
                    if i < len(boxes_xywh):
                        x, y, w, h = boxes_xywh[i]
                        x1, y1 = x, y
                        x2, y2 = x + w, y + h
                        box_norm = (x1, y1, x2, y2)
                    else:
                        box_norm = (0, 0, 0, 0)
                    
                    tracks.append(PersonTrack(
                        person_id=int(obj_id),
                        bbox=box_norm,
                        bbox_px=(0, 0, 0, 0),  # Will compute later
                        score=float(score),
                        mask=None  # Don't store masks to save memory
                    ))
            
            results[frame_idx] = tracks
        
        return results
    
    def end_session(self):
        """End the current session."""
        if self._session_id is not None and self._predictor is not None:
            self._predictor.close_session(session_id=self._session_id)
            self._session_id = None
    
    def process_video(
        self,
        video_path: str,
        max_frames: Optional[int] = None,
        sample_fps: Optional[float] = None,
        use_presampling: bool = True,
        chunk_size: int = 125,  # Process 125 frames at a time (~62.5s at 2fps)
        overlap_frames: int = 5,  # Overlap between chunks for better ID matching
        points: list = None,  # GT point prompts [[x,y],...] normalized 0-1
    ) -> Tuple[Dict[int, List[PersonTrack]], Tuple[int, int], float, Optional[List[int]]]:
        """
        Full video processing with subprocess-based chunk approach.
        
        Each chunk runs in a subprocess to guarantee GPU memory is freed.
        """
        import subprocess
        import gc
        
        # Get video metadata
        cap = cv2.VideoCapture(video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        # Pre-extract sampled frames
        temp_dir = None
        original_frame_indices = None
        
        if use_presampling and sample_fps is not None:
            temp_dir = tempfile.mkdtemp(prefix="sam3_frames_")
            print(f"  Pre-extracting frames at {sample_fps}fps...")
            
            try:
                original_frame_indices, _, _ = extract_sampled_frames_to_dir(
                    video_path=video_path,
                    output_dir=temp_dir,
                    sample_fps=sample_fps,
                    max_frames=max_frames,
                )
                print(f"  Extracted {len(original_frame_indices)} frames")
            except Exception as e:
                print(f"  Warning: Pre-extraction failed ({e})")
                if temp_dir and os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)
                return {}, (width, height), fps, None
        else:
            temp_dir = tempfile.mkdtemp(prefix="sam3_frames_")
            # Extract all frames
            original_frame_indices = list(range(min(total_frames, max_frames or total_frames)))
            cap = cv2.VideoCapture(video_path)
            for i, frame_idx in enumerate(original_frame_indices):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if ret:
                    cv2.imwrite(os.path.join(temp_dir, f"{i:06d}.jpg"), frame)
            cap.release()
        
        num_frames = len(original_frame_indices)
        
        # Calculate chunks
        num_chunks = (num_frames + chunk_size - 1) // chunk_size
        print(f"  Processing in {num_chunks} chunks of ≤{chunk_size} frames (SUBPROCESS mode)...")
        
        all_tracks = {}
        global_id_map = {}
        next_global_id = 0
        prev_chunk_last_frame_tracks = None
        
        # Get the worker script path
        worker_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sam3_chunk_worker_points.py")
        
        try:
            for chunk_idx in range(num_chunks):
                # For chunks after the first, start with overlap_frames from previous chunk
                # This gives SAM3 context to establish tracking before new frames
                if chunk_idx == 0:
                    start_idx = 0
                    overlap_start = 0  # No overlap for first chunk
                else:
                    start_idx = chunk_idx * chunk_size - overlap_frames  # Start earlier with overlap
                    overlap_start = overlap_frames  # Skip first N frames in output
                    start_idx = max(0, start_idx)  # Don't go negative
                
                end_idx = min((chunk_idx + 1) * chunk_size, num_frames)
                chunk_frame_indices = list(range(start_idx, end_idx))
                
                print(f"  Chunk {chunk_idx+1}/{num_chunks}: frames {start_idx}-{end_idx-1} (overlap={overlap_start})")
                
                # Create temp dir for this chunk
                chunk_temp_dir = tempfile.mkdtemp(prefix=f"sam3_chunk{chunk_idx}_")
                chunk_output_json = os.path.join(chunk_temp_dir, "tracks.json")
                
                try:
                    # Copy chunk frames to temp dir
                    for i, frame_idx in enumerate(chunk_frame_indices):
                        src = os.path.join(temp_dir, f"{frame_idx:06d}.jpg")
                        dst = os.path.join(chunk_temp_dir, f"{i:06d}.jpg")
                        if os.path.exists(src):
                            shutil.copy(src, dst)
                    
                    # Run chunk worker in subprocess (GPU memory freed on exit!)
                    cmd = [
                        sys.executable, worker_script,
                        "--frames_dir", chunk_temp_dir,
                        "--output_json", chunk_output_json,
                        "--confidence", str(self.confidence_threshold),
                    ]
                    if points:
                        cmd += ["--points", json.dumps(points)]
                    result = subprocess.run(cmd, capture_output=True, text=True)
                    
                    if result.returncode != 0:
                        print(f"    Chunk subprocess failed: {result.stderr}")
                        continue
                    
                    # Print worker output for debugging
                    if result.stdout.strip():
                        for line in result.stdout.strip().split('\n'):
                            print(f"      [worker] {line}")
                    
                    # Load results from JSON
                    if not os.path.exists(chunk_output_json):
                        print(f"    No output JSON found")
                        continue
                    
                    with open(chunk_output_json) as f:
                        chunk_data = json.load(f)
                    
                    print(f"    Chunk JSON has {chunk_data.get('num_frames', 0)} frames")
                    
                    # Convert JSON to PersonTrack objects
                    chunk_tracks = {}
                    for frame_idx_str, persons_data in chunk_data.get("tracks", {}).items():
                        frame_idx = int(frame_idx_str)
                        persons = []
                        for p in persons_data:
                            persons.append(PersonTrack(
                                person_id=p["person_id"],
                                bbox=tuple(p["bbox"]),
                                bbox_px=(0, 0, 0, 0),
                                score=p["score"],
                                mask=None,
                            ))
                        chunk_tracks[frame_idx] = persons
                    
                    # Map local IDs to global IDs
                    if chunk_idx == 0:
                        for local_frame_idx, persons in chunk_tracks.items():
                            for person in persons:
                                key = (chunk_idx, person.person_id)
                                if key not in global_id_map:
                                    global_id_map[key] = next_global_id
                                    next_global_id += 1
                    else:
                        # Match with previous chunk using Hungarian algorithm
                        first_frame_persons = chunk_tracks.get(0, [])
                        
                        if prev_chunk_last_frame_tracks and first_frame_persons:
                            n_new = len(first_frame_persons)
                            n_prev = len(prev_chunk_last_frame_tracks)
                            
                            cost_matrix = np.ones((n_new, n_prev))
                            for i, new_person in enumerate(first_frame_persons):
                                for j, prev_person in enumerate(prev_chunk_last_frame_tracks):
                                    iou = self._compute_iou(new_person.bbox, prev_person.bbox)
                                    cost_matrix[i, j] = 1.0 - iou
                            
                            row_indices, col_indices = linear_sum_assignment(cost_matrix)
                            
                            for row_idx, col_idx in zip(row_indices, col_indices):
                                iou = 1.0 - cost_matrix[row_idx, col_idx]
                                if iou >= 0.3:
                                    new_person = first_frame_persons[row_idx]
                                    prev_person = prev_chunk_last_frame_tracks[col_idx]
                                    prev_key = (chunk_idx - 1, prev_person.person_id)
                                    if prev_key in global_id_map:
                                        global_id_map[(chunk_idx, new_person.person_id)] = global_id_map[prev_key]
                            
                            for i, new_person in enumerate(first_frame_persons):
                                key = (chunk_idx, new_person.person_id)
                                if key not in global_id_map:
                                    global_id_map[key] = next_global_id
                                    next_global_id += 1
                        
                        for local_frame_idx, persons in chunk_tracks.items():
                            for person in persons:
                                key = (chunk_idx, person.person_id)
                                if key not in global_id_map:
                                    global_id_map[key] = next_global_id
                                    next_global_id += 1
                    
                    # Save last frame for next chunk matching
                    if chunk_tracks:
                        last_frame_idx = max(chunk_tracks.keys())
                        prev_chunk_last_frame_tracks = chunk_tracks[last_frame_idx]
                    
                    # Merge into all_tracks (skip overlap frames for chunks > 0)
                    frames_merged = 0
                    for local_frame_idx, persons in chunk_tracks.items():
                        # Skip overlap frames - they were just for context
                        if local_frame_idx < overlap_start:
                            continue
                        
                        original_idx = original_frame_indices[start_idx + local_frame_idx]
                        
                        # Avoid duplicate frames (overlap region)
                        if original_idx in all_tracks:
                            continue
                        
                        remapped_persons = []
                        for person in persons:
                            key = (chunk_idx, person.person_id)
                            global_id = global_id_map.get(key, person.person_id)
                            
                            x1, y1, x2, y2 = person.bbox
                            remapped_person = PersonTrack(
                                person_id=global_id,
                                bbox=person.bbox,
                                bbox_px=(
                                    int(x1 * width),
                                    int(y1 * height),
                                    int(x2 * width),
                                    int(y2 * height),
                                ),
                                score=person.score,
                                mask=None,
                            )
                            remapped_persons.append(remapped_person)
                        
                        all_tracks[original_idx] = remapped_persons
                        frames_merged += 1
                    
                    print(f"    Chunk {chunk_idx+1} done - {frames_merged} frames merged (skipped {overlap_start} overlap)")
                    
                finally:
                    if os.path.exists(chunk_temp_dir):
                        shutil.rmtree(chunk_temp_dir)
                    
                    # Clean up GPU memory after each chunk
                    try:
                        self.end_session()
                    except:
                        pass
                    if self._predictor is not None:
                        try:
                            if hasattr(self._predictor, 'model'):
                                self._predictor.model.cpu()
                            del self._predictor
                            self._predictor = None
                        except:
                            pass
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            
            print(f"  Tracked {next_global_id} unique persons across all chunks")
            return all_tracks, (width, height), fps, original_frame_indices
            
        finally:
            if temp_dir and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
    
    def _compute_iou(self, bbox1, bbox2):
        """Compute IoU between two normalized bboxes."""
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0


# ============================================================================
# RetinaFace Detector
# ============================================================================

class RetinaFaceDetector:
    """Face detector using InsightFace's RetinaFace model."""
    
    def __init__(self, device: str = "cuda", det_thresh: float = 0.5):
        self.device = device
        self.det_thresh = det_thresh
        self._app = None
    
    def _ensure_loaded(self):
        """Lazy-load InsightFace model."""
        if self._app is None:
            try:
                from insightface.app import FaceAnalysis
                
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if self.device == "cuda" else ['CPUExecutionProvider']
                
                self._app = FaceAnalysis(
                    name='buffalo_l',
                    providers=providers
                )
                self._app.prepare(ctx_id=0 if self.device == "cuda" else -1, det_thresh=self.det_thresh)
                print("RetinaFace detector loaded successfully")
                
            except ImportError as e:
                raise ImportError(
                    "InsightFace not installed. Install: pip install insightface onnxruntime-gpu"
                ) from e
    
    def detect_faces(self, frame_bgr: np.ndarray) -> List[FaceDetection]:
        """Detect all faces in a frame."""
        self._ensure_loaded()
        
        height, width = frame_bgr.shape[:2]
        faces = self._app.get(frame_bgr)
        
        detections = []
        for face in faces:
            x1, y1, x2, y2 = face.bbox
            
            bbox_norm = (
                float(x1) / width,
                float(y1) / height,
                float(x2) / width,
                float(y2) / height,
            )
            bbox_px = (int(x1), int(y1), int(x2), int(y2))
            
            detections.append(FaceDetection(
                bbox=bbox_norm,
                bbox_px=bbox_px,
                score=float(face.det_score),
                landmarks=face.kps if hasattr(face, 'kps') else None,
                embedding=face.embedding if hasattr(face, 'embedding') else None,
            ))
        
        return detections


# ============================================================================
# Face-to-Person Matcher
# ============================================================================

def compute_iou(box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> float:
    """Compute IoU between two bounding boxes (pixel coordinates)."""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    xi1 = max(x1_1, x1_2)
    yi1 = max(y1_1, y1_2)
    xi2 = min(x2_1, x2_2)
    yi2 = min(y2_1, y2_2)
    
    if xi2 <= xi1 or yi2 <= yi1:
        return 0.0
    
    inter_area = (xi2 - xi1) * (yi2 - yi1)
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = box1_area + box2_area - inter_area
    
    if union_area <= 0:
        return 0.0
    
    return inter_area / union_area


def is_face_in_person(face_bbox_px: Tuple[int, int, int, int], person_bbox_px: Tuple[int, int, int, int]) -> bool:
    """Check if face center is within person bbox."""
    fx1, fy1, fx2, fy2 = face_bbox_px
    px1, py1, px2, py2 = person_bbox_px
    
    face_cx = (fx1 + fx2) / 2
    face_cy = (fy1 + fy2) / 2
    
    return px1 <= face_cx <= px2 and py1 <= face_cy <= py2


def match_faces_to_persons(
    face_detections: List[FaceDetection],
    person_tracks: List[PersonTrack],
    min_iou: float = 0.0
) -> Dict[int, Optional[FaceDetection]]:
    """
    Match detected faces to tracked persons using Hungarian algorithm.
    
    Uses IoU between face bbox and upper portion of person body as cost.
    Hungarian algorithm finds globally optimal assignment.
    
    Returns:
        Dict mapping person_id to matched FaceDetection (or None if no face found)
    """
    matches: Dict[int, Optional[FaceDetection]] = {p.person_id: None for p in person_tracks}
    
    if len(face_detections) == 0 or len(person_tracks) == 0:
        return matches
    
    n_faces = len(face_detections)
    n_persons = len(person_tracks)
    
    # Build cost matrix (faces x persons)
    # Cost = 1 - IoU (lower cost = better match)
    # Use upper 50% of person bbox for matching (where face should be)
    cost_matrix = np.ones((n_faces, n_persons)) * 1e6  # High cost = no match
    
    for i, face in enumerate(face_detections):
        for j, person in enumerate(person_tracks):
            # Get upper 50% of person bbox (where face should be)
            px1, py1, px2, py2 = person.bbox_px
            person_height = py2 - py1
            upper_body_bbox = (px1, py1, px2, int(py1 + person_height * 0.5))
            
            # Check if face center is within person bbox first
            if is_face_in_person(face.bbox_px, person.bbox_px):
                # Compute IoU with upper body
                iou = compute_iou(face.bbox_px, upper_body_bbox)
                if iou >= min_iou:
                    # Cost = 1 - IoU (so higher IoU = lower cost = better)
                    cost_matrix[i, j] = 1.0 - iou
    
    # Run Hungarian algorithm
    face_indices, person_indices = linear_sum_assignment(cost_matrix)
    
    # Assign matches (only if cost is reasonable)
    for face_idx, person_idx in zip(face_indices, person_indices):
        if cost_matrix[face_idx, person_idx] < 1.0:  # Valid match (not max cost)
            person = person_tracks[person_idx]
            matches[person.person_id] = face_detections[face_idx]
    
    return matches


# ============================================================================
# GazeAnywhere Predictor
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
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            from gaze_anywhere.inference import GazePredictor as _GazePredictor
            self._predictor = _GazePredictor(
                checkpoint_path=self.checkpoint_path,
                device=self.device,
            )
            print("GazeAnywhere model loaded successfully")
    
    def predict(self, frame_rgb: np.ndarray, position_prompt: str) -> dict:
        """Predict gaze for a person given their position."""
        self._ensure_loaded()
        return self._predictor.predict(frame_rgb, position_prompt)
    
    @staticmethod
    def format_position_prompt(bbox_norm: Tuple[float, float, float, float]) -> str:
        """Format a bounding box as a position prompt."""
        x1, y1, x2, y2 = bbox_norm
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        return f"position: [{cx:.4f}, {cy:.4f}]"


# ============================================================================
# Full Pipeline Annotator
# ============================================================================

class Sam3RetinaFaceGazeAnnotator:
    """
    Hybrid pipeline: SAM3 tracking → RetinaFace per-frame → GazeAnywhere
    
    Phase 1: Run SAM3 to track all persons with consistent IDs
    Phase 2: For each frame, detect faces with RetinaFace and match to persons
    Phase 3: For each matched face, estimate gaze with GazeAnywhere
    """
    
    def __init__(
        self,
        device: str = "cuda",
        gaze_checkpoint: str = "/projects/illinois/eng/cs/jrehg/users/xucao2/ChildGaze/checkpoints/GazeAnywhere/gazeanywhere.pth",
        sam3_confidence: float = 0.5,
        face_det_thresh: float = 0.5,
    ):
        self.device = device
        self.person_tracker = Sam3PersonTracker(device=device, confidence_threshold=sam3_confidence)
        self.face_detector = RetinaFaceDetector(device=device, det_thresh=face_det_thresh)
        self.gaze_predictor = GazePredictor(checkpoint_path=gaze_checkpoint, device=device)
    
    def annotate_video(
        self,
        video_path: str,
        max_frames: Optional[int] = None,
        progress_bar: bool = True,
        sample_fps: Optional[float] = None,  # Process at this fps (e.g., 2.0 = 2fps)
        points: list = None,  # GT point prompts [[x,y],...] normalized 0-1
    ) -> Dict:
        """
        Annotate video with person tracking and gaze estimation.
        
        Args:
            video_path: Path to input video
            max_frames: Maximum frames to process
            progress_bar: Show progress bar
            sample_fps: Sample video at this fps for per-frame processing.
                        SAM3 tracking still uses all frames, but RetinaFace+Gaze
                        only runs on sampled frames. Default None = all frames.
        
        Returns:
            Dict with video metadata and per-frame annotations
        """
        # Phase 1: Track persons with SAM3
        # For long videos, SAM3 will use presampling to reduce GPU memory
        result = self.person_tracker.process_video(
            video_path,
            max_frames=max_frames,
            sample_fps=sample_fps,
            use_presampling=True,  # Enable memory optimization for long videos
            points=points,
        )
        person_tracks, video_size, video_fps, presampled_indices = result
        
        width, height = video_size
        
        # Get video info
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_duration = total_frames / video_fps if video_fps > 0 else 0
        
        # Note: process_video's chunk processing already remaps frame indices to original video coordinates
        # So person_tracks already has original frame indices, no need to remap again!
        if presampled_indices is not None:
            print(f"  Got {len(person_tracks)} frames from SAM3 (already remapped to original coordinates)")
        
        # Calculate frames to process for per-frame gaze estimation
        all_tracked_frames = sorted(person_tracks.keys())
        
        if sample_fps and sample_fps < video_fps and presampled_indices is None:
            # Sample at target fps (only if not already presampled)
            sample_interval_sec = 1.0 / sample_fps
            sampled_frames = set()
            t = 0.0
            while t < video_duration:
                frame_idx = int(t * video_fps)
                if frame_idx in person_tracks:
                    sampled_frames.add(frame_idx)
                t += sample_interval_sec
            frame_indices = sorted(sampled_frames)
            effective_fps = sample_fps
        else:
            frame_indices = all_tracked_frames
            effective_fps = sample_fps if sample_fps else video_fps
        
        if max_frames:
            frame_indices = frame_indices[:max_frames]
        
        results = {
            "video_path": video_path,
            "video_fps": video_fps,
            "sample_fps": effective_fps,
            "video_size": list(video_size),
            "total_frames": total_frames,
            "processed_frames": len(frame_indices),
            "tracking_method": "sam3_text_prompt",
            "persons_summary": {},
            "frames": [],
        }
        
        person_frames = {}  # Track frame appearances
        person_embeddings = {}  # Store face embeddings for each person ID (for re-ID)
        
        # Phase 2 & 3: For each sampled frame, detect faces and estimate gaze
        if sample_fps:
            print(f"Processing faces and gaze at {sample_fps:.1f}fps ({len(frame_indices)} frames)...")
        else:
            print(f"Processing faces and gaze per frame ({len(frame_indices)} frames)...")
        pbar = tqdm(enumerate(frame_indices), desc="Estimating gaze", disable=not progress_bar, total=len(frame_indices))
        
        for processed_idx, frame_idx in pbar:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame_bgr = cap.read()
            if not ret:
                continue
            
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            
            # Detect all faces in frame
            face_detections = self.face_detector.detect_faces(frame_bgr)
            
            # Match faces to tracked persons
            persons_this_frame = person_tracks[frame_idx]
            face_matches = match_faces_to_persons(face_detections, persons_this_frame)
            
            frame_data = {
                "frame_idx": frame_idx,
                # Use processed frame index / sample_fps for consistent timing
                # This ensures 0.5s intervals at 2fps regardless of original video fps
                "timestamp": processed_idx / effective_fps if effective_fps > 0 else 0,
                "persons": [],
            }
            
            for person in persons_this_frame:
                # Track appearances
                if person.person_id not in person_frames:
                    person_frames[person.person_id] = []
                person_frames[person.person_id].append(frame_idx)
                
                matched_face = face_matches.get(person.person_id)
                
                person_data = {
                    "person_id": person.person_id,
                    "body_bbox": list(person.bbox),
                    "body_bbox_px": list(person.bbox_px),
                    "detection_score": person.score,
                    "face_detected": matched_face is not None,
                }
                
                if matched_face is not None:
                    # Store face embedding for this person (for re-ID across scenes)
                    if matched_face.embedding is not None:
                        if person.person_id not in person_embeddings:
                            person_embeddings[person.person_id] = []
                        person_embeddings[person.person_id].append(matched_face.embedding)
                    
                    # Use face position for gaze estimation
                    position_prompt = GazePredictor.format_position_prompt(matched_face.bbox)
                    gaze_result = self.gaze_predictor.predict(frame_rgb, position_prompt)
                    
                    person_data.update({
                        "face_bbox": list(matched_face.bbox),
                        "face_bbox_px": list(matched_face.bbox_px),
                        "head_bbox": list(gaze_result["head_bbox"]),
                        "head_bbox_px": list(gaze_result["head_bbox_px"]),
                        "gaze_point": list(gaze_result["gaze_point"]),
                        "gaze_point_px": list(gaze_result["gaze_point_px"]),
                        "inout": gaze_result["inout"],
                    })
                else:
                    # No face found - still include person but no gaze
                    person_data.update({
                        "face_bbox": None,
                        "face_bbox_px": None,
                        "head_bbox": None,
                        "head_bbox_px": None,
                        "gaze_point": None,
                        "gaze_point_px": None,
                        "inout": None,
                    })
                
                frame_data["persons"].append(person_data)
            
            results["frames"].append(frame_data)
        
        cap.release()
        
        # Add person summary
        for person_id, frames in person_frames.items():
            # Calculate face detection rate and in-frame gaze percentage
            face_detected_count = sum(
                1 for fd in results["frames"]
                for p in fd["persons"]
                if p["person_id"] == person_id and p["face_detected"]
            )
            inframe_count = sum(
                1 for fd in results["frames"]
                for p in fd["persons"]
                if p["person_id"] == person_id and p.get("inout") is True
            )
            
            results["persons_summary"][str(person_id)] = {
                "first_frame": min(frames),
                "last_frame": max(frames),
                "track_length": len(frames),
                "face_detection_pct": round(100 * face_detected_count / len(frames), 1) if frames else 0,
                "inframe_gaze_pct": round(100 * inframe_count / len(frames), 1) if frames else 0,
            }
        
        # Filter out false positive persons (no face detected at all)
        # Keep only persons where face was detected at least once
        valid_persons = set()
        for pid, summary in results["persons_summary"].items():
            if summary["face_detection_pct"] > 0:
                valid_persons.add(int(pid))
        
        # Remove invalid persons from frames
        for frame_data in results["frames"]:
            frame_data["persons"] = [
                p for p in frame_data["persons"] 
                if p["person_id"] in valid_persons
            ]
        
        # Remove invalid persons from summary
        results["persons_summary"] = {
            pid: summary for pid, summary in results["persons_summary"].items()
            if int(pid) in valid_persons
        }
        
        # =====================================================================
        # Face Re-ID: Merge person IDs with similar face embeddings
        # This links the same person re-detected with new ID within or across chunks
        # =====================================================================
        if person_embeddings:
            # Compute average embedding for each valid person
            avg_embeddings = {}
            for pid, embeddings in person_embeddings.items():
                if pid in valid_persons and len(embeddings) > 0:
                    avg_emb = np.mean(embeddings, axis=0)
                    avg_emb = avg_emb / (np.linalg.norm(avg_emb) + 1e-8)  # L2 normalize
                    avg_embeddings[pid] = avg_emb
            
            # Find similar person IDs using cosine similarity
            similarity_threshold = 0.6  # Higher threshold for conservative matching
            id_mapping = {}  # Maps old_id -> canonical_id
            
            sorted_pids = sorted(avg_embeddings.keys())
            for i, pid1 in enumerate(sorted_pids):
                if pid1 in id_mapping:
                    continue
                id_mapping[pid1] = pid1  # Self-map as canonical
                
                for pid2 in sorted_pids[i+1:]:
                    if pid2 in id_mapping:
                        continue
                    
                    # Compute cosine similarity
                    sim = np.dot(avg_embeddings[pid1], avg_embeddings[pid2])
                    if sim >= similarity_threshold:
                        id_mapping[pid2] = pid1  # Merge pid2 into pid1
            
            # Apply ID merging if duplicates found
            merged_count = sum(1 for old, new in id_mapping.items() if old != new)
            if merged_count > 0:
                print(f"  Face Re-ID: Merged {merged_count} duplicate person IDs (threshold={similarity_threshold})")
                
                # Remap person IDs in frames
                for frame_data in results["frames"]:
                    for person in frame_data["persons"]:
                        if person["person_id"] in id_mapping:
                            person["person_id"] = id_mapping[person["person_id"]]
                
                # Rebuild persons_summary with merged IDs
                merged_frames = {}
                for frame_data in results["frames"]:
                    for person in frame_data["persons"]:
                        pid = person["person_id"]
                        if pid not in merged_frames:
                            merged_frames[pid] = []
                        merged_frames[pid].append(frame_data["frame_idx"])
                
                merged_summary = {}
                for pid, frame_list in merged_frames.items():
                    face_count = sum(
                        1 for fd in results["frames"]
                        for p in fd["persons"]
                        if p["person_id"] == pid and p.get("face_detected")
                    )
                    inframe_count = sum(
                        1 for fd in results["frames"]
                        for p in fd["persons"]
                        if p["person_id"] == pid and p.get("inout") is True
                    )
                    unique_frames = sorted(set(frame_list))
                    merged_summary[str(pid)] = {
                        "first_frame": min(unique_frames),
                        "last_frame": max(unique_frames),
                        "track_length": len(unique_frames),
                        "face_detection_pct": round(100 * face_count / len(unique_frames), 1) if unique_frames else 0,
                        "inframe_gaze_pct": round(100 * inframe_count / len(unique_frames), 1) if unique_frames else 0,
                    }
                results["persons_summary"] = merged_summary
        
        return results
    
    def save_results(self, results: Dict, output_path: str) -> None:
        """Save annotation results to JSON file."""
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
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
    
    def reset(self):
        """Reset state for batch processing and free GPU memory."""
        import gc
        
        # Aggressively free SAM3 predictor and model GPU memory
        if self.person_tracker._predictor is not None:
            try:
                self.person_tracker.end_session()
            except:
                pass
            
            # Delete the internal model to free GPU memory
            if hasattr(self.person_tracker._predictor, 'model'):
                try:
                    # Move model to CPU first, then delete
                    self.person_tracker._predictor.model.cpu()
                    del self.person_tracker._predictor.model
                except:
                    pass
            
            del self.person_tracker._predictor
            self.person_tracker._predictor = None
            self.person_tracker._session_id = None
        
        # Force garbage collection to free remaining references
        gc.collect()
        
        # Clear CUDA cache to free GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
        print(f"[GPU Memory Reset] After cleanup: {torch.cuda.memory_allocated()/1e9:.1f}GB allocated")


# ============================================================================
# Visualization
# ============================================================================

def visualize_annotations(
    video_path: str,
    results: Dict,
    output_path: str,
    show_body_bbox: bool = True,
    show_face_bbox: bool = True,
    show_head_bbox: bool = True,
    show_gaze: bool = True,
    progress_bar: bool = True,
) -> None:
    """Create visualization video with annotations overlaid."""
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Use sample_fps for viz output so video plays at correct speed (not fast-forward)
    output_fps = results.get("sample_fps", results.get("video_fps", cap.get(cv2.CAP_PROP_FPS)))
    
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, output_fps, (width, height))
    
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
    out_color = (0, 0, 255)  # Red for OUT
    no_face_color = (128, 128, 128)  # Gray for no face detected
    
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
            face_detected = person.get("face_detected", False)
            inout = person.get("inout")
            
            if face_detected and person.get("face_bbox_px"):
                # Draw face bbox (from RetinaFace) with person ID label
                fx1, fy1, fx2, fy2 = map(int, person["face_bbox_px"])
                cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), color, 2, cv2.LINE_AA)
                
                # Person ID label
                label = f"P{person_id}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (fx1, fy1 - th - 8), (fx1 + tw + 4, fy1), color, -1)
                cv2.putText(frame, label, (fx1 + 2, fy1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
                
                # Draw gaze line and point
                if show_gaze:
                    # Use face center as gaze origin
                    fcx, fcy = int((fx1 + fx2) / 2), int((fy1 + fy2) / 2)
                    
                    if inout and person.get("gaze_point_px"):
                        gx, gy = int(person["gaze_point_px"][0]), int(person["gaze_point_px"][1])
                        # Draw directly without overlay to save memory
                        cv2.line(frame, (fcx, fcy), (gx, gy), color, 1, cv2.LINE_AA)
                        cv2.circle(frame, (gx, gy), 4, color, -1, cv2.LINE_AA)
                        cv2.circle(frame, (gx, gy), 4, (255, 255, 255), 1, cv2.LINE_AA)
                    elif inout is False:
                        cv2.putText(frame, "OUT", (fcx - 20, fcy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, out_color, 2, cv2.LINE_AA)
        
        # Frame info
        info_text = f"Frame: {frame_idx} | t={frame_data['timestamp']:.2f}s | {len(frame_data['persons'])} person(s)"
        (tw, th), _ = cv2.getTextSize(info_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(frame, (5, 5), (15 + tw, 35), (0, 0, 0), -1)
        cv2.putText(frame, info_text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        
        out.write(frame)
    
    cap.release()
    out.release()
    
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
    parser = argparse.ArgumentParser(
        description="SAM3 Person Tracking + RetinaFace + GazeAnywhere Pipeline"
    )
    
    # Input/output
    parser.add_argument("--video_path", type=str, required=True, help="Path to input video")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR,
                        help="Output directory for results")
    parser.add_argument("--output_json", type=str, default=None, 
                        help="Path to output JSON (auto-generated if not specified)")
    parser.add_argument("--output_video", type=str, default=None, 
                        help="Path to visualization video (auto-generated if not specified)")
    parser.add_argument("--no_visualization", action="store_true",
                        help="Skip visualization video generation")
    
    # Model options
    parser.add_argument(
        "--gaze_checkpoint", type=str,
        default="/projects/illinois/eng/cs/jrehg/users/xucao2/ChildGaze/checkpoints/GazeAnywhere/gazeanywhere.pth",
        help="Path to GazeAnywhere checkpoint"
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    
    # Processing options
    parser.add_argument("--max_frames", type=int, default=None, help="Max frames to process")
    parser.add_argument("--sam3_confidence", type=float, default=0.5, help="SAM3 detection confidence")
    parser.add_argument("--face_det_thresh", type=float, default=0.5, help="RetinaFace detection threshold")
    
    args = parser.parse_args()
    
    # Setup output paths
    os.makedirs(args.output_dir, exist_ok=True)
    video_name = Path(args.video_path).stem
    
    if args.output_json is None:
        args.output_json = os.path.join(args.output_dir, f"{video_name}_sam3rf_gaze.json")
    
    if args.output_video is None and not args.no_visualization:
        args.output_video = os.path.join(args.output_dir, f"{video_name}_sam3rf_viz.mp4")
    
    # Initialize annotator
    print("="*60)
    print("SAM3 Person Tracking + RetinaFace + GazeAnywhere Pipeline")
    print("="*60)
    print("Initializing models...")
    
    annotator = Sam3RetinaFaceGazeAnnotator(
        device=args.device,
        gaze_checkpoint=args.gaze_checkpoint,
        sam3_confidence=args.sam3_confidence,
        face_det_thresh=args.face_det_thresh,
    )
    
    # Process video
    print(f"\nProcessing: {args.video_path}")
    results = annotator.annotate_video(
        args.video_path,
        max_frames=args.max_frames,
    )
    
    # Save results
    annotator.save_results(results, args.output_json)
    
    # Print summary
    print(f"\n{'='*60}")
    print("Annotation Summary")
    print(f"{'='*60}")
    print(f"  Video: {results['video_path']}")
    print(f"  Size: {results['video_size'][0]}x{results['video_size'][1]}")
    print(f"  FPS: {results['video_fps']:.2f}")
    print(f"  Total frames: {results['total_frames']}")
    print(f"  Processed frames: {results['processed_frames']}")
    print(f"  Unique persons: {len(results['persons_summary'])}")
    
    for pid, summary in results["persons_summary"].items():
        print(f"    Person {pid}: frames {summary['first_frame']}-{summary['last_frame']} "
              f"({summary['track_length']} frames, "
              f"face={summary['face_detection_pct']:.1f}%, "
              f"inframe_gaze={summary['inframe_gaze_pct']:.1f}%)")
    
    print(f"\nOutput JSON: {args.output_json}")
    
    # Create visualization
    if args.output_video and not args.no_visualization:
        print(f"\nCreating visualization...")
        visualize_annotations(
            args.video_path,
            results,
            args.output_video
        )
        print(f"Output video: {args.output_video}")


if __name__ == "__main__":
    main()
