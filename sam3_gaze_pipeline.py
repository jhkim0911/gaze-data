"""
SAM3 + GazeAnywhere Multi-Person Video Annotation Pipeline

This module integrates SAM3 for person detection/tracking with GazeAnywhere
for per-person gaze estimation in videos.

Usage:
    from sam3_gaze_pipeline import MultiPersonGazeAnnotator
    
    annotator = MultiPersonGazeAnnotator(gaze_checkpoint="path/to/gazeanywhere.pth")
    results = annotator.annotate_video("video.mp4")
    annotator.save_results(results, "output.json")

CLI Usage:
    python sam3_gaze_pipeline.py --video_path video.mp4 --output_json results.json
"""

import os
import sys
import json
import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, NamedTuple
from dataclasses import dataclass, asdict

import torch
import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm

warnings.filterwarnings("ignore")

# Add gaze_anywhere to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "gaze_anywhere"))

# Import GazeAnywhere
from gaze_anywhere import GazePredictor


@dataclass
class PersonDetection:
    """Single person detection from SAM3."""
    person_id: int
    bbox: Tuple[float, float, float, float]  # (x1, y1, x2, y2) normalized [0, 1]
    bbox_px: Tuple[int, int, int, int]  # (x1, y1, x2, y2) pixel coordinates
    score: float
    mask: Optional[np.ndarray] = None  # Optional segmentation mask


@dataclass  
class PersonGazeResult:
    """Combined person detection + gaze estimation result."""
    person_id: int
    # From SAM3
    body_bbox: Tuple[float, float, float, float]  # normalized
    body_bbox_px: Tuple[int, int, int, int]  # pixels
    detection_score: float
    # From GazeAnywhere  
    head_bbox: Tuple[float, float, float, float]  # normalized
    head_bbox_px: Tuple[float, float, float, float]  # pixels
    gaze_point: Tuple[float, float]  # normalized
    gaze_point_px: Tuple[float, float]  # pixels
    inout: bool  # True if gaze is in-frame


class Sam3PersonDetector:
    """
    SAM3-based person detection and tracking for videos.
    
    Uses the official sam3 repo with Sam3VideoPredictor API.
    """
    
    def __init__(self, device: str = "cuda", confidence_threshold: float = 0.5):
        """
        Initialize SAM3 video predictor.
        
        Args:
            device: Device to run inference on
            confidence_threshold: Minimum confidence for detections
        """
        self.device = device
        self.confidence_threshold = confidence_threshold
        self._predictor = None
        self._session_id = None
        
    def _ensure_loaded(self):
        """Lazy-load the SAM3 model."""
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
        """
        Start a new video processing session.
        
        Args:
            video_path: Path to video file (MP4) or directory of JPEGs
            
        Returns:
            session_id: Session identifier
        """
        self._ensure_loaded()
        
        response = self._predictor.start_session(
            resource_path=video_path,
            session_id=None,
        )
        self._session_id = response["session_id"]
        return self._session_id
    
    def detect_persons(self, frame_index: int = 0) -> Dict:
        """
        Add "person" text prompt and get initial detections.
        
        Args:
            frame_index: Frame to use for initial detection
            
        Returns:
            dict with masks, boxes, scores
        """
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
        max_frames: Optional[int] = None,
        progress_bar: bool = True
    ) -> Dict[int, List[PersonDetection]]:
        """
        Propagate person tracking through the video.
        
        Args:
            max_frames: Maximum number of frames to process
            progress_bar: Show progress bar
            
        Returns:
            Dict mapping frame_idx to list of PersonDetection
        """
        if self._session_id is None:
            raise RuntimeError("No active session. Call start_session() first.")
        
        results = {}
        
        # Use handle_stream_request for propagate_in_video (it's a generator/stream request)
        # See sam3/model/sam3_video_predictor.py line 88-100
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
            
            # Skip if outputs is None (happens on non-rank-0 GPUs)
            if outputs is None:
                continue
            
            detections = []
            
            # Parse outputs - SAM3 video predictor returns:
            # - out_obj_ids: numpy array of object IDs
            # - out_probs: numpy array of detection scores  
            # - out_boxes_xywh: numpy array of boxes in (cx, cy, w, h) normalized format
            # - out_binary_masks: numpy array of binary masks
            obj_ids = outputs.get("out_obj_ids", [])
            scores = outputs.get("out_probs", [])
            boxes_xywh = outputs.get("out_boxes_xywh", [])
            masks = outputs.get("out_binary_masks", None)
            
            for i, obj_id in enumerate(obj_ids):
                score = scores[i] if i < len(scores) else 1.0
                if score >= self.confidence_threshold:
                    # Convert xywh (normalized) to xyxy (normalized)
                    # SAM3 returns (x, y, w, h) where x,y is TOP-LEFT corner
                    if i < len(boxes_xywh):
                        x, y, w, h = boxes_xywh[i]
                        x1, y1 = x, y
                        x2, y2 = x + w, y + h
                        box_norm = (x1, y1, x2, y2)
                    else:
                        box_norm = (0, 0, 0, 0)

                    
                    detections.append(PersonDetection(
                        person_id=int(obj_id),
                        bbox=box_norm,  # Already normalized
                        bbox_px=(0, 0, 0, 0),  # Will compute from normalized later
                        score=float(score),
                        mask=masks[i] if masks is not None and i < len(masks) else None
                    ))
            
            results[frame_idx] = detections

            
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
        progress_bar: bool = True
    ) -> Tuple[Dict[int, List[PersonDetection]], Tuple[int, int], float]:
        """
        Full video processing: detect and track all persons.
        
        Args:
            video_path: Path to video file
            max_frames: Maximum frames to process
            progress_bar: Show progress
            
        Returns:
            Tuple of:
                - Dict mapping frame_idx to list of PersonDetection
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
            
            # Initial detection with text prompt
            if progress_bar:
                print(f"Detecting persons in video ({total_frames} frames)...")
            self.detect_persons(frame_index=0)
            
            # Propagate through video
            if progress_bar:
                print("Propagating tracking through video...")
            detections = self.propagate_tracking(max_frames=max_frames, progress_bar=progress_bar)
            
            # Convert normalized bboxes to pixel coordinates
            for frame_idx, persons in detections.items():
                for person in persons:
                    # SAM3 video returns normalized coords, convert to pixels
                    x1, y1, x2, y2 = person.bbox
                    person.bbox_px = (
                        int(x1 * width),
                        int(y1 * height),
                        int(x2 * width),
                        int(y2 * height),
                    )
            
            return detections, (width, height), fps

            
        finally:
            self.end_session()


class Sam3PersonDetectorFallback:
    """
    Fallback person detector using frame-by-frame SAM3 image model.
    
    Use this if the video predictor is not available or for simpler use cases.
    """
    
    def __init__(self, device: str = "cuda", confidence_threshold: float = 0.5):
        self.device = device
        self.confidence_threshold = confidence_threshold
        self._model = None
        self._processor = None
        
    def _ensure_loaded(self):
        """Lazy-load SAM3 image model."""
        if self._model is None:
            try:
                from sam3.model_builder import build_sam3_image_model
                from sam3.model.sam3_image_processor import Sam3Processor
                
                self._model = build_sam3_image_model()
                self._processor = Sam3Processor(
                    self._model, 
                    confidence_threshold=self.confidence_threshold
                )
                print("SAM3 image model loaded successfully")
            except ImportError as e:
                raise ImportError(
                    "SAM3 not installed. Please install from: "
                    "git clone https://github.com/facebookresearch/sam3.git && "
                    "cd sam3 && pip install -e '.[notebooks]'"
                ) from e
    
    def detect_in_frame(self, frame: np.ndarray) -> List[PersonDetection]:
        """
        Detect persons in a single frame.
        
        Args:
            frame: RGB numpy array
            
        Returns:
            List of PersonDetection
        """
        self._ensure_loaded()
        
        # Convert to PIL
        image = Image.fromarray(frame)
        height, width = frame.shape[:2]
        
        # Set image and detect with text prompt
        inference_state = self._processor.set_image(image)
        output = self._processor.set_text_prompt(
            state=inference_state, 
            prompt="person"
        )
        
        # Parse results
        detections = []
        boxes = output.get("boxes", torch.zeros(0, 4))
        scores = output.get("scores", torch.zeros(0))
        masks = output.get("masks", None)
        
        if isinstance(boxes, torch.Tensor):
            boxes = boxes.cpu().numpy()
        if isinstance(scores, torch.Tensor):
            scores = scores.cpu().numpy()
        if masks is not None and isinstance(masks, torch.Tensor):
            masks = masks.cpu().numpy()
        
        for i, (box, score) in enumerate(zip(boxes, scores)):
            if score >= self.confidence_threshold:
                x1, y1, x2, y2 = box
                detections.append(PersonDetection(
                    person_id=i,  # No tracking, just index
                    bbox=(x1/width, y1/height, x2/width, y2/height),
                    bbox_px=(int(x1), int(y1), int(x2), int(y2)),
                    score=float(score),
                    mask=masks[i] if masks is not None else None
                ))
        
        return detections
    
    def process_video(
        self,
        video_path: str,
        sample_fps: Optional[float] = None,
        max_frames: Optional[int] = None,
        progress_bar: bool = True
    ) -> Tuple[Dict[int, List[PersonDetection]], Tuple[int, int], float]:
        """
        Process video frame-by-frame (no tracking).
        
        Args:
            video_path: Path to video
            sample_fps: Sample at this FPS (default: original)
            max_frames: Max frames to process
            progress_bar: Show progress
            
        Returns:
            Same as Sam3PersonDetector.process_video()
        """
        self._ensure_loaded()
        
        cap = cv2.VideoCapture(video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate frame step
        if sample_fps and sample_fps < fps:
            frame_step = max(1, int(fps / sample_fps))
        else:
            frame_step = 1
        
        frame_indices = list(range(0, total_frames, frame_step))
        if max_frames:
            frame_indices = frame_indices[:max_frames]
        
        results = {}
        pbar = tqdm(frame_indices, desc="Processing frames", disable=not progress_bar)
        
        for frame_idx in pbar:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue
            
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detect persons
            detections = self.detect_in_frame(frame_rgb)
            results[frame_idx] = detections
        
        cap.release()
        return results, (width, height), fps


class MultiPersonGazeAnnotator:
    """
    Combines SAM3 person detection with GazeAnywhere gaze estimation.
    
    Pipeline:
    1. Use SAM3 to detect all persons in each frame
    2. For each person, create position-based text prompt
    3. Run GazeAnywhere to estimate gaze direction
    4. Output combined annotations
    """
    
    def __init__(
        self,
        gaze_checkpoint: str,
        device: str = "cuda",
        use_video_predictor: bool = True,
        confidence_threshold: float = 0.5,
        use_dark_inference: bool = True
    ):
        """
        Initialize the annotator.
        
        Args:
            gaze_checkpoint: Path to GazeAnywhere checkpoint
            device: Device for inference
            use_video_predictor: Use SAM3 video predictor (with tracking) vs image model
            confidence_threshold: Minimum detection confidence
            use_dark_inference: Use DARK refinement for gaze
        """
        self.device = device
        
        # Initialize person detector
        if use_video_predictor:
            self.person_detector = Sam3PersonDetector(
                device=device,
                confidence_threshold=confidence_threshold
            )
        else:
            self.person_detector = Sam3PersonDetectorFallback(
                device=device,
                confidence_threshold=confidence_threshold
            )
        
        # Initialize gaze predictor
        self.gaze_predictor = GazePredictor(
            checkpoint_path=gaze_checkpoint,
            device=device,
            use_dark_inference=use_dark_inference
        )
    
    @staticmethod
    def format_position_prompt(bbox_norm: Tuple[float, float, float, float]) -> str:
        """
        Convert normalized bbox to position prompt for GazeAnywhere.
        
        Args:
            bbox_norm: (x1, y1, x2, y2) normalized coordinates
            
        Returns:
            Position prompt string like "position: [0.35, 0.42]"
        """
        cx = (bbox_norm[0] + bbox_norm[2]) / 2
        cy = (bbox_norm[1] + bbox_norm[3]) / 2
        return f"position: [{cx:.2f}, {cy:.2f}]"
    
    def annotate_video(
        self,
        video_path: str,
        sample_fps: Optional[float] = None,
        max_frames: Optional[int] = None,
        progress_bar: bool = True
    ) -> Dict:
        """
        Annotate a video with person detection and gaze estimation.
        
        Args:
            video_path: Path to video file
            sample_fps: Sample at this FPS (only for fallback detector)
            max_frames: Maximum frames to process
            progress_bar: Show progress bars
            
        Returns:
            Annotation dict with video metadata and per-frame results
        """
        video_path = str(video_path)
        
        # Step 1: Detect all persons
        if isinstance(self.person_detector, Sam3PersonDetectorFallback):
            person_detections, video_size, video_fps = self.person_detector.process_video(
                video_path,
                sample_fps=sample_fps,
                max_frames=max_frames,
                progress_bar=progress_bar
            )
        else:
            person_detections, video_size, video_fps = self.person_detector.process_video(
                video_path,
                max_frames=max_frames,
                progress_bar=progress_bar
            )
        
        width, height = video_size
        
        # Step 2: For each frame, run gaze estimation on each person
        cap = cv2.VideoCapture(video_path)
        
        results = {
            "video_path": video_path,
            "video_fps": video_fps,
            "video_size": list(video_size),
            "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "processed_frames": len(person_detections),
            "persons_summary": {},
            "frames": []
        }
        
        # Track person appearances
        person_frames = {}  # person_id -> list of frame indices
        
        frame_indices = sorted(person_detections.keys())
        pbar = tqdm(frame_indices, desc="Estimating gaze", disable=not progress_bar)
        
        for frame_idx in pbar:
            # Read frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            frame_result = {
                "frame_idx": frame_idx,
                "timestamp": frame_idx / video_fps,
                "persons": []
            }
            
            # Process each detected person
            for person in person_detections[frame_idx]:
                # Track person appearances
                if person.person_id not in person_frames:
                    person_frames[person.person_id] = []
                person_frames[person.person_id].append(frame_idx)
                
                # Create position prompt from body bbox
                text_prompt = self.format_position_prompt(person.bbox)
                
                # Run gaze estimation
                gaze_result = self.gaze_predictor.predict(frame_rgb, text_prompt)
                
                # Combine results
                person_result = {
                    "person_id": person.person_id,
                    "body_bbox": list(person.bbox),
                    "body_bbox_px": list(person.bbox_px),
                    "detection_score": person.score,
                    "head_bbox": list(gaze_result["head_bbox"]),
                    "head_bbox_px": list(gaze_result["head_bbox_px"]),
                    "gaze_point": list(gaze_result["gaze_point"]),
                    "gaze_point_px": list(gaze_result["gaze_point_px"]),
                    "inout": gaze_result["inout"],
                }
                frame_result["persons"].append(person_result)
            
            results["frames"].append(frame_result)
        
        cap.release()
        
        # Add person summary
        for person_id, frames in person_frames.items():
            results["persons_summary"][str(person_id)] = {
                "first_frame": min(frames),
                "last_frame": max(frames),
                "track_length": len(frames)
            }
        
        return results
    
    def save_results(self, results: Dict, output_path: str) -> None:
        """Save annotation results to JSON file."""
        # Custom encoder for numpy types
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.floating):
                    return float(obj)
                if isinstance(obj, np.integer):
                    return int(obj)
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super().default(obj)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, cls=NumpyEncoder)
        print(f"Saved results to: {output_path}")

    
    def load_results(self, json_path: str) -> Dict:
        """Load annotation results from JSON file."""
        with open(json_path, 'r') as f:
            return json.load(f)


def overlay_heatmap_on_image(
    image_bgr: np.ndarray,
    heatmap_01: np.ndarray,
    alpha: float = 0.3,
    mask_thresh: float = 0.10,
    colormap: int = cv2.COLORMAP_JET
) -> np.ndarray:
    """
    Overlay heatmap on image with masking (like GazeAnywhere visualization).
    
    Args:
        image_bgr: BGR image
        heatmap_01: Heatmap array with values in [0, 1]
        alpha: Blend strength
        mask_thresh: Only show heatmap where value >= thresh
        colormap: OpenCV colormap
        
    Returns:
        Blended BGR image
    """
    H, W = image_bgr.shape[:2]
    
    # Resize heatmap to image size
    heat_resized = cv2.resize(heatmap_01.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
    heat_resized = np.clip(heat_resized, 0.0, 1.0)
    
    # Convert to colormap
    heat_uint8 = (heat_resized * 255).round().astype(np.uint8)
    colored_heat_bgr = cv2.applyColorMap(heat_uint8, colormap)
    
    # Blend
    img_f = image_bgr.astype(np.float32)
    heat_f = colored_heat_bgr.astype(np.float32)
    blended = cv2.addWeighted(heat_f, alpha, img_f, 1 - alpha, 0.0)
    
    # Only apply where heatmap is meaningful
    mask = (heat_resized >= mask_thresh).astype(np.float32)[..., None]
    overlay_f = np.where(mask > 0, blended, img_f)
    
    return np.clip(overlay_f, 0, 255).astype(np.uint8)


def visualize_annotations(
    video_path: str,
    results: Dict,
    output_path: str,
    show_body_bbox: bool = True,
    show_head_bbox: bool = True,
    show_gaze: bool = True,
    show_heatmap: bool = False,
    heatmap_alpha: float = 0.3,
    progress_bar: bool = True
) -> None:
    """
    Create visualization video from annotations (GazeAnywhere style).
    
    Args:
        video_path: Original video path
        results: Annotation results from MultiPersonGazeAnnotator
        output_path: Output video path
        show_body_bbox: Draw body bounding boxes (from SAM3)
        show_head_bbox: Draw head bounding boxes (from GazeAnywhere)
        show_gaze: Draw gaze lines and points
        show_heatmap: Overlay gaze heatmap (requires heatmaps in results)
        heatmap_alpha: Heatmap blend alpha
        progress_bar: Show progress
    """
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = results.get("video_fps", cap.get(cv2.CAP_PROP_FPS))
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Color palette for different persons (BGR)
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
    
    # Out-of-frame color
    out_color = (0, 0, 255)  # Red
    
    pbar = tqdm(results["frames"], desc="Creating visualization", disable=not progress_bar)
    
    for frame_data in pbar:
        frame_idx = frame_data["frame_idx"]
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if not ret:
            continue
        
        # Apply heatmap overlay if available
        if show_heatmap and "heatmaps" in frame_data:
            for heatmap in frame_data["heatmaps"]:
                frame = overlay_heatmap_on_image(frame, heatmap, alpha=heatmap_alpha)
        
        for person in frame_data["persons"]:
            person_id = person["person_id"]
            color = colors[person_id % len(colors)]
            inout = person["inout"]
            
            # Draw body bbox (from SAM3)
            if show_body_bbox:
                x1, y1, x2, y2 = map(int, person["body_bbox_px"])
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, lineType=cv2.LINE_AA)
                # Person ID label with background
                label = f"P{person_id}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
                cv2.putText(
                    frame, label,
                    (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA
                )
            
            # Draw head bbox (from GazeAnywhere)
            if show_head_bbox:
                hx1, hy1, hx2, hy2 = map(int, person["head_bbox_px"])
                # Thicker line for head
                cv2.rectangle(frame, (hx1, hy1), (hx2, hy2), color if inout else out_color, 3, lineType=cv2.LINE_AA)
            
            # Draw gaze line and point
            if show_gaze:
                # Head center
                hx1, hy1, hx2, hy2 = person["head_bbox_px"]
                hcx, hcy = int((hx1 + hx2) / 2), int((hy1 + hy2) / 2)
                
                if inout:
                    # Gaze point
                    gx, gy = map(int, person["gaze_point_px"])
                    
                    # Draw line from head center to gaze point
                    cv2.line(frame, (hcx, hcy), (gx, gy), color, 3, lineType=cv2.LINE_AA)
                    
                    # Draw filled circle at gaze point
                    cv2.circle(frame, (gx, gy), 12, color, -1, lineType=cv2.LINE_AA)
                    cv2.circle(frame, (gx, gy), 12, (255, 255, 255), 2, lineType=cv2.LINE_AA)
                else:
                    # Out-of-frame indicator
                    cv2.putText(
                        frame, "OUT",
                        (hcx - 20, hcy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, out_color, 2, cv2.LINE_AA
                    )
        
        # Frame info overlay
        info_text = f"Frame: {frame_idx} | t={frame_data['timestamp']:.2f}s | {len(frame_data['persons'])} person(s)"
        # Background for text
        (tw, th), _ = cv2.getTextSize(info_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(frame, (5, 5), (15 + tw, 35), (0, 0, 0), -1)
        cv2.putText(
            frame, info_text,
            (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA
        )
        
        out.write(frame)
    
    cap.release()
    out.release()
    
    # Re-encode with H.264 for better compatibility (VS Code, web browsers)
    _reencode_video_h264(output_path)
    
    print(f"Saved visualization to: {output_path}")


def _reencode_video_h264(video_path: str) -> None:
    """
    Re-encode video with H.264 codec for better compatibility.
    
    OpenCV's mp4v codec isn't always compatible with web players and VS Code.
    This function re-encodes the video using ffmpeg's libx264.
    """
    import subprocess
    import shutil
    
    # Check if ffmpeg is available
    if shutil.which("ffmpeg") is None:
        print("Warning: ffmpeg not found, skipping H.264 re-encoding. Video may not play in VS Code.")
        return
    
    temp_path = video_path + ".temp.mp4"
    
    try:
        # Rename original to temp
        os.rename(video_path, temp_path)
        
        # Re-encode with H.264
        cmd = [
            "ffmpeg", "-y",  # Overwrite output
            "-i", temp_path,  # Input
            "-c:v", "libx264",  # H.264 codec
            "-preset", "fast",  # Encoding speed
            "-crf", "23",  # Quality (lower = better, 18-28 is good range)
            "-pix_fmt", "yuv420p",  # Pixel format for compatibility
            "-movflags", "+faststart",  # Web-friendly
            "-loglevel", "error",  # Quiet
            video_path
        ]
        
        subprocess.run(cmd, check=True)
        
        # Remove temp file
        os.remove(temp_path)
        
    except subprocess.CalledProcessError as e:
        print(f"Warning: H.264 re-encoding failed: {e}")
        # Restore original if re-encoding failed
        if os.path.exists(temp_path):
            if os.path.exists(video_path):
                os.remove(video_path)
            os.rename(temp_path, video_path)
    except Exception as e:
        print(f"Warning: H.264 re-encoding error: {e}")
        if os.path.exists(temp_path) and not os.path.exists(video_path):
            os.rename(temp_path, video_path)



# Default output directory
DEFAULT_OUTPUT_DIR = "/u/arkimjh/code/ECCV-jh/data_results"


def main():
    parser = argparse.ArgumentParser(
        description="SAM3 + GazeAnywhere Multi-Person Video Annotation"
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
    parser.add_argument("--use_image_model", action="store_true", 
                        help="Use SAM3 image model (no tracking) instead of video predictor")
    
    # Processing options
    parser.add_argument("--sample_fps", type=float, default=None, 
                        help="Sample video at this FPS (image model only)")
    parser.add_argument("--max_frames", type=int, default=None, help="Max frames to process")
    parser.add_argument("--confidence", type=float, default=0.5, help="Detection confidence threshold")
    parser.add_argument("--no_dark_inference", action="store_true", help="Disable DARK gaze refinement")
    
    args = parser.parse_args()
    
    # Setup output paths
    os.makedirs(args.output_dir, exist_ok=True)
    video_name = Path(args.video_path).stem
    
    if args.output_json is None:
        args.output_json = os.path.join(args.output_dir, f"{video_name}_gaze.json")
    
    if args.output_video is None and not args.no_visualization:
        args.output_video = os.path.join(args.output_dir, f"{video_name}_viz.mp4")
    
    # Initialize annotator
    print("Initializing models...")
    annotator = MultiPersonGazeAnnotator(
        gaze_checkpoint=args.gaze_checkpoint,
        device=args.device,
        use_video_predictor=not args.use_image_model,
        confidence_threshold=args.confidence,
        use_dark_inference=not args.no_dark_inference
    )
    
    # Process video
    print(f"Processing: {args.video_path}")
    results = annotator.annotate_video(
        args.video_path,
        sample_fps=args.sample_fps,
        max_frames=args.max_frames
    )
    
    # Save results
    annotator.save_results(results, args.output_json)
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"Annotation Summary")
    print(f"{'='*50}")
    print(f"  Video: {results['video_path']}")
    print(f"  Size: {results['video_size'][0]}x{results['video_size'][1]}")
    print(f"  FPS: {results['video_fps']:.2f}")
    print(f"  Total frames: {results['total_frames']}")
    print(f"  Processed frames: {results['processed_frames']}")
    print(f"  Unique persons: {len(results['persons_summary'])}")
    
    for pid, summary in results["persons_summary"].items():
        inframe_pct = sum(1 for f in results["frames"] 
                         for p in f["persons"] 
                         if p["person_id"] == int(pid) and p["inout"]) / max(1, summary["track_length"]) * 100
        print(f"    Person {pid}: frames {summary['first_frame']}-{summary['last_frame']} "
              f"({summary['track_length']} frames, {inframe_pct:.1f}% in-frame gaze)")
    
    print(f"\nOutput JSON: {args.output_json}")
    
    # Create visualization if requested
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

