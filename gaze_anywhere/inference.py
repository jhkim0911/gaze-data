"""
GazeAnywhere Inference Script for Data Annotation

Usage:
    python inference.py --image_path <path_to_image> --text "description of person"
    python inference.py --video_path <path_to_video> --text "description of person" --output_json results.json

Or use as a module:
    from inference import GazePredictor
    predictor = GazePredictor(checkpoint_path)
    result = predictor.predict(image_path, text_prompt)

    # For video processing:
    results = predictor.predict_video(video_path, text_prompt, batch_size=16)
"""
import sys
import os
from os import path as osp
import argparse
import warnings
from typing import Tuple, List, Dict, Optional, Union
from pathlib import Path
import json

import torch
import numpy as np
from PIL import Image
import cv2
from torchvision import transforms
from tqdm import tqdm

# Add current directory to path
sys.path.insert(0, osp.dirname(osp.abspath(__file__)))

from detectron2.config import instantiate, LazyConfig
from tools.utils import dark_inference, argmax_pts

warnings.simplefilter(action="ignore", category=FutureWarning)

# Default checkpoint path
DEFAULT_CHECKPOINT = "/projects/illinois/eng/cs/jrehg/users/xucao2/ChildGaze/checkpoints/GazeAnywhere/gazeanywhere.pth"


class GazePredictor:
    """
    GazeAnywhere model wrapper for easy inference.

    Outputs:
        - gaze_point: (x, y) normalized coordinates in [0, 1]
        - inout: bool, True if gaze is in-frame
        - head_bbox: (x1, y1, x2, y2) normalized coordinates in [0, 1]
        - heatmap: (H, W) numpy array of gaze probability
    """

    def __init__(
        self,
        checkpoint_path: str = DEFAULT_CHECKPOINT,
        config_path: Optional[str] = None,
        device: str = "cuda",
        use_dark_inference: bool = True
    ):
        """
        Initialize GazeAnywhere predictor.

        Args:
            checkpoint_path: Path to model checkpoint (.pth file)
            config_path: Path to config file (uses default if None)
            device: Device to run inference on ("cuda" or "cpu")
            use_dark_inference: Use DARK refinement for sub-pixel accuracy
        """
        # Validate checkpoint
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        # Check CUDA availability
        if device == "cuda" and not torch.cuda.is_available():
            warnings.warn("CUDA not available, falling back to CPU")
            device = "cpu"

        self.device = device
        self.use_dark_inference = use_dark_inference

        # Load config
        if config_path is None:
            config_path = osp.join(osp.dirname(__file__), "configs", "gazeanywhere_config.py")

        cfg = LazyConfig.load(config_path)

        # Instantiate model
        self.model = instantiate(cfg.model)

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        self.model.load_state_dict(checkpoint["model"])

        self.model = self.model.to(device)
        self.model.eval()

        # Image transform
        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

        self.out_size = self.model.out_size  # (64, 64)

    @staticmethod
    def format_position_prompt(
        bbox_norm: Tuple[float, float, float, float]
    ) -> str:
        """
        Convert normalized bbox to position prompt for multi-person scenarios.
        
        The GazeAnywhere model supports position-based prompts in the format:
        "position: [cx, cy]" where cx, cy are normalized center coordinates.
        
        Args:
            bbox_norm: (x1, y1, x2, y2) normalized bounding box coordinates [0, 1]
            
        Returns:
            Position prompt string like "position: [0.35, 0.42]"
            
        Example:
            >>> GazePredictor.format_position_prompt((0.1, 0.2, 0.4, 0.6))
            'position: [0.25, 0.40]'
        """
        cx = (bbox_norm[0] + bbox_norm[2]) / 2
        cy = (bbox_norm[1] + bbox_norm[3]) / 2
        return f"position: [{cx:.2f}, {cy:.2f}]"

    def predict(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        text_prompt: str = ""
    ) -> Dict:
        """
        Predict gaze for a single image.

        Args:
            image: Image path, PIL Image, or numpy array (RGB)
            text_prompt: Text description of the person (e.g., "person with blonde hair")

        Returns:
            dict with keys:
                - gaze_point: (x, y) tuple, normalized [0, 1]
                - gaze_point_px: (x, y) tuple, pixel coordinates
                - inout: bool, True if gaze is in-frame
                - head_bbox: (x1, y1, x2, y2) normalized [0, 1]
                - head_bbox_px: (x1, y1, x2, y2) pixel coordinates
                - heatmap: (64, 64) numpy array
                - image_size: (width, height) original size
        """
        # Load image
        if isinstance(image, (str, Path)):
            img_pil = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            # Handle both RGB and BGR arrays
            if image.ndim == 3 and image.shape[2] == 3:
                img_pil = Image.fromarray(image)
            else:
                raise ValueError(f"Unexpected image shape: {image.shape}")
        else:
            img_pil = image.convert("RGB")

        img_size = img_pil.size  # (width, height)

        # Transform
        img_tensor = self.transform(img_pil).unsqueeze(0).to(self.device)

        # Inference
        with torch.no_grad():
            gaze_heatmap, inout_pred, bbox_pred = self.model.inference(img_tensor, [text_prompt])

        # Process outputs (convert to float32 first for numpy compatibility)
        heatmap = gaze_heatmap.squeeze().cpu().float().numpy()
        inout_score = inout_pred.cpu().float().item()   # 연속 score (AP 등 threshold-free 지표용)
        inout = inout_score > 0.5
        bbox = bbox_pred.cpu().float().numpy()[0]  # (cx, cy, w, h) normalized

        # Extract gaze point
        if self.use_dark_inference:
            pred_x, pred_y = dark_inference(heatmap)
        else:
            pred_x, pred_y = argmax_pts(heatmap)

        # Normalize gaze point to [0, 1]
        gaze_x_norm = pred_x / self.out_size[0]
        gaze_y_norm = pred_y / self.out_size[1]

        # Convert to pixel coordinates
        gaze_x_px = gaze_x_norm * img_size[0]
        gaze_y_px = gaze_y_norm * img_size[1]

        # Convert bbox from (cx, cy, w, h) to (x1, y1, x2, y2)
        bbox_xyxy_norm = (
            bbox[0] - bbox[2] / 2,  # x1
            bbox[1] - bbox[3] / 2,  # y1
            bbox[0] + bbox[2] / 2,  # x2
            bbox[1] + bbox[3] / 2,  # y2
        )
        bbox_xyxy_px = (
            bbox_xyxy_norm[0] * img_size[0],
            bbox_xyxy_norm[1] * img_size[1],
            bbox_xyxy_norm[2] * img_size[0],
            bbox_xyxy_norm[3] * img_size[1],
        )

        return {
            "gaze_point": (gaze_x_norm, gaze_y_norm),
            "gaze_point_px": (gaze_x_px, gaze_y_px),
            "inout": inout,
            "inout_score": inout_score,
            "head_bbox": bbox_xyxy_norm,
            "head_bbox_px": bbox_xyxy_px,
            "heatmap": heatmap,
            "image_size": img_size,
        }

    def predict_batch(
        self,
        images: List[Union[str, Path, Image.Image, np.ndarray]],
        text_prompts: Union[str, List[str]]
    ) -> List[Dict]:
        """
        Predict gaze for a batch of images.

        Args:
            images: List of image paths, PIL Images, or numpy arrays
            text_prompts: Single text prompt (applied to all) or list of prompts (one per image)

        Returns:
            List of prediction dicts (same format as predict())
        """
        # Handle single prompt for all images
        if isinstance(text_prompts, str):
            text_prompts = [text_prompts] * len(images)
        
        assert len(images) == len(text_prompts), "Number of images and prompts must match"

        # Process images one at a time (more reliable for different sizes)
        # The model's inference method handles text tokenization internally
        results = []
        for img, prompt in zip(images, text_prompts):
            results.append(self.predict(img, prompt))

        return results

    def predict_video(
        self,
        video_path: Union[str, Path],
        text_prompt: str = "",
        batch_size: int = 16,
        sample_fps: Optional[float] = None,
        max_frames: Optional[int] = None,
        return_heatmaps: bool = False,
        progress_bar: bool = True
    ) -> Dict:
        """
        Predict gaze for all frames in a video.

        Args:
            video_path: Path to video file
            text_prompt: Text description of the person to track
            batch_size: Number of frames to process at once
            sample_fps: If set, sample video at this FPS (default: use original FPS)
            max_frames: Maximum number of frames to process (default: all)
            return_heatmaps: If True, include heatmaps in results (memory intensive)
            progress_bar: Show progress bar during processing

        Returns:
            dict with keys:
                - video_path: str, input video path
                - video_fps: float, original video FPS
                - video_size: (width, height) tuple
                - total_frames: int, total frames in video
                - processed_frames: int, number of frames processed
                - sample_fps: float, FPS used for sampling
                - text_prompt: str, text prompt used
                - frames: List[Dict] with per-frame results:
                    - frame_idx: int, original frame index
                    - timestamp: float, timestamp in seconds
                    - gaze_point: (x, y) normalized [0, 1]
                    - gaze_point_px: (x, y) pixel coordinates
                    - inout: bool
                    - head_bbox: (x1, y1, x2, y2) normalized
                    - head_bbox_px: (x1, y1, x2, y2) pixels
                    - heatmap: (optional) numpy array
        """
        video_path = str(video_path)
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")

        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")

        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Calculate frame sampling
        if sample_fps is None or sample_fps >= video_fps:
            sample_fps = video_fps
            frame_step = 1
        else:
            frame_step = max(1, int(video_fps / sample_fps))

        # Calculate frames to process
        frame_indices = list(range(0, total_frames, frame_step))
        if max_frames is not None:
            frame_indices = frame_indices[:max_frames]

        # Process frames in batches
        results = {
            "video_path": video_path,
            "video_fps": video_fps,
            "video_size": (width, height),
            "total_frames": total_frames,
            "processed_frames": len(frame_indices),
            "sample_fps": video_fps / frame_step,
            "text_prompt": text_prompt,
            "frames": []
        }

        # Create batches
        batches = [
            frame_indices[i:i + batch_size]
            for i in range(0, len(frame_indices), batch_size)
        ]

        pbar = tqdm(batches, desc="Processing video", disable=not progress_bar)
        
        for batch_indices in pbar:
            frames = []
            valid_indices = []

            # Read frames for this batch
            for frame_idx in batch_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if ret:
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame_rgb)
                    valid_indices.append(frame_idx)

            if not frames:
                continue

            # Process batch
            batch_results = self.predict_batch(frames, text_prompt)

            # Store results
            for frame_idx, result in zip(valid_indices, batch_results):
                frame_result = {
                    "frame_idx": frame_idx,
                    "timestamp": frame_idx / video_fps,
                    "gaze_point": result["gaze_point"],
                    "gaze_point_px": result["gaze_point_px"],
                    "inout": result["inout"],
                    "head_bbox": result["head_bbox"],
                    "head_bbox_px": result["head_bbox_px"],
                }
                if return_heatmaps:
                    frame_result["heatmap"] = result["heatmap"]
                
                results["frames"].append(frame_result)

        cap.release()
        return results

    def predict_frames(
        self,
        frames: List[np.ndarray],
        text_prompt: str = "",
        batch_size: int = 16,
        progress_bar: bool = True
    ) -> List[Dict]:
        """
        Predict gaze for a list of frame arrays (useful for pre-extracted frames).

        Args:
            frames: List of numpy arrays (RGB format)
            text_prompt: Text description of the person
            batch_size: Number of frames to process at once
            progress_bar: Show progress bar

        Returns:
            List of prediction dicts (same format as predict())
        """
        results = []
        batches = [
            frames[i:i + batch_size]
            for i in range(0, len(frames), batch_size)
        ]

        pbar = tqdm(batches, desc="Processing frames", disable=not progress_bar)
        
        for batch in pbar:
            batch_results = self.predict_batch(batch, text_prompt)
            results.extend(batch_results)

        return results


def visualize_gaze(
    image: Union[str, np.ndarray],
    result: Dict,
    output_path: Optional[str] = None,
    alpha: float = 0.3
) -> np.ndarray:
    """
    Visualize gaze prediction on image.

    Args:
        image: Image path or BGR numpy array
        result: Output from GazePredictor.predict()
        output_path: Optional path to save visualization
        alpha: Heatmap blend alpha

    Returns:
        Visualization as BGR numpy array
    """
    if isinstance(image, str):
        img_bgr = cv2.imread(image)
    else:
        img_bgr = image.copy()

    H, W = img_bgr.shape[:2]

    # Resize heatmap
    heatmap = result["heatmap"]
    heatmap_resized = cv2.resize(heatmap.astype(np.float32), (W, H))
    heatmap_uint8 = (np.clip(heatmap_resized, 0, 1) * 255).astype(np.uint8)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    # Blend
    overlay = cv2.addWeighted(heatmap_colored, alpha, img_bgr, 1 - alpha, 0)

    # Draw head bbox
    bbox = result["head_bbox_px"]
    x1, y1, x2, y2 = map(int, bbox)
    color = (0, 255, 0) if result["inout"] else (0, 0, 255)
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)

    # Draw gaze point and line
    if result["inout"]:
        gx, gy = map(int, result["gaze_point_px"])
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.line(overlay, (cx, cy), (gx, gy), color, 2)
        cv2.circle(overlay, (gx, gy), 8, color, -1)

    if output_path:
        cv2.imwrite(output_path, overlay)

    return overlay


def visualize_video(
    video_path: str,
    results: Dict,
    output_path: str,
    show_heatmap: bool = True,
    alpha: float = 0.3,
    progress_bar: bool = True
) -> None:
    """
    Create visualization video from prediction results.

    Args:
        video_path: Path to original video
        results: Output from GazePredictor.predict_video()
        output_path: Path to save output video
        show_heatmap: Whether to overlay heatmap (requires return_heatmaps=True)
        alpha: Heatmap blend alpha
        progress_bar: Show progress bar
    """
    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, results["sample_fps"], (width, height))

    # Create lookup for frame results
    frame_lookup = {r["frame_idx"]: r for r in results["frames"]}

    pbar = tqdm(results["frames"], desc="Creating video", disable=not progress_bar)
    
    for frame_result in pbar:
        frame_idx = frame_result["frame_idx"]
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if not ret:
            continue

        H, W = frame.shape[:2]
        color = (0, 255, 0) if frame_result["inout"] else (0, 0, 255)

        # Draw heatmap if available
        if show_heatmap and "heatmap" in frame_result:
            heatmap = frame_result["heatmap"]
            heatmap_resized = cv2.resize(heatmap.astype(np.float32), (W, H))
            heatmap_uint8 = (np.clip(heatmap_resized, 0, 1) * 255).astype(np.uint8)
            heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
            frame = cv2.addWeighted(heatmap_colored, alpha, frame, 1 - alpha, 0)

        # Draw head bbox
        bbox = frame_result["head_bbox_px"]
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Draw gaze point and line
        if frame_result["inout"]:
            gx, gy = map(int, frame_result["gaze_point_px"])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            cv2.line(frame, (cx, cy), (gx, gy), color, 2)
            cv2.circle(frame, (gx, gy), 8, color, -1)

        # Add frame info
        cv2.putText(
            frame, f"Frame: {frame_idx} | t={frame_result['timestamp']:.2f}s",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
        )

        out.write(frame)

    cap.release()
    out.release()


def save_results_json(results: Dict, output_path: str) -> None:
    """
    Save video prediction results to JSON file.

    Args:
        results: Output from GazePredictor.predict_video()
        output_path: Path to save JSON file
    """
    # Convert to JSON-serializable format
    json_results = {
        "video_path": results["video_path"],
        "video_fps": results["video_fps"],
        "video_size": results["video_size"],
        "total_frames": results["total_frames"],
        "processed_frames": results["processed_frames"],
        "sample_fps": results["sample_fps"],
        "text_prompt": results["text_prompt"],
        "frames": []
    }

    for frame in results["frames"]:
        frame_data = {
            "frame_idx": frame["frame_idx"],
            "timestamp": frame["timestamp"],
            "gaze_point": list(frame["gaze_point"]),
            "gaze_point_px": list(frame["gaze_point_px"]),
            "inout": frame["inout"],
            "head_bbox": list(frame["head_bbox"]),
            "head_bbox_px": list(frame["head_bbox_px"]),
        }
        json_results["frames"].append(frame_data)

    with open(output_path, 'w') as f:
        json.dump(json_results, f, indent=2)


def load_results_json(json_path: str) -> Dict:
    """
    Load video prediction results from JSON file.

    Args:
        json_path: Path to JSON file

    Returns:
        Results dict (same format as predict_video output, minus heatmaps)
    """
    with open(json_path, 'r') as f:
        results = json.load(f)

    # Convert lists back to tuples
    for frame in results["frames"]:
        frame["gaze_point"] = tuple(frame["gaze_point"])
        frame["gaze_point_px"] = tuple(frame["gaze_point_px"])
        frame["head_bbox"] = tuple(frame["head_bbox"])
        frame["head_bbox_px"] = tuple(frame["head_bbox_px"])

    results["video_size"] = tuple(results["video_size"])
    return results


def main():
    parser = argparse.ArgumentParser(description="GazeAnywhere Inference")
    
    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--image_path", type=str, help="Path to input image")
    input_group.add_argument("--video_path", type=str, help="Path to input video")
    
    # Common arguments
    parser.add_argument("--text", type=str, default="", help="Text prompt describing the person")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT, help="Path to checkpoint")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--no_dark_inference", action="store_true", help="Disable DARK refinement")
    
    # Output arguments
    parser.add_argument("--save_path", type=str, default=None, help="Path to save visualization")
    parser.add_argument("--output_json", type=str, default=None, help="Path to save JSON results (video only)")
    parser.add_argument("--output_video", type=str, default=None, help="Path to save visualization video")
    
    # Video-specific arguments
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for video processing")
    parser.add_argument("--sample_fps", type=float, default=None, help="Sample video at this FPS")
    parser.add_argument("--max_frames", type=int, default=None, help="Maximum frames to process")
    parser.add_argument("--save_heatmaps", action="store_true", help="Include heatmaps in video output")
    
    args = parser.parse_args()

    # Initialize predictor
    predictor = GazePredictor(
        checkpoint_path=args.checkpoint,
        device=args.device,
        use_dark_inference=not args.no_dark_inference
    )

    if args.image_path:
        # Single image prediction
        result = predictor.predict(args.image_path, args.text)

        # Print results
        print(f"Image: {args.image_path}")
        print(f"Image size: {result['image_size']}")
        print(f"Gaze point (normalized): ({result['gaze_point'][0]:.4f}, {result['gaze_point'][1]:.4f})")
        print(f"Gaze point (pixels): ({result['gaze_point_px'][0]:.1f}, {result['gaze_point_px'][1]:.1f})")
        print(f"In-frame: {result['inout']}")
        print(f"Head bbox (normalized): {tuple(f'{x:.4f}' for x in result['head_bbox'])}")
        print(f"Head bbox (pixels): {tuple(f'{x:.1f}' for x in result['head_bbox_px'])}")

        # Visualize
        if args.save_path:
            visualize_gaze(args.image_path, result, args.save_path)
            print(f"Saved visualization to: {args.save_path}")

    else:
        # Video prediction
        print(f"Processing video: {args.video_path}")
        results = predictor.predict_video(
            args.video_path,
            args.text,
            batch_size=args.batch_size,
            sample_fps=args.sample_fps,
            max_frames=args.max_frames,
            return_heatmaps=args.save_heatmaps or args.output_video is not None
        )

        # Print summary
        print(f"\nVideo: {args.video_path}")
        print(f"Video size: {results['video_size']}")
        print(f"Original FPS: {results['video_fps']:.2f}")
        print(f"Sample FPS: {results['sample_fps']:.2f}")
        print(f"Total frames: {results['total_frames']}")
        print(f"Processed frames: {results['processed_frames']}")

        # Calculate in-frame statistics
        inout_count = sum(1 for f in results["frames"] if f["inout"])
        print(f"In-frame predictions: {inout_count}/{len(results['frames'])} ({100*inout_count/len(results['frames']):.1f}%)")

        # Save JSON
        if args.output_json:
            save_results_json(results, args.output_json)
            print(f"Saved JSON results to: {args.output_json}")

        # Create visualization video
        if args.output_video:
            visualize_video(
                args.video_path,
                results,
                args.output_video,
                show_heatmap=args.save_heatmaps
            )
            print(f"Saved visualization video to: {args.output_video}")


if __name__ == "__main__":
    main()
