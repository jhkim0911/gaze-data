#!/usr/bin/env python3
"""
Lightweight Person BBox Visualization Script

Creates visualization videos with only person bounding boxes (not face bboxes)
using existing gaze annotation JSON files. Designed for minimal visual footprint
to support downstream Gemini gesture understanding.

Usage:
    # Single dataset
    python visualize_person_bbox.py --dataset social_gesture
    
    # Multiple datasets
    python visualize_person_bbox.py --dataset social_gesture tvqa
    
    # All datasets
    python visualize_person_bbox.py --dataset all
    
    # Dry run (list files only)
    python visualize_person_bbox.py --dataset social_gesture --dry_run
"""

import argparse
import glob
import json
import os
import subprocess
import sys
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm


# Base path for all datasets
BASE_PATH = "/projects/illinois/eng/cs/jrehg/users/arkimjh/gaze_social"

# Dataset configurations: {name: {"input": ..., "output": ...}}
DATASETS = {
    "avsbench": {
        "input": f"{BASE_PATH}/avsbench/gaze_videos",
        "output": f"{BASE_PATH}/avsbench/bbox_videos",
    },
    "embody3d": {
        "input": f"{BASE_PATH}/embody3d/gaze_videos",
        "output": f"{BASE_PATH}/embody3d/bbox_videos",
    },
    "friendsmmc": {
        "input": f"{BASE_PATH}/friendsmmc/gaze_videos",
        "output": f"{BASE_PATH}/friendsmmc/bbox_videos",
    },
    "social_gesture": {
        "input": f"{BASE_PATH}/social_gesture/gaze_videos",
        "output": f"{BASE_PATH}/social_gesture/bbox_videos",
    },
    "social-iq": {
        "input": f"{BASE_PATH}/social-iq/gaze_videos",
        "output": f"{BASE_PATH}/social-iq/bbox_videos",
    },
    "tvqa": {
        "input": f"{BASE_PATH}/tvqa/gaze_videos",
        "output": f"{BASE_PATH}/tvqa/bbox_videos",
    },
    "werewolf": {
        "input": f"{BASE_PATH}/werewolf/gaze_videos",
        "output": f"{BASE_PATH}/werewolf/bbox_videos",
    },
}

# Color palette (BGR format for OpenCV) - 10 colors for up to 10 people
COLORS = [
    (0, 255, 0),    # Green
    (255, 128, 0),  # Blue-ish
    (0, 128, 255),  # Orange
    (255, 255, 0),  # Cyan
    (255, 0, 255),  # Magenta
    (0, 255, 255),  # Yellow
    (128, 0, 255),  # Purple
    (255, 0, 128),  # Pink
    (128, 255, 0),  # Lime green
    (0, 64, 255),   # Dark orange
]


def draw_person_bbox_fast(
    frame: np.ndarray,
    bbox_px: List[int],
    person_id: int,
    color: Tuple[int, int, int],
    line_thickness: int = 1,
    show_bbox: bool = True,
) -> None:
    """
    Draw person bounding box with ID label directly (no alpha blending for speed).
    
    Args:
        frame: BGR image (modified in-place)
        bbox_px: [x1, y1, x2, y2] in pixels
        person_id: Person ID for labeling
        color: BGR color tuple
        line_thickness: Line thickness in pixels
        show_bbox: Whether to draw the bbox rectangle (False = label only)
    """
    x1, y1, x2, y2 = map(int, bbox_px)
    
    # Draw bbox rectangle (optional)
    if show_bbox:
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, line_thickness, cv2.LINE_AA)
    
    # Draw person ID label (modest size - visible but not dominant)
    label = f"P{person_id}"
    font_scale = 0.6  # Modest size for visibility
    font_thickness = 2
    (text_w, text_h), baseline = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
    )
    
    # Label background (centered horizontally)
    # If no room above bbox, draw inside bbox instead
    bbox_center_x = (x1 + x2) // 2
    label_x1 = bbox_center_x - (text_w + 6) // 2
    label_x2 = label_x1 + text_w + 6
    
    label_height = text_h + 8
    if y1 >= label_height:
        # Draw above bbox (normal case)
        label_y1 = y1 - label_height
        label_y2 = y1
        text_y = y1 - 4
    else:
        # No room above - draw inside bbox at top
        label_y1 = y1
        label_y2 = y1 + label_height
        text_y = y1 + text_h + 4
    
    # Draw label background
    cv2.rectangle(frame, (label_x1, label_y1), (label_x2, label_y2), color, -1)
    
    # Draw label text (black for contrast)
    cv2.putText(
        frame, label, (label_x1 + 3, text_y),
        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), font_thickness, cv2.LINE_AA
    )


def draw_gaze_line(
    frame: np.ndarray,
    face_bbox_px: List[int],
    gaze_point_px: List[float],
    inout: bool,
    color: Tuple[int, int, int],
    line_thickness: int = 1,
) -> None:
    """
    Draw gaze line from face center to gaze point.
    
    Args:
        frame: BGR image (modified in-place)
        face_bbox_px: [x1, y1, x2, y2] face bbox in pixels
        gaze_point_px: [x, y] gaze point in pixels
        inout: True if looking inside the frame
        color: BGR color tuple
        line_thickness: Line thickness in pixels
    """
    if not inout or not gaze_point_px:
        return
    
    # Face center as gaze origin
    fx1, fy1, fx2, fy2 = map(int, face_bbox_px)
    fcx, fcy = int((fx1 + fx2) / 2), int((fy1 + fy2) / 2)
    
    gx, gy = int(gaze_point_px[0]), int(gaze_point_px[1])
    
    # Draw gaze line
    cv2.line(frame, (fcx, fcy), (gx, gy), color, line_thickness, cv2.LINE_AA)
    
    # Draw gaze point circle
    cv2.circle(frame, (gx, gy), 4, color, -1, cv2.LINE_AA)
    cv2.circle(frame, (gx, gy), 4, (255, 255, 255), 1, cv2.LINE_AA)


def reencode_video_h264(video_path: str) -> None:
    """Re-encode video to H.264 for better compatibility (uses ultrafast preset)."""
    temp_path = video_path.replace(".mp4", "_temp.mp4")
    
    try:
        cmd = [
            "ffmpeg", "-y", "-i", video_path,
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
            "-pix_fmt", "yuv420p",
            "-an",  # No audio
            temp_path
        ]
        subprocess.run(cmd, capture_output=True, check=True)
        os.replace(temp_path, video_path)
    except subprocess.CalledProcessError:
        # If ffmpeg fails, keep original
        if os.path.exists(temp_path):
            os.remove(temp_path)


def process_single_video(
    json_path: str,
    output_dir: str,
    line_thickness: int = 1,
    alpha: float = 0.6,
    skip_existing: bool = True,
    reencode: bool = True,
    show_gaze: bool = False,
    show_bbox: bool = True,
    show_info: bool = True,
) -> bool:
    """
    Process a single JSON annotation file and create bbox visualization.
    
    Args:
        json_path: Path to *_sam3rf_gaze.json file
        output_dir: Directory to save output video
        line_thickness: Line thickness for bbox and gaze lines
        alpha: Transparency for overlay (0=invisible, 1=opaque)
        skip_existing: Skip if output already exists
        reencode: Whether to re-encode to H.264 (slower but more compatible)
        show_gaze: Whether to draw gaze lines
        show_bbox: Whether to draw bbox rectangles (False = labels only)
    
    Returns:
        True if processed successfully, False otherwise
    """
    # Determine output path (replace _sam3rf_gaze.json with _bbox_viz.mp4)
    json_name = os.path.basename(json_path)
    video_name = json_name.replace("_sam3rf_gaze.json", "_bbox_viz.mp4")
    output_path = os.path.join(output_dir, video_name)
    
    if skip_existing and os.path.exists(output_path):
        return True
    
    try:
        # Load JSON
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        video_path = data.get("video_path")
        if not video_path or not os.path.exists(video_path):
            print(f"  WARNING: Video not found: {video_path}")
            return False
        
        frames_data = data.get("frames", [])
        if not frames_data:
            print(f"  WARNING: No frames in JSON: {json_path}")
            return False
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"  WARNING: Cannot open video: {video_path}")
            return False
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Use sample_fps for output (matches annotation sampling rate)
        output_fps = data.get("sample_fps", data.get("video_fps", 2.0))
        
        # Create output directory if needed
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, output_fps, (width, height))
        
        # Process each annotated frame
        for frame_data in frames_data:
            frame_idx = frame_data["frame_idx"]
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if not ret:
                continue
            
            # Create overlay for efficient per-frame alpha blending (only one blend per frame)
            overlay = frame.copy()
            
            # Draw person bboxes on overlay
            for person in frame_data.get("persons", []):
                person_id = person.get("person_id", 0)
                body_bbox_px = person.get("body_bbox_px")
                
                if body_bbox_px:
                    color = COLORS[person_id % len(COLORS)]
                    draw_person_bbox_fast(
                        overlay, body_bbox_px, person_id, color,
                        line_thickness=line_thickness,
                        show_bbox=show_bbox
                    )
                    
                    # Draw gaze line if enabled
                    if show_gaze:
                        face_bbox_px = person.get("face_bbox_px")
                        gaze_point_px = person.get("gaze_point_px")
                        inout = person.get("inout", False)
                        face_detected = person.get("face_detected", False)
                        
                        if face_detected and face_bbox_px and gaze_point_px:
                            draw_gaze_line(
                                overlay, face_bbox_px, gaze_point_px, inout, color,
                                line_thickness=line_thickness
                            )
            
            # Blend overlay with original frame (single blend for all bboxes)
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
            
            # Draw frame info overlay (timestamp, person count) - after blending so it's fully visible
            if show_info:
                timestamp = frame_data.get("timestamp", 0.0)
                num_persons = len(frame_data.get("persons", []))
                info_text = f"t={timestamp:.2f}s | {num_persons} person(s)"
                (tw, th), _ = cv2.getTextSize(info_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                cv2.rectangle(frame, (5, 5), (15 + tw, 35), (0, 0, 0), -1)
                cv2.putText(frame, info_text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            
            out.write(frame)
        
        cap.release()
        out.release()
        
        # Re-encode to H.264 for compatibility (optional)
        if reencode:
            reencode_video_h264(output_path)
        
        return True
        
    except Exception as e:
        print(f"  ERROR processing {json_path}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Create lightweight person bbox visualization videos from gaze annotation JSONs"
    )
    parser.add_argument(
        "--dataset", nargs="+", required=True,
        help=f"Dataset name(s) to process. Options: {', '.join(DATASETS.keys())}, all"
    )
    parser.add_argument(
        "--line_thickness", type=int, default=1,
        help="Line thickness for bbox (default: 1)"
    )
    parser.add_argument(
        "--alpha", type=float, default=0.6,
        help="Transparency 0-1, higher=more visible (default: 0.6)"
    )
    parser.add_argument(
        "--gaze", action="store_true",
        help="Draw gaze lines from face center to gaze point"
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="List files only, don't process"
    )
    parser.add_argument(
        "--no_skip", action="store_true",
        help="Re-process even if output exists"
    )
    parser.add_argument(
        "--no_bbox", action="store_true",
        help="Don't draw bbox rectangles (show person ID labels only)"
    )
    parser.add_argument(
        "--no_reencode", action="store_true",
        help="Skip H.264 re-encoding (faster but less compatible)"
    )
    parser.add_argument(
        "--no_info", action="store_true",
        help="Don't draw frame info overlay (timestamp, person count)"
    )
    parser.add_argument(
        "--output_subdir", type=str, default=None,
        help="Override output subdirectory name (default: auto based on flags)"
    )
    parser.add_argument(
        "--input_subdir", type=str, default=None,
        help="Override input subdirectory name (default: dataset's gaze_videos). "
             "e.g. gaze_videos_5fps for a higher-fps re-render."
    )
    parser.add_argument(
        "--split", type=int, default=1,
        help="Total number of parallel splits (default: 1)"
    )
    parser.add_argument(
        "--split_id", type=int, default=1,
        help="Which split to run (1-indexed, 1 to --split)"
    )
    
    args = parser.parse_args()
    
    # Expand "all" to all datasets
    if "all" in args.dataset:
        datasets_to_process = list(DATASETS.keys())
    else:
        datasets_to_process = args.dataset
        # Validate dataset names
        for ds in datasets_to_process:
            if ds not in DATASETS:
                print(f"ERROR: Unknown dataset '{ds}'. Valid options: {', '.join(DATASETS.keys())}")
                sys.exit(1)
    
    print(f"Datasets to process: {datasets_to_process}")
    
    # Determine output subdirectory based on options
    # Default: bbox_videos, --gaze: o_gaze_videos, --no_bbox: nbbox_videos
    if args.output_subdir:
        output_subdir = args.output_subdir
    elif args.no_bbox:
        output_subdir = "nbbox_videos"
    elif args.gaze:
        output_subdir = "o_gaze_videos"
    else:
        output_subdir = "bbox_videos"
    
    print(f"Output subdirectory: {output_subdir}")
    
    # Collect all JSON files
    all_json_files = []
    for ds_name in datasets_to_process:
        if args.input_subdir:
            input_dir = os.path.join(BASE_PATH, ds_name, args.input_subdir)
        else:
            input_dir = DATASETS[ds_name]["input"]
        # Construct output_dir from base path + dataset + output_subdir
        output_dir = os.path.join(BASE_PATH, ds_name, output_subdir)
        
        if not os.path.exists(input_dir):
            print(f"WARNING: Input directory not found: {input_dir}")
            continue
        
        json_files = sorted(glob.glob(os.path.join(input_dir, "*_sam3rf_gaze.json")))
        for jf in json_files:
            all_json_files.append((jf, output_dir))
    
    print(f"Found {len(all_json_files)} JSON files total")
    
    # Apply split filtering for parallel processing
    if args.split > 1:
        if args.split_id < 1 or args.split_id > args.split:
            print(f"ERROR: --split_id must be between 1 and {args.split}")
            sys.exit(1)
        all_json_files = [
            jf for i, jf in enumerate(all_json_files)
            if i % args.split == (args.split_id - 1)
        ]
        print(f"Split {args.split_id}/{args.split}: Processing {len(all_json_files)} files")
    
    if args.dry_run:
        print("\n=== DRY RUN: Files to process ===")
        for json_path, output_dir in all_json_files:
            print(f"  {json_path}")
            print(f"    -> {output_dir}/")
        print(f"\nTotal: {len(all_json_files)} files")
        return
    
    # Process all files
    processed = 0
    skipped = 0
    errors = 0
    
    pbar = tqdm(all_json_files, desc="Processing videos")
    for json_path, output_dir in pbar:
        json_name = os.path.basename(json_path)
        pbar.set_postfix_str(json_name[:40])
        
        # Check if already exists
        video_name = json_name.replace("_sam3rf_gaze.json", "_bbox_viz.mp4")
        output_path = os.path.join(output_dir, video_name)
        
        if not args.no_skip and os.path.exists(output_path):
            skipped += 1
            continue
        
        success = process_single_video(
            json_path, output_dir,
            line_thickness=args.line_thickness,
            alpha=args.alpha,
            skip_existing=not args.no_skip,
            reencode=not args.no_reencode,
            show_gaze=args.gaze,
            show_bbox=not args.no_bbox,
            show_info=not args.no_info,
        )
        
        if success:
            processed += 1
        else:
            errors += 1
    
    print(f"\n=== Done ===")
    print(f"Processed: {processed}, Skipped: {skipped}, Errors: {errors}")


if __name__ == "__main__":
    main()
