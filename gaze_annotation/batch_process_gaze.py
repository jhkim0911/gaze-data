#!/usr/bin/env python3
"""
Batch processing script for RetinaFace + SAM3 + GazeAnywhere pipeline.

Processes all videos in a directory and saves results to output directory.

Usage:
    python batch_process_gaze.py \
        --input_dir /path/to/videos \
        --output_dir /path/to/output
"""

import argparse
import glob
import os
import sys

# Add parent directory to path for running as script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gaze_annotation.retinaface_perframe_pipeline import (
    RetinaFacePerFrameAnnotator,
    visualize_annotations,
)
import torch


def main():
    parser = argparse.ArgumentParser(description="Batch process videos with gaze estimation")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing input videos")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save outputs")
    parser.add_argument("--det_thresh", type=float, default=0.6, help="Face detection threshold")
    parser.add_argument("--sample_fps", type=float, default=2.0, help="Process at this fps (default: 2.0)")
    parser.add_argument("--no_visualization", action="store_true", help="Skip visualization videos")
    parser.add_argument("--max_frames", type=int, default=None, help="Max frames per video")
    parser.add_argument("--gaze_checkpoint", type=str,
                        default="/projects/illinois/eng/cs/jrehg/users/xucao2/ChildGaze/checkpoints/GazeAnywhere/gazeanywhere.pth")
    
    args = parser.parse_args()
    
    # Find all videos
    video_paths = sorted(glob.glob(os.path.join(args.input_dir, "*.mp4")))
    video_paths += sorted(glob.glob(os.path.join(args.input_dir, "*.avi")))
    video_paths += sorted(glob.glob(os.path.join(args.input_dir, "*.mov")))
    
    print(f"Found {len(video_paths)} videos in {args.input_dir}")
    print(f"Processing at {args.sample_fps} fps")
    
    if len(video_paths) == 0:
        print("No videos found!")
        return
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize annotator once
    print("Initializing models...")
    annotator = RetinaFacePerFrameAnnotator(
        device="cuda" if torch.cuda.is_available() else "cpu",
        gaze_checkpoint=args.gaze_checkpoint,
        det_thresh=args.det_thresh,
    )
    
    # Process each video
    total = len(video_paths)
    processed = 0
    skipped = 0
    errors = 0
    
    for idx, video_path in enumerate(video_paths):
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        output_json = os.path.join(args.output_dir, f"{video_name}_gaze.json")
        output_video = os.path.join(args.output_dir, f"{video_name}_viz.mp4")
        
        # Skip if already processed
        if os.path.exists(output_json):
            print(f"[{idx+1}/{total}] Skipping {video_name} (exists)")
            skipped += 1
            continue
        
        print(f"[{idx+1}/{total}] Processing: {video_name}")
        
        try:
            results = annotator.annotate_video(
                video_path=video_path,
                max_frames=args.max_frames,
                progress_bar=True,
                sample_fps=args.sample_fps,
            )
            
            annotator.save_results(results, output_json)
            print(f"  -> {results['num_unique_faces']} faces, {results['processed_frames']} frames")
            
            if not args.no_visualization and results["processed_frames"] > 0:
                visualize_annotations(
                    video_path=video_path,
                    results=results,
                    output_path=output_video,
                    progress_bar=False,
                )
            
            processed += 1
            
            # Reset tracker for next video
            annotator.reset_tracker()
            
        except Exception as e:
            print(f"  ERROR: {e}")
            errors += 1
            continue
    
    print(f"\nDone! Processed: {processed}, Skipped: {skipped}, Errors: {errors}")


if __name__ == "__main__":
    main()
