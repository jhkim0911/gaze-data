#!/usr/bin/env python3
"""
Batch processing script for SAM3 Person Tracking + RetinaFace + GazeAnywhere pipeline.

Processes all videos in a directory and saves results to output directory.

Usage:
    python batch_process_sam3rf_gaze.py \
        --input_dir /path/to/videos \
        --output_dir /path/to/output
"""

import argparse
import glob
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sam3_retinaface_gaze_pipeline import (
    Sam3RetinaFaceGazeAnnotator,
    visualize_annotations,
)
import torch


def main():
    parser = argparse.ArgumentParser(description="Batch process videos with SAM3+RetinaFace+Gaze pipeline")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing input videos")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save outputs")
    parser.add_argument("--sam3_confidence", type=float, default=0.45, help="SAM3 person detection confidence")
    parser.add_argument("--face_det_thresh", type=float, default=0.54, help="RetinaFace detection threshold")
    parser.add_argument("--no_visualization", action="store_true", help="Skip visualization videos")
    parser.add_argument("--max_frames", type=int, default=None, help="Max frames per video")
    parser.add_argument("--sample_fps", type=float, default=2.0, help="Process at this fps (default: 2.0)")
    parser.add_argument("--gaze_checkpoint", type=str,
                        default="/projects/illinois/eng/cs/jrehg/users/xucao2/ChildGaze/checkpoints/GazeAnywhere/gazeanywhere.pth")
    
    args = parser.parse_args()
    
    # Find all videos
    video_paths = sorted(glob.glob(os.path.join(args.input_dir, "*.mp4")))
    video_paths += sorted(glob.glob(os.path.join(args.input_dir, "*.avi")))
    video_paths += sorted(glob.glob(os.path.join(args.input_dir, "*.mov")))
    
    print(f"Found {len(video_paths)} videos in {args.input_dir}")
    
    if len(video_paths) == 0:
        print("No videos found!")
        return
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize annotator once
    print("Initializing models...")
    annotator = Sam3RetinaFaceGazeAnnotator(
        device="cuda" if torch.cuda.is_available() else "cpu",
        gaze_checkpoint=args.gaze_checkpoint,
        sam3_confidence=args.sam3_confidence,
        face_det_thresh=args.face_det_thresh,
    )
    
    # Process each video
    total = len(video_paths)
    processed = 0
    skipped = 0
    errors = 0
    
    for idx, video_path in enumerate(video_paths):
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        output_json = os.path.join(args.output_dir, f"{video_name}_sam3rf_gaze.json")
        output_video = os.path.join(args.output_dir, f"{video_name}_sam3rf_viz.mp4")
        
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
            print(f"  -> {len(results['persons_summary'])} persons, {results['processed_frames']} frames")
            
            if not args.no_visualization and results["processed_frames"] > 0:
                visualize_annotations(
                    video_path=video_path,
                    results=results,
                    output_path=output_video,
                    progress_bar=False,
                )
            
            processed += 1
            
            # Aggressive memory cleanup for next video
            del results  # Free results dict
            annotator.reset()
            import gc
            gc.collect()
            
            # Also clear CUDA cache explicitly
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            errors += 1
            continue
    
    print(f"\nDone! Processed: {processed}, Skipped: {skipped}, Errors: {errors}")


if __name__ == "__main__":
    main()
