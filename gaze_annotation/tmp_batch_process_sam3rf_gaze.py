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
import json
import os
import sys

# Set CUDA memory allocation config to avoid fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Add parent directory to path for running as script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gaze_annotation.tmp_sam3_retinaface_gaze_pipeline import (
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
    parser.add_argument("--manifest", type=str, default=None, help="Manifest JSON with kp_positions for point prompts")

    args = parser.parse_args()

    # Build kp_positions lookup from manifest
    kp_lookup = {}  # clip_name -> [[x,y], ...]
    if args.manifest:
        manifest = json.load(open(args.manifest))
        for entry in manifest:
            kp = entry.get("kp_positions", {})
            if kp:
                kp_lookup[entry["clip_name"]] = [pos for pos in kp.values()]
        print(f"Loaded {len(kp_lookup)} kp_positions from manifest")

    # Find all videos in input_dir
    video_paths = sorted(glob.glob(os.path.join(args.input_dir, "*.mp4")))
    video_paths += sorted(glob.glob(os.path.join(args.input_dir, "**/*.mp4"), recursive=True))
    video_paths = list(dict.fromkeys(video_paths))

    print(f"Found {len(video_paths)} videos in {args.input_dir}")
    
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
        # Encode relative directory path in filename to avoid collisions in flat output dir
        # e.g., input_dir/bbt/season_1/episode_1.mp4 -> output_dir/bbt_season_1_episode_1_sam3rf_gaze.json
        rel_path = os.path.relpath(video_path, args.input_dir)
        video_name = os.path.splitext(rel_path)[0]  # Keep full path without extension
        
        # Replace directory separators with underscores for flat structure
        video_name = video_name.replace(os.sep, '_')
        
        output_json = os.path.join(args.output_dir, f"{video_name}_sam3rf_gaze.json")
        output_video = os.path.join(args.output_dir, f"{video_name}_sam3rf_viz.mp4")
        
        # Skip if already processed
        if os.path.exists(output_json):
            print(f"[{idx+1}/{total}] Skipping {video_name} (exists)")
            skipped += 1
            continue
        
        # Look up GT points for this clip
        points = kp_lookup.get(video_name)
        prompt_info = f" (points: {len(points)})" if points else ""
        print(f"[{idx+1}/{total}] Processing: {video_name}{prompt_info}")

        try:
            results = annotator.annotate_video(
                video_path=video_path,
                max_frames=args.max_frames,
                progress_bar=True,
                sample_fps=args.sample_fps,
                points=points,
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
            
            # CRITICAL: Reset even on error to free GPU memory
            try:
                annotator.reset()
            except:
                pass
            
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            print(f"  [GPU Memory after error cleanup: {torch.cuda.memory_allocated()/1e9:.1f}GB]")
            continue
    
    print(f"\nDone! Processed: {processed}, Skipped: {skipped}, Errors: {errors}")


if __name__ == "__main__":
    main()
