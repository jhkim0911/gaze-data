#!/usr/bin/env python3
"""
Worker script that processes a single chunk of video frames with SAM3.
This runs in a subprocess so GPU memory is fully freed when it exits.
"""

import argparse
import json
import os
import sys
import gc

import cv2
import numpy as np
import torch

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def process_chunk(
    frames_dir: str,
    output_json: str,
    confidence_threshold: float = 0.4,
):
    """Process a chunk of frames and save results to JSON."""
    
    # Import SAM3 here so it loads fresh each time
    try:
        from sam3.model_builder import build_sam3_video_predictor
        predictor = build_sam3_video_predictor()
        print(f"SAM3 loaded, GPU: {torch.cuda.memory_allocated()/1e9:.1f}GB")
    except ImportError as e:
        print(f"Failed to import SAM3: {e}")
        sys.exit(1)
    
    try:
        # Start session
        response = predictor.start_session(
            resource_path=frames_dir,
            session_id=None,
        )
        session_id = response["session_id"]
        
        # Detect persons
        predictor.add_prompt(
            session_id=session_id,
            frame_idx=0,
            text="people",
        )
        
        # Propagate tracking using handle_stream_request (matches original API)
        tracks = {}
        request = dict(
            type="propagate_in_video",
            session_id=session_id,
            propagation_direction="forward",
            start_frame_index=0,
            max_frame_num_to_track=None,
        )
        propagation = predictor.handle_stream_request(request=request)
        
        for frame_output in propagation:
            frame_idx = frame_output["frame_index"]
            outputs = frame_output["outputs"]
            
            frame_tracks = []  # Always create, even if empty
            
            if outputs is not None:
                obj_ids = outputs.get("out_obj_ids", [])
                scores = outputs.get("out_probs", [])
                boxes_xywh = outputs.get("out_boxes_xywh", [])
                
                for i, obj_id in enumerate(obj_ids):
                    score = scores[i] if i < len(scores) else 1.0
                    if score >= confidence_threshold:
                        if i < len(boxes_xywh):
                            x, y, w, h = boxes_xywh[i]
                            # Convert to Python floats for JSON serialization
                            box = [float(x), float(y), float(x + w), float(y + h)]
                        else:
                            box = [0.0, 0.0, 0.0, 0.0]
                        
                        frame_tracks.append({
                            "person_id": int(obj_id),
                            "bbox": box,
                            "score": float(score),
                        })
            
            # Always add frame to tracks (even if empty)
            tracks[frame_idx] = frame_tracks
        
        # Close session
        predictor.close_session(session_id=session_id)
        
        print(f"Worker: {len(tracks)} frames with detections from propagation")
        
        # Save results
        with open(output_json, 'w') as f:
            json.dump({
                "tracks": {str(k): v for k, v in tracks.items()},
                "num_frames": len(tracks),
            }, f)
        
        print(f"Saved {len(tracks)} frames to {output_json}")
        
    except Exception as e:
        print(f"Error processing chunk: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    finally:
        # Cleanup
        del predictor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames_dir", required=True, help="Directory with frame images")
    parser.add_argument("--output_json", required=True, help="Output JSON path")
    parser.add_argument("--confidence", type=float, default=0.4, help="Confidence threshold")
    
    args = parser.parse_args()
    
    process_chunk(
        frames_dir=args.frames_dir,
        output_json=args.output_json,
        confidence_threshold=args.confidence,
    )


if __name__ == "__main__":
    main()
