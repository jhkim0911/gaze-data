#!/usr/bin/env python3
"""
Worker script that processes a single chunk of video frames with SAM3.
Uses GT point prompts (kp_positions) when available, falls back to text prompt.

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


def _propagate_and_collect(predictor, session_id, confidence_threshold):
    """Run forward propagation and collect tracks."""
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

        frame_tracks = []

        if outputs is not None:
            obj_ids = outputs.get("out_obj_ids", [])
            scores = outputs.get("out_probs", [])
            boxes_xywh = outputs.get("out_boxes_xywh", [])

            for i, obj_id in enumerate(obj_ids):
                score = scores[i] if i < len(scores) else 1.0
                if score >= confidence_threshold:
                    if i < len(boxes_xywh):
                        x, y, w, h = boxes_xywh[i]
                        box = [float(x), float(y), float(x + w), float(y + h)]
                    else:
                        box = [0.0, 0.0, 0.0, 0.0]

                    frame_tracks.append({
                        "person_id": int(obj_id),
                        "bbox": box,
                        "score": float(score),
                    })

        tracks[frame_idx] = frame_tracks

    return tracks


def process_chunk(
    frames_dir: str,
    output_json: str,
    confidence_threshold: float = 0.4,
    points: list = None,
):
    """Process a chunk of frames. Use point prompts if provided, else text fallback."""

    # Import SAM3 here so it loads fresh each time
    try:
        from sam3.model_builder import build_sam3_video_predictor
        predictor = build_sam3_video_predictor()
        print(f"SAM3 loaded, GPU: {torch.cuda.memory_allocated()/1e9:.1f}GB")
    except ImportError as e:
        print(f"Failed to import SAM3: {e}")
        sys.exit(1)

    num_frames = len([f for f in os.listdir(frames_dir) if f.endswith('.jpg')])

    try:
        tracks = {}

        response = predictor.start_session(
            resource_path=frames_dir,
            session_id=None,
        )
        session_id = response["session_id"]

        if points:
            # Use text="person" + one box as visual example (combined prompt)
            # Pick the first point, create a box around it as visual hint
            cx, cy = points[0]
            box_half = 0.15  # 15% of image around the point
            x = max(0.0, cx - box_half)
            y = max(0.0, cy - box_half)
            w = min(box_half * 2, 1.0 - x)
            h = min(box_half * 2, 1.0 - y)
            predictor.add_prompt(
                session_id=session_id,
                frame_idx=0,
                text="person",
                bounding_boxes=[[x, y, w, h]],
                bounding_box_labels=[1],
            )
            prompt_desc = f"text+box prompt (box at [{cx:.2f},{cy:.2f}])"
        else:
            # Fallback to text prompt only
            predictor.add_prompt(
                session_id=session_id,
                frame_idx=0,
                text="person",
            )
            prompt_desc = "text prompt"

        tracks = _propagate_and_collect(predictor, session_id, confidence_threshold)
        total_detections = sum(len(t) for t in tracks.values())
        predictor.close_session(session_id=session_id)
        print(f"Worker: {prompt_desc} -> {total_detections} detections across {len(tracks)} frames")

        if total_detections == 0:
            print(f"Worker: no detections")

        print(f"Worker: {len(tracks)} frames from propagation")

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
        del predictor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames_dir", required=True, help="Directory with frame images")
    parser.add_argument("--output_json", required=True, help="Output JSON path")
    parser.add_argument("--confidence", type=float, default=0.4, help="Confidence threshold")
    parser.add_argument("--points", type=str, default=None, help="JSON string of [[x,y],...] normalized points")

    args = parser.parse_args()

    pts = json.loads(args.points) if args.points else None

    process_chunk(
        frames_dir=args.frames_dir,
        output_json=args.output_json,
        confidence_threshold=args.confidence,
        points=pts,
    )


if __name__ == "__main__":
    main()
