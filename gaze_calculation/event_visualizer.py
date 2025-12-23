#!/usr/bin/env python3
"""
Event Visualizer for Manual Inspection

Overlays detected events (from candidate_event_detector) on visualization videos
for human inspection and validation.

Takes 2fps visualization videos with gaze annotations and adds:
- Event type text when event is occurring
- Timestamp display
- Persons involved highlighting

Usage:
    python event_visualizer.py \
        --viz_video /path/to/video_viz.mp4 \
        --events_json /path/to/events.json \
        --output_video /path/to/output.mp4
        
Or for batch processing:
    python event_visualizer.py \
        --video_dir /path/to/videos \
        --output_dir /path/to/output
"""

import argparse
import json
import os
import glob
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np


# Colors for different event types (BGR)
EVENT_COLORS = {
    "sudden_gaze_shift": (0, 165, 255),    # Orange
    "joint_attention": (0, 255, 0),         # Green
    "gaze_following": (255, 255, 0),        # Cyan
    "attention_capture": (0, 0, 255),       # Red
    "mutual_gaze": (255, 0, 255),           # Magenta
}

# Short labels for display
EVENT_SHORT_LABELS = {
    "sudden_gaze_shift": "GAZE SHIFT",
    "joint_attention": "JOINT ATTN",
    "gaze_following": "GAZE FOLLOW",
    "attention_capture": "ATTN CAPTURE",
    "mutual_gaze": "MUTUAL GAZE",
}


def get_events_at_time(events: List[Dict], time_sec: float) -> List[Dict]:
    """Get all events occurring at a given time."""
    active_events = []
    for event in events:
        start = event.get("start_time", 0)
        end = event.get("end_time", 0)
        if start <= time_sec <= end:
            active_events.append(event)
    return active_events


def draw_event_overlay(
    frame: np.ndarray,
    events: List[Dict],
    time_sec: float,
    frame_idx: int,
    classifications: Optional[Dict] = None,
) -> np.ndarray:
    """Draw event information overlay on frame.
    
    Note: The input viz video already has Frame/time/persons info in top-left,
    so we only add event info in top-right to avoid duplication.
    """
    h, w = frame.shape[:2]
    
    if not events:
        return frame
    
    # Calculate box height based on content
    box_h = 30
    for event in events:
        box_h += 35
        event_id = event.get("event_id")
        if classifications and event_id in classifications:
            box_h += 60  # Extra space for Gemini info
    
    # Draw event box in top-right
    box_x = w - 320
    box_y = 10
    
    # Semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(overlay, (box_x, box_y), (w - 10, box_y + box_h), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
    
    y_pos = box_y + 25
    
    # Draw each event
    for event in events:
        event_type = event.get("event_type", "unknown")
        persons = event.get("persons_involved", [])
        event_id = event.get("event_id")
        
        color = EVENT_COLORS.get(event_type, (255, 255, 255))
        label = EVENT_SHORT_LABELS.get(event_type, event_type.upper())
        
        # Event type with colored indicator
        cv2.circle(frame, (box_x + 15, y_pos - 5), 8, color, -1)
        cv2.putText(
            frame, label,
            (box_x + 30, y_pos),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA
        )
        
        # Persons involved
        persons_text = f"P{', P'.join(map(str, persons))}" if persons else ""
        cv2.putText(
            frame, persons_text,
            (box_x + 180, y_pos),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA
        )
        
        y_pos += 25
        
        # Add Gemini classification info if available
        if classifications and event_id in classifications:
            cls = classifications[event_id]
            
            # Validation status
            confirmed = cls.get("event_confirmed", False)
            status_color = (0, 255, 0) if confirmed else (0, 0, 255)
            status_text = "✓ CONFIRMED" if confirmed else "✗ REJECTED"
            cv2.putText(
                frame, status_text,
                (box_x + 30, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1, cv2.LINE_AA
            )
            y_pos += 20
            
            # Deictic gesture
            if cls.get("deictic_gesture_detected"):
                gesture_type = cls.get("deictic_gesture_type", "")
                initiator = cls.get("initiator_id")
                gesture_text = f"GESTURE: {gesture_type}"
                if initiator is not None:
                    gesture_text += f" by P{initiator}"
                cv2.putText(
                    frame, gesture_text,
                    (box_x + 30, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA
                )
                y_pos += 20
                
                # Target info
                target_type = cls.get("gesture_target_type")
                if target_type == "person":
                    target_id = cls.get("gesture_target_person_id")
                    cv2.putText(
                        frame, f"→ P{target_id}",
                        (box_x + 30, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1, cv2.LINE_AA
                    )
                elif target_type == "object":
                    target_desc = cls.get("gesture_target_description", "")[:20]
                    cv2.putText(
                        frame, f"→ {target_desc}",
                        (box_x + 30, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1, cv2.LINE_AA
                    )
                y_pos += 20
            
            # Responders
            responders = cls.get("responder_ids", [])
            if responders:
                resp_text = f"Responders: P{', P'.join(map(str, responders))}"
                cv2.putText(
                    frame, resp_text,
                    (box_x + 30, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 255), 1, cv2.LINE_AA
                )
                y_pos += 20
        
        y_pos += 10  # Gap between events
    
    return frame


def visualize_events(
    viz_video_path: str,
    events_json_path: str,
    output_path: str,
    gestures_json_path: Optional[str] = None,
    fps: float = 2.0,
) -> None:
    """
    Create visualization video with event overlays.
    
    Args:
        viz_video_path: Path to existing visualization video (2fps with gaze annotations)
        events_json_path: Path to events JSON from candidate_event_detector
        output_path: Output video path
        gestures_json_path: Optional path to Gemini gesture classifications JSON
        fps: Video FPS (default 2.0 for pre-sampled videos)
    """
    # Load events
    with open(events_json_path, 'r') as f:
        events_data = json.load(f)
    events = events_data.get("events", [])
    
    print(f"Loaded {len(events)} events from: {events_json_path}")
    
    # Load Gemini classifications if available
    classifications = {}
    if gestures_json_path and os.path.exists(gestures_json_path):
        with open(gestures_json_path, 'r') as f:
            gestures_data = json.load(f)
        for cls in gestures_data.get("classifications", []):
            classifications[cls["event_id"]] = cls
        print(f"Loaded {len(classifications)} Gemini classifications from: {gestures_json_path}")
    
    # Open video
    cap = cv2.VideoCapture(viz_video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video: {viz_video_path}")
        return
    
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video: {total_frames} frames @ {video_fps:.1f}fps, {width}x{height}")
    
    # Create output
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, video_fps, (width, height))
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Calculate time (assuming video is already at target fps)
        time_sec = frame_idx / video_fps
        
        # Get active events at this time
        active_events = get_events_at_time(events, time_sec)
        
        # Draw overlay (with Gemini classifications if available)
        frame = draw_event_overlay(frame, active_events, time_sec, frame_idx, classifications)
        
        out.write(frame)
        frame_idx += 1
        
        if frame_idx % 100 == 0:
            print(f"Processed {frame_idx}/{total_frames} frames")
    
    cap.release()
    out.release()
    
    print(f"Saved: {output_path}")
    
    # Re-encode with H.264 for better compatibility
    _reencode_h264(output_path)


def _reencode_h264(video_path: str) -> None:
    """Re-encode video with H.264 for browser compatibility."""
    import subprocess
    
    temp_path = video_path + ".temp.mp4"
    os.rename(video_path, temp_path)
    
    try:
        subprocess.run([
            "ffmpeg", "-y", "-i", temp_path,
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-c:a", "copy",
            video_path
        ], check=True, capture_output=True)
        os.remove(temp_path)
        print(f"Re-encoded with H.264: {video_path}")
    except Exception as e:
        print(f"Warning: H.264 re-encoding failed, keeping original: {e}")
        os.rename(temp_path, video_path)


def batch_visualize(
    video_dir: str,
    output_dir: str,
) -> None:
    """
    Batch process all visualization videos in a directory.
    
    Expects files like:
    - *_sam3rf_viz.mp4 (visualization video)
    - *_events.json (detected events)
    """
    # Find all viz videos
    viz_videos = glob.glob(os.path.join(video_dir, "*_viz.mp4"))
    viz_videos += glob.glob(os.path.join(video_dir, "*_sam3rf_viz.mp4"))
    viz_videos = list(set(viz_videos))  # Remove duplicates
    
    print(f"Found {len(viz_videos)} visualization videos")
    
    os.makedirs(output_dir, exist_ok=True)
    
    for viz_path in viz_videos:
        # Find corresponding events JSON
        base_name = os.path.basename(viz_path)
        # Try different naming conventions
        events_candidates = [
            viz_path.replace("_viz.mp4", "_events.json"),
            viz_path.replace("_sam3rf_viz.mp4", "_events.json"),
            viz_path.replace("_viz.mp4", "_features.json").replace("_features", "_events"),
        ]
        
        events_path = None
        for candidate in events_candidates:
            if os.path.exists(candidate):
                events_path = candidate
                break
        
        if not events_path:
            print(f"Skipping {base_name}: no events JSON found")
            continue
        
        output_name = base_name.replace("_viz.mp4", "_events_viz.mp4")
        output_name = output_name.replace("_sam3rf_viz.mp4", "_events_viz.mp4")
        output_path = os.path.join(output_dir, output_name)
        
        print(f"\nProcessing: {base_name}")
        visualize_events(viz_path, events_path, output_path)


def main():
    parser = argparse.ArgumentParser(description="Visualize detected events on videos")
    parser.add_argument("--viz_video", type=str, default=None, help="Single visualization video")
    parser.add_argument("--events_json", type=str, default=None, help="Events JSON for single video")
    parser.add_argument("--gestures_json", type=str, default=None, help="Gemini gestures JSON (optional)")
    parser.add_argument("--output_video", type=str, default=None, help="Output path for single video")
    parser.add_argument("--video_dir", type=str, default=None, help="Directory for batch processing")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for batch processing")
    
    args = parser.parse_args()
    
    if args.viz_video:
        # Single video mode
        if not args.events_json:
            # Try to auto-detect
            args.events_json = args.viz_video.replace("_viz.mp4", "_events.json")
            args.events_json = args.events_json.replace("_sam3rf_viz.mp4", "_events.json")
        
        if not args.gestures_json:
            # Try to auto-detect Gemini gestures
            gestures_candidate = args.viz_video.replace("_viz.mp4", "_gestures.json")
            gestures_candidate = gestures_candidate.replace("_sam3rf_viz.mp4", "_gestures.json")
            if os.path.exists(gestures_candidate):
                args.gestures_json = gestures_candidate
        
        if not args.output_video:
            suffix = "_full_viz.mp4" if args.gestures_json else "_events_viz.mp4"
            args.output_video = args.viz_video.replace("_viz.mp4", suffix)
            args.output_video = args.output_video.replace("_sam3rf_viz.mp4", suffix)
        
        visualize_events(args.viz_video, args.events_json, args.output_video, args.gestures_json)
    
    elif args.video_dir:
        # Batch mode
        if not args.output_dir:
            args.output_dir = os.path.join(args.video_dir, "events_viz")
        
        batch_visualize(args.video_dir, args.output_dir)
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
