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

# Colors for gesture types (BGR)
GESTURE_COLORS = {
    "pointing": (255, 165, 0),     # Orange
    "showing": (0, 255, 255),      # Yellow
    "giving": (0, 215, 255),       # Gold
    "reaching": (147, 20, 255),    # Pink
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
    
    Note: The input viz video already has Frame/time/persons info in top-left
    (roughly y=0-50), so we position event info below that starting at y=60.
    """
    h, w = frame.shape[:2]
    
    if not events:
        return frame
    
    # Calculate box height (single line per event: 22px each + padding)
    box_h = 15 + len(events) * 22
    
    # Draw event box in top-left, below existing text (starts at y=60)
    box_x = 10
    box_y = 60  # Below the Frame/time/persons text
    box_w = 300
    
    # Semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(overlay, (box_x, box_y), (box_x + box_w, box_y + box_h), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
    
    y_pos = box_y + 20
    
    # Draw each event (compact single-line format)
    for event in events:
        event_type = event.get("event_type", "unknown")
        persons = event.get("persons_involved", [])
        event_id = event.get("event_id")
        
        color = EVENT_COLORS.get(event_type, (255, 255, 255))
        label = EVENT_SHORT_LABELS.get(event_type, event_type.upper())
        
        # Build compact single-line text: ● JOINT_ATTN P0,P1
        persons_text = f"P{','.join(map(str, persons))}" if persons else ""
        line_text = f"{label} {persons_text}"
        
        # Determine text color based on Gemini validation status
        # Blue = confirmed, Red = rejected, White = no classification
        text_color = (255, 255, 255)  # Default white
        if classifications and event_id in classifications:
            cls = classifications[event_id]
            if cls.get("event_confirmed"):
                text_color = (255, 0, 0)    # Blue - confirmed
            else:
                text_color = (0, 0, 255)    # Red - rejected
        
        # Colored indicator circle (event type color)
        cv2.circle(frame, (box_x + 10, y_pos - 4), 6, color, -1)
        
        # Event text (colored by confirmation status)
        cv2.putText(
            frame, line_text,
            (box_x + 22, y_pos),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA
        )
        
        y_pos += 22
    
    return frame


def draw_gesture_overlay(
    frame: np.ndarray,
    events: List[Dict],
    classifications: Optional[Dict] = None,
) -> np.ndarray:
    """Draw gesture information overlay at bottom-left.
    
    Shows deictic_gestures and responder_ids for active events.
    Stacks upward when multiple gestures exist.
    """
    if not classifications:
        return frame
    
    h, w = frame.shape[:2]
    
    # Collect all gestures from active events
    all_gestures = []
    for event in events:
        event_id = event.get("event_id")
        if event_id in classifications:
            cls = classifications[event_id]
            deictic = cls.get("deictic_gestures", [])
            responder_ids = cls.get("responder_ids", [])
            
            for gesture in deictic:
                all_gestures.append({
                    "gesture_type": gesture.get("gesture_type", "unknown"),
                    "initiator_id": gesture.get("initiator_id"),
                    "target_type": gesture.get("target_type"),
                    "target_description": gesture.get("target_description"),
                    "responder_ids": responder_ids,
                })
    
    # If no gestures found but events are active, show null
    if not all_gestures and events:
        # Check if any active event has classification
        has_classification = any(
            event.get("event_id") in classifications for event in events
        )
        if has_classification:
            all_gestures = [None]  # Placeholder for "null" display
    
    if not all_gestures:
        return frame
    
    # Calculate box dimensions
    line_height = 22
    box_padding = 10
    box_w = 350
    num_lines = len(all_gestures)
    box_h = box_padding + num_lines * line_height
    
    # Position at bottom-left, stacking upward
    box_x = 10
    box_y = h - box_h - 10
    
    # Semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(overlay, (box_x, box_y), (box_x + box_w, box_y + box_h), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
    
    # Draw title
    # cv2.putText(frame, "GESTURES", (box_x + 5, box_y + 15),
    #             cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1, cv2.LINE_AA)
    
    # Draw each gesture (stacking upward from bottom)
    y_pos = box_y + box_padding + line_height - 5
    for gesture in all_gestures:
        if gesture is None:
            # Show "null" when no gestures
            cv2.putText(
                frame, "GESTURE: null",
                (box_x + 10, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1, cv2.LINE_AA
            )
        else:
            g_type = gesture["gesture_type"].upper()
            initiator = gesture.get("initiator_id")
            target_type = gesture.get("target_type", "")
            target_desc = gesture.get("target_description", "")[:20]  # Truncate
            responders = gesture.get("responder_ids", [])
            
            color = GESTURE_COLORS.get(gesture["gesture_type"], (255, 255, 255))
            
            # Build text: "POINTING P1 -> object (book) | resp: P2,P3"
            text_parts = [g_type]
            if initiator is not None:
                text_parts.append(f"P{initiator}")
            if target_type:
                if target_type == "person" and gesture.get("target_person_id") is not None:
                    text_parts.append(f"-> P{gesture['target_person_id']}")
                elif target_desc:
                    text_parts.append(f"-> {target_desc}")
                else:
                    text_parts.append(f"-> {target_type}")
            
            line_text = " ".join(text_parts)
            
            if responders:
                resp_text = f" | resp: P{','.join(map(str, responders))}"
                line_text += resp_text
            
            # Colored indicator
            cv2.circle(frame, (box_x + 10, y_pos - 4), 5, color, -1)
            
            # Text
            cv2.putText(
                frame, line_text,
                (box_x + 22, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA
            )
        
        y_pos += line_height
    
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
    sample_fps = None
    if gestures_json_path and os.path.exists(gestures_json_path):
        with open(gestures_json_path, 'r') as f:
            gestures_data = json.load(f)
        for cls in gestures_data.get("classifications", []):
            classifications[cls["event_id"]] = cls
        # Get sample_fps from gesture data for accurate timing
        sample_fps = gestures_data.get("sample_fps")
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
    
    # Use sample_fps from data if available, otherwise use video fps
    # The viz video is at sample_fps (typically 2.0fps)
    timing_fps = sample_fps or events_data.get("sample_fps") or video_fps
    print(f"Video: {total_frames} frames @ {video_fps:.1f}fps (timing: {timing_fps:.1f}fps), {width}x{height}")
    
    # Create output
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, video_fps, (width, height))
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Calculate time using sample_fps (the fps the video was sampled at)
        time_sec = frame_idx / timing_fps
        
        # Get active events at this time
        active_events = get_events_at_time(events, time_sec)
        
        # Draw overlay (with Gemini classifications if available)
        frame = draw_event_overlay(frame, active_events, time_sec, frame_idx, classifications)
        
        # Draw gesture overlay at bottom-left (only during event duration)
        frame = draw_gesture_overlay(frame, active_events, classifications)
        
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
    events_dir: str,
    output_dir: str,
    gestures_dir: Optional[str] = None,
) -> None:
    """
    Batch process all visualization videos in a directory.
    
    Args:
        video_dir: Directory with *_viz.mp4 files
        events_dir: Directory with *_events.json files
        output_dir: Output directory for visualization videos
    """
    # Find all viz videos
    viz_videos = glob.glob(os.path.join(video_dir, "*_viz.mp4"))
    viz_videos += glob.glob(os.path.join(video_dir, "*_sam3rf_viz.mp4"))
    viz_videos = list(set(viz_videos))  # Remove duplicates
    
    print(f"Found {len(viz_videos)} visualization videos")
    print(f"Events dir: {events_dir}")
    print(f"Output dir: {output_dir}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    for viz_path in viz_videos:
        # Find corresponding events JSON in events_dir
        base_name = os.path.basename(viz_path)
        # Try different naming conventions
        events_base = base_name.replace("_viz.mp4", "_events.json")
        events_base = events_base.replace("_sam3rf_viz.mp4", "_events.json")
        # Also try with _gaze suffix
        events_candidates = [
            os.path.join(events_dir, events_base),
            os.path.join(events_dir, events_base.replace("_events.json", "_gaze_events.json")),
            os.path.join(events_dir, events_base.replace("_events.json", "_sam3rf_gaze_events.json")),
        ]
        
        events_path = None
        for candidate in events_candidates:
            if os.path.exists(candidate):
                events_path = candidate
                break
        
        if not events_path:
            print(f"Skipping {base_name}: no events JSON found")
            continue
        
        # Find matching gestures JSON if gestures_dir provided
        gestures_path = None
        if gestures_dir:
            # Strip _sam3rf_viz.mp4 first, then _viz.mp4 to get base name
            gestures_base = base_name.replace("_sam3rf_viz.mp4", "_gestures.json")
            if gestures_base == base_name:  # Didn't match, try other pattern
                gestures_base = base_name.replace("_viz.mp4", "_gestures.json")
            gestures_candidate = os.path.join(gestures_dir, gestures_base)
            if os.path.exists(gestures_candidate):
                gestures_path = gestures_candidate
            else:
                # When gestures_dir is provided, skip videos without gesture data
                print(f"Skipping {base_name}: no gestures JSON found (gesture mode)")
                continue
        
        # Output name reflects whether gestures are included
        suffix = "_full_viz.mp4" if gestures_path else "_events_viz.mp4"
        output_name = base_name.replace("_sam3rf_viz.mp4", suffix)
        if output_name == base_name:  # Didn't match, try other pattern
            output_name = base_name.replace("_viz.mp4", suffix)
        output_path = os.path.join(output_dir, output_name)
        
        print(f"\nProcessing: {base_name}")
        visualize_events(viz_path, events_path, output_path, gestures_path)


def main():
    parser = argparse.ArgumentParser(description="Visualize detected events on videos")
    parser.add_argument("--viz_video", type=str, default=None, help="Single visualization video")
    parser.add_argument("--events_json", type=str, default=None, help="Events JSON for single video")
    parser.add_argument("--gestures_json", type=str, default=None, help="Gemini gestures JSON (optional)")
    parser.add_argument("--video_dir", type=str, default=None, help="Directory with *_viz.mp4 for batch")
    parser.add_argument("--events_dir", type=str, default=None, help="Directory with *_events.json for batch")
    parser.add_argument("--gestures_dir", type=str, default=None, help="Directory with *_gestures.json for batch (optional)")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory (required)")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
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
        
        # Generate output filename
        base_name = os.path.basename(args.viz_video)
        suffix = "_full_viz.mp4" if args.gestures_json else "_events_viz.mp4"
        out_name = base_name.replace("_viz.mp4", suffix).replace("_sam3rf_viz.mp4", suffix)
        output_path = os.path.join(args.output_dir, out_name)
        
        visualize_events(args.viz_video, args.events_json, output_path, args.gestures_json)
    
    elif args.video_dir:
        # Batch mode
        events_dir = args.events_dir or args.video_dir
        batch_visualize(args.video_dir, events_dir, args.output_dir, args.gestures_dir)
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
