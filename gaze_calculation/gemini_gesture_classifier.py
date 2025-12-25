#!/usr/bin/env python3
"""
Gemini Gesture Classifier for Social Gesture Detection

Uses Gemini Vision API to classify social gestures in candidate event clips.

For each candidate event:
1. Extract video clip around the event
2. Optionally overlay gaze visualization
3. Query Gemini with structured prompt
4. Parse and save gesture classification

Usage:
    python gemini_gesture_classifier.py \
        --video_path /path/to/video.mp4 \
        --events_json /path/to/events.json \
        --gaze_json /path/to/gaze.json \
        --output_json /path/to/gestures.json
"""

import argparse
import base64
import json
import os
import tempfile
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np

# Load .env file if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not installed, use env vars directly

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("Warning: google-generativeai not installed. Install: pip install google-generativeai")


# Deictic gesture taxonomy (these require visual analysis - can't be detected from gaze alone)
DEICTIC_GESTURES = {
    "pointing": "Directing attention to an object or location by pointing",
    "showing": "Presenting or displaying an object to another person",
    "giving": "Offering or handing over an object to another person",
    "reaching": "Extending arm to acquire or touch an object",
}

# Note: Social attention patterns (joint_attention, gaze_following, turn_taking, attention_capture)
# are detected by gaze_feature_extractor.py and candidate_event_detector.py
# Gemini's role is to VALIDATE those detections and identify accompanying deictic gestures


@dataclass
class GestureClassification:
    """Classification result for a candidate event."""
    event_id: int
    # Binary validation of detected gaze event
    event_confirmed: bool
    rejection_reason: Optional[str]  # Only if rejected
    # Deictic gestures (supports multiple concurrent gestures)
    deictic_gestures: List[Dict]  # List of {gesture_type, initiator_id, target_type, target_person_id, target_description}
    # Causality (for QA: C2 task)
    caused_gaze_shift: bool
    responder_ids: List[int]
    # Description
    description: str
    raw_response: str


def extract_video_clip(
    video_path: str,
    start_frame: int,
    end_frame: int,
    context_frames: int = 10,
    max_frames: int = 30,
    output_path: Optional[str] = None,
) -> str:
    """
    Extract video clip around an event.
    
    Args:
        video_path: Path to source video
        start_frame: Event start frame
        end_frame: Event end frame
        context_frames: Frames to include before/after event
        max_frames: Maximum frames to include
        output_path: Output path (temp file if None)
    
    Returns:
        Path to extracted clip
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate frame range
    clip_start = max(0, start_frame - context_frames)
    clip_end = min(total_frames - 1, end_frame + context_frames)
    
    # Limit to max_frames
    if clip_end - clip_start > max_frames:
        mid = (start_frame + end_frame) // 2
        clip_start = max(0, mid - max_frames // 2)
        clip_end = clip_start + max_frames
    
    if output_path is None:
        fd, output_path = tempfile.mkstemp(suffix=".mp4")
        os.close(fd)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, clip_start)
    
    for _ in range(clip_end - clip_start):
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
    
    cap.release()
    out.release()
    
    return output_path


def video_to_frames_base64(
    video_path: str,
    start_sec: float = 0,
    end_sec: float = None,
    sample_fps: float = 2.0,
    max_frames: int = 10,
    resize: Tuple[int, int] = (640, 360),
) -> List[str]:
    """
    Extract frames from video at target FPS and convert to base64 for API.
    
    Args:
        video_path: Path to video
        start_sec: Start time in seconds
        end_sec: End time in seconds (None = to end)
        sample_fps: Target sampling FPS (default: 2.0 to match gaze annotation)
        max_frames: Maximum frames to extract
        resize: Resize frames to this size
    
    Returns:
        List of base64-encoded JPEG frames
    """
    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / video_fps if video_fps > 0 else 0
    
    if end_sec is None:
        end_sec = duration
    
    # Calculate frame indices at target sample_fps
    frame_interval = video_fps / sample_fps  # e.g., 30fps/2fps = every 15th frame
    
    start_frame = int(start_sec * video_fps)
    end_frame = min(int(end_sec * video_fps), total_frames - 1)
    
    # Sample frames at sample_fps
    frame_indices = []
    current_frame = start_frame
    while current_frame <= end_frame and len(frame_indices) < max_frames:
        frame_indices.append(int(current_frame))
        current_frame += frame_interval
    
    frames_b64 = []
    
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        
        if resize:
            frame = cv2.resize(frame, resize)
        
        _, buffer = cv2.imencode('.jpg', frame)
        b64 = base64.b64encode(buffer).decode('utf-8')
        frames_b64.append(b64)
    
    cap.release()
    return frames_b64


def build_classification_prompt(event: Dict, gaze_context: Dict) -> str:
    """
    Build a structured prompt for Gemini to validate and classify gestures.
    """
    event_type = event.get('event_type', 'unknown')
    persons = event.get('persons_involved', [])
    
    # Event type definitions
    event_definitions = {
        "sudden_gaze_shift": "A person suddenly changed where they're looking",
        "joint_attention": "Multiple people looking at the same target",
        "gaze_following": "One person looks where another person is looking",
        "attention_capture": "One person's action caused others to look at them",
        "mutual_gaze": "Two people looking at each other",
    }
    event_desc = event_definitions.get(event_type, event_type)
    
    prompt = f"""Analyze these video frames to validate a detected social interaction event.

## About These Frames
- These frames have annotations: Colored bounding boxes (P0, P1, P2...) and gaze direction lines.
- **Note on Gaze Accuracy**: Gaze lines are approximate. Do not be overly strict about the exact endpoint of the line. Prioritize the logical context and the general trajectory of the gaze to determine the target.

## Event Detected
- Type: **{event_type}** = {event_desc}
- Time: {event.get('start_time', 0):.2f}s - {event.get('end_time', 0):.2f}s
- Persons: {persons}

## Tasks
1. **Validate**: Do gaze patterns support this {event_type}?
2. **Detect Gestures**: List ALL deictic gestures (pointing, showing, giving, reaching)
3. **Causality**: Did any gesture cause gaze shifts?

## Response Format (JSON only)

{{
    "event_confirmed": true | false,
    "rejection_reason": "<only if rejected>",
    
    "deictic_gestures": [
        {{
            "gesture_type": "pointing" | "showing" | "giving" | "reaching",
            "initiator_id": <person ID>,
            "target_type": "person" | "object" | "location",
            "target_person_id": <ID if person, else null>,
            "target_description": "<what/who>"
        }}
    ],
    
    "caused_gaze_shift": true | false,
    "responder_ids": [<person IDs who shifted gaze>],
    
    "description": "<what's happening>"
}}

Return empty array for deictic_gestures if none detected. Respond with ONLY a valid JSON object. No reasoning or introductory text.
"""
    return prompt


def classify_with_gemini(
    frames_b64: List[str],
    prompt: str,
    model_name: str = "gemini-2.0-flash",
) -> Tuple[str, Dict]:
    """
    Send frames and prompt to Gemini for classification.
    
    Returns:
        Tuple of (raw_response, parsed_json)
    """
    if not GEMINI_AVAILABLE:
        return "ERROR: Gemini not available", {}
    
    model = genai.GenerativeModel(model_name)
    
    # Build content with images
    content = []
    for b64 in frames_b64:
        content.append({
            "inline_data": {
                "mime_type": "image/jpeg",
                "data": b64
            }
        })
    content.append(prompt)
    
    try:
        response = model.generate_content(content)
        raw_text = response.text.strip()
        
        # Parse JSON from response
        # Handle case where response might have markdown code blocks
        if raw_text.startswith("```"):
            lines = raw_text.split("\n")
            json_lines = []
            in_json = False
            for line in lines:
                if line.startswith("```json"):
                    in_json = True
                    continue
                if line.startswith("```"):
                    in_json = False
                    continue
                if in_json:
                    json_lines.append(line)
            raw_text = "\n".join(json_lines)
        
        parsed = json.loads(raw_text)
        return response.text, parsed
        
    except Exception as e:
        return f"ERROR: {str(e)}", {}


def classify_events(
    video_path: str,
    events: List[Dict],
    gaze_data: Dict,
    api_key: Optional[str] = None,
    model_name: str = "gemini-2.0-flash",
    max_events: Optional[int] = None,
    sample_fps: float = 2.0,
    viz_video_path: Optional[str] = None,
) -> List[GestureClassification]:
    """
    Classify all candidate events using Gemini.
    
    Args:
        video_path: Path to original video (used if viz_video_path not provided)
        events: List of candidate events to classify
        gaze_data: Gaze annotation data
        api_key: Gemini API key
        model_name: Gemini model to use
        max_events: Maximum events to classify
        sample_fps: FPS for frame extraction (default: 2.0 to match gaze annotation)
        viz_video_path: Path to visualization video with annotations overlay.
                        If None, will auto-detect by looking for *_viz.mp4 or *_sam3rf_viz.mp4
    """
    if api_key:
        genai.configure(api_key=api_key)
    
    results = []
    events_to_process = events[:max_events] if max_events else events
    
    # Try to find visualization video if not provided
    if viz_video_path is None:
        base_name = os.path.splitext(video_path)[0]
        # Try common visualization video suffixes
        for suffix in ['_viz.mp4', '_sam3rf_viz.mp4']:
            candidate = base_name + suffix
            if os.path.exists(candidate):
                viz_video_path = candidate
                print(f"Using visualization video: {viz_video_path}")
                break
    
    # Use viz video if available, otherwise use raw video
    source_video = viz_video_path if viz_video_path and os.path.exists(viz_video_path) else video_path
    if source_video == video_path:
        print(f"Using raw video (no visualization found): {video_path}")
    
    # Get video FPS for time conversion
    cap = cv2.VideoCapture(source_video)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    
    for i, event in enumerate(events_to_process):
        print(f"Classifying event {i+1}/{len(events_to_process)}: {event.get('event_type')}")
        
        # Convert frame indices to time (use original frame indices from gaze annotation)
        start_time = event["start_time"] - 1.0  # 1 second context before
        end_time = event["end_time"] + 1.0  # 1 second context after
        
        # Extract frames from visualization video (has person IDs and gaze arrows drawn)
        frames_b64 = video_to_frames_base64(
            source_video,
            start_sec=max(0, start_time),
            end_sec=end_time,
            sample_fps=sample_fps,
            max_frames=10,  # ~5 seconds at 2fps
        )
        
        # Build prompt
        prompt = build_classification_prompt(event, gaze_data)
        
        # Query Gemini
        raw_response, parsed = classify_with_gemini(frames_b64, prompt, model_name)
        
        if parsed:
            result = GestureClassification(
                event_id=event.get("event_id", i),
                event_confirmed=parsed.get("event_confirmed", False),
                rejection_reason=parsed.get("rejection_reason"),
                deictic_gestures=parsed.get("deictic_gestures", []),
                caused_gaze_shift=parsed.get("caused_gaze_shift", False),
                responder_ids=parsed.get("responder_ids", []),
                description=parsed.get("description", ""),
                raw_response=raw_response,
            )
        else:
            result = GestureClassification(
                event_id=event.get("event_id", i),
                event_confirmed=False,
                rejection_reason="Failed to parse Gemini response",
                deictic_gestures=[],
                caused_gaze_shift=False,
                responder_ids=[],
                description="",
                raw_response=raw_response,
            )
        
        results.append(result)
    
    return results


def save_classifications(
    results: List[GestureClassification],
    events_data: Dict,
    output_path: str,
) -> None:
    """Save classification results to JSON."""
    # Count validation results and deictic gestures
    confirmed_count = sum(1 for r in results if r.event_confirmed)
    rejected_count = len(results) - confirmed_count
    deictic_counts = {}
    
    for r in results:
        for gesture in r.deictic_gestures:
            g_type = gesture.get("gesture_type")
            if g_type:
                deictic_counts[g_type] = deictic_counts.get(g_type, 0) + 1
    
    output = {
        "video_path": events_data.get("video_path"),
        "num_events_classified": len(results),
        "events_confirmed": confirmed_count,
        "events_rejected": rejected_count,
        "deictic_gestures_detected": deictic_counts,
        "classifications": [],
    }
    
    # Convert to dicts
    for r in results:
        output["classifications"].append({
            "event_id": r.event_id,
            "event_confirmed": r.event_confirmed,
            "rejection_reason": r.rejection_reason,
            "deictic_gestures": r.deictic_gestures,
            "caused_gaze_shift": r.caused_gaze_shift,
            "responder_ids": r.responder_ids,
            "description": r.description,
        })
    
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"Saved {len(results)} classifications to: {output_path}")


def get_output_filename(events_json_path: str) -> str:
    """Generate output filename from events JSON path.
    
    Pattern: {video_id}_sam3rf_gaze_events.json -> {video_id}_gestures.json
    """
    basename = os.path.basename(events_json_path)
    # Remove _sam3rf_gaze_events.json suffix
    video_id = basename.replace("_sam3rf_gaze_events.json", "").replace("_gaze_events.json", "").replace("_events.json", "")
    return f"{video_id}_gestures.json"


def process_single_video(
    video_path: str,
    events_json: str,
    output_path: str,
    api_key: str,
    model_name: str = "gemini-2.0-flash",
    gaze_json: Optional[str] = None,
    max_events: Optional[int] = None,
) -> bool:
    """Process a single video and save classifications.
    
    Returns True if successful, False otherwise.
    """
    print(f"\n{'='*60}")
    print(f"Processing: {os.path.basename(events_json)}")
    print(f"{'='*60}")
    
    # Load events
    print(f"Loading events from: {events_json}")
    with open(events_json, 'r') as f:
        events_data = json.load(f)
    
    # Load gaze data if provided
    gaze_data = {}
    if gaze_json and os.path.exists(gaze_json):
        print(f"Loading gaze data from: {gaze_json}")
        with open(gaze_json, 'r') as f:
            gaze_data = json.load(f)
    
    events = events_data.get("events", [])
    print(f"Found {len(events)} candidate events")
    
    if not events:
        print("No events to process, skipping...")
        return True
    
    # Classify events
    print("Classifying with Gemini...")
    results = classify_events(
        video_path,
        events,
        gaze_data,
        api_key=api_key,
        model_name=model_name,
        max_events=max_events,
    )
    
    # Print summary
    print(f"\nClassification Summary:")
    confirmed_count = sum(1 for r in results if r.event_confirmed)
    rejected_count = len(results) - confirmed_count
    deictic_counts = {}
    
    for r in results:
        for gesture in r.deictic_gestures:
            g_type = gesture.get("gesture_type")
            if g_type:
                deictic_counts[g_type] = deictic_counts.get(g_type, 0) + 1
    
    print(f"  Events confirmed: {confirmed_count}")
    print(f"  Events rejected: {rejected_count}")
    
    if deictic_counts:
        print("  Deictic Gestures Detected:")
        for gesture, count in deictic_counts.items():
            print(f"    - {gesture}: {count}")
    
    # Save results
    save_classifications(results, events_data, output_path)
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Classify social gestures using Gemini",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single file mode
  python gemini_gesture_classifier.py \\
      --video_path /path/to/video.mp4 \\
      --events_json /path/to/events.json

  # Single file with output directory
  python gemini_gesture_classifier.py \\
      --video_path /path/to/video.mp4 \\
      --events_json /path/to/events.json \\
      --output_dir /path/to/output

  # Batch mode
  python gemini_gesture_classifier.py \\
      --batch \\
      --events_dir /path/to/event_data \\
      --videos_dir /path/to/gaze_videos \\
      --output_dir /path/to/gesture_output
        """
    )
    
    # Mode selection
    parser.add_argument("--batch", action="store_true", help="Enable batch processing mode")
    
    # Single file mode args
    parser.add_argument("--video_path", type=str, help="Path to video (single mode)")
    parser.add_argument("--events_json", type=str, help="Path to candidate events JSON (single mode)")
    parser.add_argument("--gaze_json", type=str, default=None, help="Path to gaze annotation JSON (optional)")
    parser.add_argument("--output_json", type=str, default=None, help="Output path (single mode, overrides output_dir)")
    
    # Batch mode args
    parser.add_argument("--events_dir", type=str, help="Directory containing event JSON files (batch mode)")
    parser.add_argument("--videos_dir", type=str, help="Directory containing visualization videos (batch mode)")
    
    # Common args
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for gesture JSONs")
    parser.add_argument("--api_key", type=str, default=None, help="Gemini API key (or set GOOGLE_API_KEY env)")
    parser.add_argument("--model", type=str, default="gemini-2.0-flash", help="Gemini model to use")
    parser.add_argument("--max_events", type=int, default=None, help="Max events to classify per video")
    parser.add_argument("--skip_existing", action="store_true", help="Skip if output file already exists")
    
    args = parser.parse_args()
    
    # Get API key
    api_key = args.api_key or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("ERROR: No API key provided. Set GOOGLE_API_KEY env or use --api_key")
        return
    
    if args.batch:
        # ========== BATCH MODE ==========
        if not args.events_dir or not args.videos_dir:
            print("ERROR: Batch mode requires --events_dir and --videos_dir")
            return
        
        if not args.output_dir:
            print("ERROR: Batch mode requires --output_dir")
            return
        
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Find all event files
        event_files = sorted([
            f for f in os.listdir(args.events_dir) 
            if f.endswith("_events.json")
        ])
        
        print(f"Found {len(event_files)} event files in {args.events_dir}")
        
        processed = 0
        skipped = 0
        failed = 0
        
        for event_file in event_files:
            events_json = os.path.join(args.events_dir, event_file)
            output_filename = get_output_filename(events_json)
            output_path = os.path.join(args.output_dir, output_filename)
            
            # Skip if exists
            if args.skip_existing and os.path.exists(output_path):
                print(f"Skipping (exists): {output_filename}")
                skipped += 1
                continue
            
            # Find matching video
            # Pattern: {video_id}_sam3rf_gaze_events.json -> {video_id}_sam3rf_viz.mp4
            video_id = event_file.replace("_sam3rf_gaze_events.json", "").replace("_gaze_events.json", "").replace("_events.json", "")
            
            # Try different video naming patterns
            video_candidates = [
                os.path.join(args.videos_dir, f"{video_id}_sam3rf_viz.mp4"),
                os.path.join(args.videos_dir, f"{video_id}_viz.mp4"),
                os.path.join(args.videos_dir, f"{video_id}.mp4"),
            ]
            
            video_path = None
            for candidate in video_candidates:
                if os.path.exists(candidate):
                    video_path = candidate
                    break
            
            if not video_path:
                print(f"WARNING: No video found for {event_file}, skipping...")
                failed += 1
                continue
            
            # Find matching gaze JSON (optional)
            gaze_json = None
            gaze_candidates = [
                os.path.join(args.videos_dir, f"{video_id}_sam3rf_gaze.json"),
                os.path.join(args.videos_dir, f"{video_id}_gaze.json"),
            ]
            for candidate in gaze_candidates:
                if os.path.exists(candidate):
                    gaze_json = candidate
                    break
            
            try:
                success = process_single_video(
                    video_path=video_path,
                    events_json=events_json,
                    output_path=output_path,
                    api_key=api_key,
                    model_name=args.model,
                    gaze_json=gaze_json,
                    max_events=args.max_events,
                )
                if success:
                    processed += 1
                else:
                    failed += 1
            except Exception as e:
                print(f"ERROR processing {event_file}: {e}")
                failed += 1
        
        print(f"\n{'='*60}")
        print(f"Batch Processing Complete")
        print(f"{'='*60}")
        print(f"  Processed: {processed}")
        print(f"  Skipped:   {skipped}")
        print(f"  Failed:    {failed}")
        print(f"  Total:     {len(event_files)}")
    
    else:
        # ========== SINGLE FILE MODE ==========
        if not args.video_path or not args.events_json:
            print("ERROR: Single mode requires --video_path and --events_json")
            return
        
        # Determine output path
        if args.output_json:
            output_path = args.output_json
        elif args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)
            output_filename = get_output_filename(args.events_json)
            output_path = os.path.join(args.output_dir, output_filename)
        else:
            # Default: same directory as events_json
            base = os.path.splitext(args.events_json)[0].replace("_events", "")
            output_path = f"{base}_gestures.json"
        
        # Skip if exists
        if args.skip_existing and os.path.exists(output_path):
            print(f"Skipping (exists): {output_path}")
            return
        
        process_single_video(
            video_path=args.video_path,
            events_json=args.events_json,
            output_path=output_path,
            api_key=api_key,
            model_name=args.model,
            gaze_json=args.gaze_json,
            max_events=args.max_events,
        )


if __name__ == "__main__":
    main()
