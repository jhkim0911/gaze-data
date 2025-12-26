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
import sys
import tempfile

# Force unbuffered output for SLURM/logging
sys.stdout.reconfigure(line_buffering=True)

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np

# Load .env file from project root
ENV_FILE_PATH = "/u/arkimjh/code/ECCV-jh/.env"

try:
    from dotenv import load_dotenv
    import os as _os
    if _os.path.exists(ENV_FILE_PATH):
        load_dotenv(ENV_FILE_PATH)
    else:
        load_dotenv()  # Fallback to default .env search
except ImportError:
    pass  # dotenv not installed, use env vars directly

try:
    from google import genai
    from google.genai import types
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("Warning: google-genai not installed. Install: pip install google-genai")



# Deictic gesture taxonomy (these require visual analysis - can't be detected from gaze alone)
DEICTIC_GESTURES = {
    "pointing": "Directing attention to an object or location by pointing",
    "showing": "Presenting or displaying an object to another person",
    "giving": "Offering or handing over an object to another person",
    "reaching": "Extending arm to acquire or touch an object",
}


@dataclass
class GestureClassification:
    """Classification result for a candidate event."""
    event_id: int
    event_confirmed: bool
    rejection_reason: Optional[str]
    deictic_gestures: List[Dict]
    caused_gaze_shift: bool
    responder_ids: List[int]
    description: str
    raw_response: str


# Define the Schema for Gemini JSON Mode
# Using a dictionary for compatibility with the new SDK's Schema type
GESTURE_RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "event_confirmed": {"type": "BOOLEAN"},
        "rejection_reason": {"type": "STRING"},
        "deictic_gestures": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "gesture_type": {"type": "STRING", "enum": ["pointing", "showing", "giving", "reaching"]},
                    "initiator_id": {"type": "INTEGER"},
                    "target_type": {"type": "STRING", "enum": ["person", "object", "location"]},
                    "target_person_id": {"type": "INTEGER"},
                    "target_description": {"type": "STRING"}
                },
                "required": ["gesture_type", "initiator_id", "target_type", "target_description"]
            }
        },
        "caused_gaze_shift": {"type": "BOOLEAN"},
        "responder_ids": {"type": "ARRAY", "items": {"type": "INTEGER"}},
        "description": {"type": "STRING"}
    },
    "required": ["event_confirmed", "deictic_gestures", "caused_gaze_shift", "responder_ids", "description"]
}


def upload_video_to_gemini(client: 'genai.Client', video_path: str) -> Optional[any]:
    """
    Upload a video file to Gemini for processing using genai.Client.
    """
    if not GEMINI_AVAILABLE:
        return None
    
    try:
        print(f"Uploading video to Gemini: {video_path}...")
        # Upload the video file (new SDK syntax)
        # client.files.upload returns a wrapper/object with .name, .state, etc.
        video_file = client.files.upload(file=video_path, config={"mime_type": "video/mp4"})
        
        # Wait for file to be processed
        import time
        max_wait = 300  # 5 minutes max wait for full video
        wait_time = 0
        print(f"Waiting for video processing (current state: {video_file.state})...")
        
        # Note: In new SDK, state is an enum or string. Defensive check.
        # Usually it's "PROCESSING", "ACTIVE", "FAILED"
        print(f"Waiting for video processing (current state: {video_file.state})...")
        
        while video_file.state == types.FileState.PROCESSING and wait_time < max_wait:
            time.sleep(5)
            wait_time += 5
            # Refresh file status
            video_file = client.files.get(name=video_file.name)
            if wait_time % 10 == 0:
                print(f"  Still processing... ({wait_time}s)")
        
        if video_file.state == types.FileState.FAILED:
            print(f"Video processing failed: {video_file.name}")
            return None
            
        print(f"Video ready: {video_file.name}")
        return video_file
    except Exception as e:
        print(f"Error uploading video: {e}")
        return None


def delete_uploaded_file(client: 'genai.Client', file_obj) -> None:
    """Delete an uploaded file from Gemini to clean up."""
    if file_obj is not None:
        try:
            client.files.delete(name=file_obj.name)
            print(f"Deleted remote file: {file_obj.name}")
        except Exception:
            pass


def build_classification_prompt(event: Dict, gaze_context: Dict) -> str:
    """
    Build a structured prompt for Gemini using timestamps.
    """
    event_type = event.get('event_type', 'unknown')
    persons = event.get('persons_involved', [])
    start_time = event.get('start_time', 0)
    end_time = event.get('end_time', 0)
    
    # Add context window (1s before/after)
    focus_start = max(0, start_time - 1.0)
    focus_end = end_time + 1.0
    
    event_definitions = {
        "sudden_gaze_shift": "A person suddenly changed where they're looking",
        "joint_attention": "Multiple people looking at the same target",
        "gaze_following": "One person looks where another person is looking",
        "attention_capture": "One person's action caused others to look at them",
        "mutual_gaze": "Two people looking at each other",
    }
    event_desc = event_definitions.get(event_type, event_type)
    
    prompt = f"""Analyze the uploaded video for the following event.
    
    ## Focus Segment
    **TIMESTAMPS: {focus_start:.2f}s to {focus_end:.2f}s**
    Strictly focus your analysis on this time window.
    
    ## Event Details
    - Type: **{event_type}** ({event_desc})
    - Persons Involved IDs: {persons}
    - Visuals: The video has bounding boxes and gaze lines. Use them to infer targets.
    
    ## Tasks
    1. **Validate**: Is the `{event_type}` event genuinely happening during this segment?
    2. **Detect Gestures**: Identify any DEICTIC gestures (pointing, showing, giving, reaching) performed by the people involved.
    3. **Causality**: Did a gesture CAUSE the gaze event?
    """
    return prompt


def classify_with_gemini(
    client: 'genai.Client',
    prompt: str,
    video_file: any,
    model_name: str = "gemini-2.0-flash-exp", 
) -> Tuple[str, Dict]:
    """
    Send prompt with PRE-UPLOADED video to Gemini using genai.Client.
    Uses Native JSON Mode for output.
    """
    if not GEMINI_AVAILABLE:
        return "ERROR: Gemini not available", {}
    
    # Configure generation config using types (new SDK)
    config = types.GenerateContentConfig(
        temperature=0.2,
        response_mime_type="application/json",
        response_schema=GESTURE_RESPONSE_SCHEMA
    )
    
    try:
        # Pass the file object directly in contents list (new SDK handles it)
        response = client.models.generate_content(
            model=model_name,
            contents=[video_file, prompt],
            config=config
        )
        
        # New SDK response.text accesses the generated text
        raw_text = response.text
        try:
             # Try parsing manually first to be safe
             parsed = json.loads(raw_text)
        except:
             # If manual parsing fails, check if the SDK did something weird
             return raw_text, {}

        return raw_text, parsed
        
    except Exception as e:
        print(f"Gemini API Error: {e}")
        return f"ERROR: {str(e)}", {}


def classify_events(
    video_path: str,
    events: List[Dict],
    gaze_data: Dict,
    api_key: Optional[str] = None,
    model_name: str = "gemini-2.0-flash-exp",
    max_events: Optional[int] = None,
    sample_fps: float = None, 
    viz_video_path: Optional[str] = None,
) -> List[GestureClassification]:
    """
    Classify all candidate events using Single Video Upload + Long Context.
    Uses genai.Client.
    """
    if not GEMINI_AVAILABLE:
        print("ERROR: google-genai package not installed. Skipping classification.")
        return []

    # Initialize Client (New SDK)
    if not api_key:
         print("ERROR: No API key provided")
         return []
         
    client = genai.Client(api_key=api_key)
    
    results = []
    events_to_process = events[:max_events] if max_events else events
    
    # 1. Determine which video to use
    if viz_video_path is None:
        base_name = os.path.splitext(video_path)[0]
        # Check standard visualization paths
        for suffix in ['_viz.mp4', '_sam3rf_viz.mp4']:
            candidate = base_name + suffix
            if os.path.exists(candidate):
                viz_video_path = candidate
                print(f"Using visualization video: {viz_video_path}")
                break
    
    # Validation
    source_video = viz_video_path if viz_video_path and os.path.exists(viz_video_path) else video_path
    if source_video == video_path:
        print(f"Using raw video (no visualization found): {video_path}")
    
    # 2. Upload Video ONCE
    print("Step 1: Uploading full video to Gemini...")
    video_file = upload_video_to_gemini(client, source_video)
    if not video_file:
        print("CRITICAL: Failed to upload video. Aborting.")
        return []
        
    try:
        # 3. Iterate Events
        print(f"Step 2: Processing {len(events_to_process)} events using Long Context...")
        
        for i, event in enumerate(events_to_process):
            # Build prompt with timestamps
            prompt = build_classification_prompt(event, gaze_data)
            
            # Query Gemini
            print(f"  [{i+1}/{len(events_to_process)}] {event.get('event_type')} @ {event.get('start_time'):.1f}s")
            
            raw_response, parsed = classify_with_gemini(
                client=client,
                prompt=prompt,
                video_file=video_file,
                model_name=model_name
            )
            
            if parsed:
                # Create result object with validated data
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
                # Handle failure
                result = GestureClassification(
                    event_id=event.get("event_id", i),
                    event_confirmed=False,
                    rejection_reason="Failed to parse Gemini response or API error",
                    deictic_gestures=[],
                    caused_gaze_shift=False,
                    responder_ids=[],
                    description="",
                    raw_response=raw_response,
                )
            
            results.append(result)
            
            # Rate limit politeness
            # gemini-2.0-flash-exp has 10 RPM limit. Sleep 10s to be safe (6 RPM).
            import time
            time.sleep(10)
            
    finally:
        # 4. Cleanup
        print("Cleaning up remote video file...")
        delete_uploaded_file(client, video_file)
    
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
    model_name: str = "gemini-2.5-pro",
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
    parser.add_argument("--model", type=str, default="gemini-2.0-flash-exp", help="Gemini model to use")
    parser.add_argument("--max_events", type=int, default=500, help="Max events to classify per video (default: 500)")
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
