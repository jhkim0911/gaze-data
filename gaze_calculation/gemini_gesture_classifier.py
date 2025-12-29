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
import time
import random
import functools
from concurrent.futures import ThreadPoolExecutor

# Force unbuffered output for SLURM/logging
sys.stdout.reconfigure(line_buffering=True)

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kwargs): return x
import cv2
import numpy as np

def retry_with_backoff(retries=5, initial_delay=5, backoff_factor=2):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay
            for i in range(retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    msg = str(e)
                    if "429" in msg or "RESOURCE_EXHAUSTED" in msg or "500" in msg:
                        if i == retries - 1: raise
                        sleep = delay + random.uniform(0, 1)
                        print(f"API Error ({msg}). Retrying in {sleep:.1f}s...")
                        time.sleep(sleep)
                        delay *= backoff_factor
                    else:
                        raise
            return func(*args, **kwargs)
        return wrapper
    return decorator

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
# Using types.Schema for better SDK enforcement
GESTURE_RESPONSE_SCHEMA = types.Schema(
    type="OBJECT",
    properties={
        "event_confirmed": types.Schema(type="BOOLEAN"),
        "rejection_reason": types.Schema(type="STRING", nullable=True),
        "deictic_gestures": types.Schema(
            type="ARRAY",
            items=types.Schema(
                type="OBJECT",
                properties={
                    "gesture_type": types.Schema(type="STRING", enum=["pointing", "showing", "giving", "reaching"]),
                    "initiator_id": types.Schema(type="INTEGER"),
                    "target_type": types.Schema(type="STRING", enum=["person", "object", "location"]),
                    "target_person_id": types.Schema(type="INTEGER", nullable=True),
                    "target_description": types.Schema(type="STRING")
                },
                required=["gesture_type", "initiator_id", "target_type", "target_description"]
            )
        ),
        "caused_gaze_shift": types.Schema(type="BOOLEAN"),
        "responder_ids": types.Schema(type="ARRAY", items=types.Schema(type="INTEGER")),
        "description": types.Schema(type="STRING")
    },
    required=["event_confirmed", "deictic_gestures", "caused_gaze_shift", "responder_ids", "description"]
)

@retry_with_backoff()
def upload_video_to_gemini(client: 'genai.Client', video_path: str) -> Optional[any]:
    """Refactored upload using new genai.Client."""
    try:
        if not os.path.exists(video_path):
            print(f"Video not found: {video_path}")
            return None
            
        print(f"Uploading video: {os.path.basename(video_path)}...")
        
        # Note: media_resolution is not supported in upload config. Using default.
        video_file = client.files.upload(file=video_path, config={"mime_type": "video/mp4"})
        
        print(f"Waiting for processing (URI: {video_file.uri})...")
        
        wait_time = 0
        max_wait = 300
        while video_file.state == types.FileState.PROCESSING and wait_time < max_wait:
            time.sleep(2)
            wait_time += 5
            try:
                video_file = client.files.get(name=video_file.name)
            except Exception as e:
                 if "429" in str(e): raise
                 pass # Transient get error, continue waiting
            
            if wait_time % 10 == 0:
                print(f"  Still processing... ({wait_time}s)")
        
        if video_file.state != types.FileState.ACTIVE:
            print(f"CRITICAL: Video state is {video_file.state.name}, expected ACTIVE.")
            return None
            
        print(f"Video ready: {video_file.name}")
        return video_file
        
    except Exception as e:
        if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e): raise
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


def build_validation_prompt(event: Dict, persons: str) -> str:
    """Stage 1: Lightweight prompt for validation only."""
    event_type = event.get('event_type', 'unknown')
    event_definitions = {
        "sudden_gaze_shift": "A person suddenly changed where they're looking",
        "joint_attention": "Multiple people looking at the same target",
        "gaze_following": "One person looks where another person is looking",
        "attention_capture": "One person's action caused others to look at them",
        "mutual_gaze": "Two people looking at each other",
    }
    event_desc = event_definitions.get(event_type, event_type)
    
    prompt = f"""Validate this detected social interaction event.

## Event
- Type: **{event_type}** = {event_desc}
- Time: {event.get('start_time', 0):.2f}s - {event.get('end_time', 0):.2f}s
- Persons: {persons}

## Task
Does the video evidence support this {event_type} event? Answer only with JSON.

{{{{
    "event_confirmed": true | false
}}}}
"""
    return prompt


def build_classification_prompt(event: Dict, persons: str) -> str:
    """
    Build a structured prompt for Gemini to validate event and detect gestures.
    """
    event_type = event.get('event_type', 'unknown')
    
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
- **Note on Gaze Accuracy**: Gaze lines are approximate. Prioritize logical context over exact line endpoints.

## Event Detected
- Type: **{event_type}** = {event_desc}
- Time: {event.get('start_time', 0):.2f}s - {event.get('end_time', 0):.2f}s
- Persons: {persons}

## Task
Detect deictic gestures and analyze gaze causality in this confirmed {event_type} event.

## Deictic Gesture Definitions (Be STRICT)

**pointing** - Directing others' attention to a target via extended finger/hand
  ✓ YES: Extended arm with finger pointing at object/person/location
  ✗ NOT: Talking with hands, hand gestures during speech, arm resting

**showing** - Presenting an object for others to visually inspect
  ✓ YES: Holding object UP and orienting it toward another person's view
  ✗ NOT: Simply holding an object, object on table, object in lap

**giving** - Transferring an object to another person's possession  
  ✓ YES: Extending hand WITH object toward recipient, offering motion
  ✗ NOT: Object at rest, just holding something, passing nearby

**reaching** - Extending hand toward an object to acquire/touch it
  ✓ YES: Arm extending TOWARD object with intent to grab/touch
  ✗ NOT: Arms relaxed, hands in lap, casual arm movement

## Key Distinctions
- pointing vs reaching: ATTENTION direction vs ACQUISITION intent
- showing vs giving: VISUAL inspection vs POSSESSION transfer
- giving + reaching often occur TOGETHER (one gives, other reaches)

## ANTI-HALLUCINATION RULES
1. Default to EMPTY array if uncertain
2. Require CLEAR arm/hand extension for any gesture
3. Hand movements during speech are NOT pointing
4. Object must be ACTIVELY manipulated for showing/giving
5. Report ONLY gestures you are >90% confident about

## Response Format (JSON only)

{{{{
    "deictic_gestures": [
        {{{{
            "gesture_type": "pointing" | "showing" | "giving" | "reaching",
            "initiator_id": <person ID>,
            "target_type": "person" | "object" | "location",
            "target_person_id": <ID if person, else null>,
            "target_description": "<what/who>"
        }}}}
    ],
    
    "caused_gaze_shift": true | false,
    "responder_ids": [<person IDs who shifted gaze>],
    
    "description": "<what's happening>"
}}}}

Return EMPTY array for deictic_gestures if none detected or uncertain. JSON only, no other text.
"""
    return prompt


@retry_with_backoff()
def classify_with_gemini(
    client: 'genai.Client',
    prompt: str,
    video_file: any,
    model_name: str = "models/gemini-2.5-flash", 
) -> Tuple[str, Dict]:
    """
    Send prompt with PRE-UPLOADED video to Gemini using genai.Client.
    Uses Native JSON Mode for output.
    """
    if not GEMINI_AVAILABLE:
        return "ERROR: Gemini not available", {}
    
    # Configure generation config using types (new SDK)
    config_dict = {
        "temperature": 0.0,  # Zero for maximum determinism, reduce hallucinations
        "response_mime_type": "application/json",
        "response_schema": GESTURE_RESPONSE_SCHEMA,
    }
    
    # Add Thinking Config for Gemini 3
    if "gemini-3" in model_name:
         config_dict["thinking_config"] = {"thinking_level": "medium"}

    # Pass video file object directly (SDK handles conversion)
    contents = [video_file, prompt]
        
    config = types.GenerateContentConfig(**config_dict)
    
    try:
        # Pass the file object directly in contents list (new SDK handles it)
        response = client.models.generate_content(
            model=model_name,
            contents=contents,
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
        if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e): raise
        print(f"Gemini API Error: {e}")
        return f"ERROR: {str(e)}", {}


def classify_events(
    video_path: str,
    events: List[Dict],
    gaze_data: Dict,
    api_key: Optional[str] = None,
    model_name: str = "models/gemini-2.5-flash",
    max_events: Optional[int] = None,
    sample_fps: float = None, 
    viz_video_path: Optional[str] = None,
) -> List[GestureClassification]:
    """
    Classify all candidate events using Single Video Upload + Long Context.
    """
    if not GEMINI_AVAILABLE:
        print("ERROR: google-genai package not installed. Skipping classification.")
        return []

    if not api_key:
         print("ERROR: No API key provided")
         return []
         
    client = genai.Client(api_key=api_key)
    
    results = []
    
    # Filter and limit events
    MIN_CONFIDENCE = 0.75
    MAX_EVENTS_TO_PROCESS = 15
    
    # First apply max_events limit if set
    candidate_events = events[:max_events] if max_events else events
    
    # Filter by confidence threshold
    filtered_events = [e for e in candidate_events if e.get("confidence", 1.0) >= MIN_CONFIDENCE]
    skipped_low_conf = len(candidate_events) - len(filtered_events)
    if skipped_low_conf > 0:
        print(f"Filtered out {skipped_low_conf} events with confidence < {MIN_CONFIDENCE}")
    
    # Sort by confidence descending and take top N
    sorted_events = sorted(filtered_events, key=lambda e: e.get("confidence", 0), reverse=True)
    if len(sorted_events) > MAX_EVENTS_TO_PROCESS:
        print(f"Clamping from {len(sorted_events)} to top {MAX_EVENTS_TO_PROCESS} events by confidence")
        events_to_process = sorted_events[:MAX_EVENTS_TO_PROCESS]
    else:
        events_to_process = sorted_events
    
    print(f"Processing {len(events_to_process)} events (from {len(events)} total)")
    
    # 1. Determine which video to use
    if viz_video_path is None:
        base_name = os.path.splitext(video_path)[0]
        for suffix in ['_viz.mp4', '_sam3rf_viz.mp4']:
            candidate = base_name + suffix
            if os.path.exists(candidate):
                viz_video_path = candidate
                print(f"Using visualization video: {viz_video_path}")
                break
    
    source_video = viz_video_path if viz_video_path and os.path.exists(viz_video_path) else video_path
    if source_video == video_path:
        print(f"Using raw video (no visualization found): {video_path}")
    
    # 2. Upload Video ONCE
    print("Step 1: Uploading full video to Gemini...")
    video_file = upload_video_to_gemini(client, source_video)
    if not video_file:
        print("CRITICAL: Failed to upload video. Aborting.")
        return []
    
    import time
    
    try:
        # =================== STAGE 1: VALIDATION ===================
        print(f"Stage 1: Validating {len(events_to_process)} events...")
        
        validation_results = {}
        
        for i, event in enumerate(events_to_process):
            event_id = event.get("event_id", i)
            persons = ", ".join([f"P{p}" for p in event.get("persons_involved", [])])
            prompt = build_validation_prompt(event, persons)
            
            print(f"  [{i+1}/{len(events_to_process)}] {event.get('event_type')} @ {event.get('start_time'):.1f}s")
            
            raw_response, parsed = classify_with_gemini(
                client=client,
                prompt=prompt,
                video_file=video_file,
                model_name=model_name
            )
            
            confirmed = parsed.get("event_confirmed", False) if parsed else False
            validation_results[event_id] = {"confirmed": confirmed, "raw_response": raw_response}
            time.sleep(3)
        
        confirmed_count = sum(1 for v in validation_results.values() if v["confirmed"])
        print(f"\nStage 1 Complete: {confirmed_count}/{len(events_to_process)} confirmed")
        
        # =================== STAGE 2: FULL CLASSIFICATION ===================
        classification_results = {}
        
        if confirmed_count > 0:
            print(f"\nStage 2: Classifying {confirmed_count} confirmed events...")
            
            confirmed_events = [e for e in events_to_process 
                               if validation_results.get(e.get("event_id", -1), {}).get("confirmed")]
            
            for i, event in enumerate(confirmed_events):
                event_id = event.get("event_id", i)
                persons = ", ".join([f"P{p}" for p in event.get("persons_involved", [])])
                prompt = build_classification_prompt(event, persons)
                
                print(f"  [{i+1}/{len(confirmed_events)}] {event.get('event_type')} @ {event.get('start_time'):.1f}s")
                
                raw_response, parsed = classify_with_gemini(
                    client=client,
                    prompt=prompt,
                    video_file=video_file,
                    model_name=model_name
                )
                
                if parsed:
                    classification_results[event_id] = parsed
                time.sleep(5)
        
        # =================== MERGE RESULTS ===================
        for i, event in enumerate(events_to_process):
            event_id = event.get("event_id", i)
            val = validation_results.get(event_id, {})
            cls = classification_results.get(event_id, {})
            
            result = GestureClassification(
                event_id=event_id,
                event_confirmed=val.get("confirmed", False),
                rejection_reason=None,
                deictic_gestures=cls.get("deictic_gestures", []),
                caused_gaze_shift=cls.get("caused_gaze_shift", False),
                responder_ids=cls.get("responder_ids", []),
                description=cls.get("description", ""),
                raw_response=val.get("raw_response", ""),
            )
            results.append(result)
            
    finally:
        print("Cleaning up resources...")
        delete_uploaded_file(client, video_file)
    
    return results


def save_classifications(
    results: List[GestureClassification],
    events_data: Dict,
    output_path: str,
) -> None:
    """Save classification results to JSON with full temporal info.
    
    Merges event metadata (timestamps, frames, persons) into each classification
    so the output is self-contained for Video-LLM training.
    """
    # Build event lookup for merging temporal info
    events_list = events_data.get("events", [])
    event_lookup = {e.get("event_id"): e for e in events_list}
    
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
        "video_fps": events_data.get("video_fps"),
        "sample_fps": events_data.get("sample_fps"),
        "num_events_classified": len(results),
        "events_confirmed": confirmed_count,
        "events_rejected": rejected_count,
        "deictic_gestures_detected": deictic_counts,
        "classifications": [],
    }
    
    # Convert to dicts with merged event metadata
    for r in results:
        # Get original event for temporal info
        event = event_lookup.get(r.event_id, {})
        
        classification_entry = {
            # Temporal info from events.json (for Video-LLM training)
            "event_id": r.event_id,
            "event_type": event.get("event_type"),
            "start_time": event.get("start_time"),
            "end_time": event.get("end_time"),
            "start_frame": event.get("start_frame"),
            "end_frame": event.get("end_frame"),
            "persons_involved": event.get("persons_involved", []),
            "event_confidence": event.get("confidence"),
            "event_details": event.get("details", {}),
            
            # Classification results from Gemini
            "event_confirmed": r.event_confirmed,
            "rejection_reason": r.rejection_reason,
            "deictic_gestures": r.deictic_gestures,
            "caused_gaze_shift": r.caused_gaze_shift,
            "responder_ids": r.responder_ids,
            "description": r.description,
        }
        output["classifications"].append(classification_entry)
    
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
    model_name: str = "models/gemini-2.5-flash",
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
    parser.add_argument("--workers", type=int, default=3, help="Number of concurrent workers (default: 3)")
    
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
    parser.add_argument("--model", type=str, default="models/gemini-2.5-flash", help="Gemini model to use")
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
        
        # Prepare Batch Tasks
        batch_tasks = []
        skipped = 0
        
        print(f"Preparing tasks for {args.workers} workers...")
        
        for event_file in event_files:
            events_json = os.path.join(args.events_dir, event_file)
            video_id = os.path.basename(event_file).replace("_gaze_events.json", "").replace("_events.json", "")
             
            # 1. Output Path
            output_filename = get_output_filename(event_file)
            output_path = os.path.join(args.output_dir, output_filename)
             
            # 2. Check Exists
            if args.skip_existing and os.path.exists(output_path):
                skipped += 1
                continue

            # 3. Find Video
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
                print(f"Warning: No video for {video_id}")
                continue
                
            # 4. Find Gaze (Optional)
            gaze_json = None
            gaze_candidates = [
                os.path.join(args.videos_dir, f"{video_id}_sam3rf_gaze.json"),
                os.path.join(args.videos_dir, f"{video_id}_gaze.json"),
            ]
            for candidate in gaze_candidates:
                if os.path.exists(candidate):
                    gaze_json = candidate
                    break
            
            # Add to Queue
            batch_tasks.append((
                video_path,
                events_json,
                output_path,
                api_key,
                args.model,
                gaze_json,
                args.max_events,
            ))

        print(f"Queued {len(batch_tasks)} tasks. Skipped {skipped} existing.")
        
        if not batch_tasks:
            print("No tasks to run.")
            return

        print(f"Starting execution with {args.workers} threads...")
        
        # Execute Parallel
        processed_count = 0
        failed_count = 0
        
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            # Map returns iterator of results
            results = list(tqdm(
                executor.map(lambda args: process_single_video(*args), batch_tasks), 
                total=len(batch_tasks),
                desc="Processing Videos",
                file=sys.stdout
            ))
            
        processed_count = sum(1 for r in results if r)
        failed_count = sum(1 for r in results if not r)

        print(f"\n{'='*60}")
        print(f"Batch Processing Complete")
        print(f"{'='*60}")
        print(f"  Processed: {processed_count}")
        print(f"  Skipped:   {skipped}")
        print(f"  Failed:    {failed_count}")
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
