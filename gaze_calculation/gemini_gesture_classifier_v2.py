#!/usr/bin/env python3
"""
Gemini Gesture Classifier V2 - Full Video Gesture Detection

Two-stage classification:
- Stage 1: Validate candidate gaze events (same as v1)
- Stage 2: Independently scan the FULL VIDEO for gestures (not tied to events)

This version detects gestures anywhere in the video, not just within event windows.
"""

import argparse
import json
import os
import sys
import time
import random
import functools
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
from pathlib import Path

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)

# Load .env
ENV_FILE_PATH = "/u/arkimjh/code/ECCV-jh/.env"
try:
    from dotenv import load_dotenv
    if os.path.exists(ENV_FILE_PATH):
        load_dotenv(ENV_FILE_PATH)
    else:
        load_dotenv()
except ImportError:
    pass

try:
    from google import genai
    from google.genai import types
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("Warning: google-genai not installed. Install: pip install google-genai")


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


@dataclass
class GestureInstance:
    """A detected gesture instance."""
    gesture_type: str  # pointing, showing, giving, reaching
    initiator_id: int
    start_time: float
    end_time: float
    target_type: str  # person, object, location
    target_person_id: Optional[int]
    target_description: str
    confidence: float = 1.0


@dataclass
class EventValidation:
    """Validation result for a gaze event."""
    event_id: int
    event_confirmed: bool
    event_type: str
    start_time: float
    end_time: float


@retry_with_backoff()
def upload_video_to_gemini(client, video_path: str):
    """Upload video to Gemini."""
    try:
        if not os.path.exists(video_path):
            print(f"Video not found: {video_path}")
            return None
            
        print(f"Uploading video: {os.path.basename(video_path)}...")
        video_file = client.files.upload(file=video_path, config={"mime_type": "video/mp4"})
        
        print(f"Waiting for processing (URI: {video_file.uri})...")
        wait_time = 0
        max_wait = 300
        while video_file.state == types.FileState.PROCESSING and wait_time < max_wait:
            time.sleep(5)
            wait_time += 5
            try:
                video_file = client.files.get(name=video_file.name)
            except:
                pass
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


def delete_uploaded_file(client, file_obj) -> None:
    """Delete an uploaded file from Gemini."""
    if file_obj is not None:
        try:
            client.files.delete(name=file_obj.name)
            print(f"Deleted remote file: {file_obj.name}")
        except:
            pass


def build_validation_prompt(event: Dict, persons: str) -> str:
    """Stage 1: Lightweight prompt for event validation only."""
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


def build_full_video_gesture_prompt(video_duration: float) -> str:
    """Stage 2: Prompt for detecting ALL gestures in the full video."""
    
    prompt = f"""Scan this entire video ({video_duration:.1f} seconds) and detect ALL deictic gestures.

## About This Video
- The video has annotations: Colored bounding boxes (P0, P1, P2...) and gaze direction lines.
- Scan the ENTIRE video from start to end.

## Deictic Gesture Types (Be STRICT)

**pointing** - Extended arm with finger pointing at a target
  ✓ YES: Clear arm extension with finger/hand pointing at object/person/location
  ✗ NOT: Talking with hands, casual hand gestures, arm resting

**showing** - Presenting an object for others to visually inspect
  ✓ YES: Holding object UP and orienting it toward another person
  ✗ NOT: Simply holding an object, object on table/lap

**giving** - Transferring an object to another person
  ✓ YES: Extending hand WITH object toward recipient
  ✗ NOT: Object at rest, just holding something

**reaching** - Extending hand toward an object to acquire/touch it
  ✓ YES: Arm extending TOWARD object with intent to grab/touch
  ✗ NOT: Arms relaxed, casual arm movement

## Rules
1. Report ONLY clearly visible gestures (>80% confidence)
2. Each gesture should have start_time and end_time in seconds
3. Person IDs match the annotations (P0, P1, P2...)
4. Return EMPTY array if no clear gestures are visible

## Response Format (JSON only)

{{{{
    "gestures": [
        {{{{
            "gesture_type": "pointing" | "showing" | "giving" | "reaching",
            "initiator_id": <person ID number>,
            "start_time": <seconds>,
            "end_time": <seconds>,
            "target_type": "person" | "object" | "location",
            "target_person_id": <ID if person, else null>,
            "target_description": "<what/who is the target>",
            "confidence": <0.0 to 1.0, your certainty>
        }}}}
    ]
}}}}

Return EMPTY array if no gestures detected. JSON only.
"""
    return prompt


@retry_with_backoff()
def classify_with_gemini(client, prompt: str, video_file, model_name: str) -> Tuple[str, Dict]:
    """Send prompt to Gemini with video."""
    if not GEMINI_AVAILABLE:
        return "ERROR: Gemini not available", {}
    
    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        temperature=0.2,
    )
    
    try:
        response = client.models.generate_content(
            model=model_name,
            contents=[video_file, prompt],
            config=config
        )
        
        raw_text = response.text.strip() if response.text else ""
        
        # Parse JSON
        try:
            parsed = json.loads(raw_text)
        except json.JSONDecodeError:
            # Try to extract JSON from response
            import re
            json_match = re.search(r'\{.*\}', raw_text, re.DOTALL)
            if json_match:
                try:
                    parsed = json.loads(json_match.group())
                except:
                    parsed = {}
            else:
                parsed = {}
        
        return raw_text, parsed
        
    except Exception as e:
        if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e) or "500" in str(e):
            raise
        print(f"Error: {e}")
        return str(e), {}


def get_video_duration(video_path: str) -> float:
    """Get video duration in seconds."""
    import cv2
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    return frame_count / fps if fps > 0 else 0


def classify_video_v2(
    video_path: str,
    events: List[Dict],
    api_key: str,
    model_name: str = "models/gemini-2.5-flash",
    max_events: int = 15,
    min_confidence: float = 0.75,
) -> Tuple[List[EventValidation], List[GestureInstance]]:
    """
    Two-stage classification:
    - Stage 1: Validate gaze events
    - Stage 2: Scan full video for gestures (independent)
    """
    if not GEMINI_AVAILABLE:
        print("ERROR: google-genai not installed")
        return [], []
    
    client = genai.Client(api_key=api_key)
    
    # Find visualization video
    base_name = os.path.splitext(video_path)[0]
    viz_video_path = None
    for suffix in ['_sam3rf_viz.mp4', '_viz.mp4']:
        candidate = base_name + suffix
        if os.path.exists(candidate):
            viz_video_path = candidate
            break
    
    source_video = viz_video_path if viz_video_path else video_path
    print(f"Using video: {source_video}")
    
    # Get video duration
    video_duration = get_video_duration(source_video)
    print(f"Video duration: {video_duration:.1f}s")
    
    # Upload video
    print("Uploading video to Gemini...")
    video_file = upload_video_to_gemini(client, source_video)
    if not video_file:
        return [], []
    
    validations = []
    gestures = []
    
    try:
        # =================== STAGE 1: EVENT VALIDATION ===================
        # Filter events by confidence
        filtered_events = [e for e in events if e.get("confidence", 1.0) >= min_confidence]
        sorted_events = sorted(filtered_events, key=lambda e: e.get("confidence", 0), reverse=True)
        events_to_validate = sorted_events[:max_events]
        
        print(f"\nStage 1: Validating {len(events_to_validate)} events...")
        
        for i, event in enumerate(events_to_validate):
            event_id = event.get("event_id", i)
            persons = ", ".join([f"P{p}" for p in event.get("persons_involved", [])])
            prompt = build_validation_prompt(event, persons)
            
            print(f"  [{i+1}/{len(events_to_validate)}] {event.get('event_type')} @ {event.get('start_time'):.1f}s")
            
            raw_response, parsed = classify_with_gemini(client, prompt, video_file, model_name)
            
            confirmed = parsed.get("event_confirmed", False) if parsed else False
            
            validations.append(EventValidation(
                event_id=event_id,
                event_confirmed=confirmed,
                event_type=event.get("event_type", "unknown"),
                start_time=event.get("start_time", 0),
                end_time=event.get("end_time", 0),
            ))
            
            time.sleep(3)
        
        confirmed_count = sum(1 for v in validations if v.event_confirmed)
        print(f"\nStage 1 Complete: {confirmed_count}/{len(validations)} events confirmed")
        
        # =================== STAGE 2: FULL VIDEO GESTURE SCAN ===================
        print(f"\nStage 2: Scanning full video for gestures...")
        
        prompt = build_full_video_gesture_prompt(video_duration)
        raw_response, parsed = classify_with_gemini(client, prompt, video_file, model_name)
        
        if parsed and "gestures" in parsed:
            for g in parsed["gestures"]:
                gestures.append(GestureInstance(
                    gesture_type=g.get("gesture_type", "unknown"),
                    initiator_id=g.get("initiator_id", 0),
                    start_time=g.get("start_time", 0),
                    end_time=g.get("end_time", 0),
                    target_type=g.get("target_type", "unknown"),
                    target_person_id=g.get("target_person_id"),
                    target_description=g.get("target_description", ""),
                    confidence=g.get("confidence", 1.0),
                ))
        
        print(f"Stage 2 Complete: {len(gestures)} gestures detected")
        
    finally:
        print("Cleaning up...")
        delete_uploaded_file(client, video_file)
    
    return validations, gestures


def save_results_v2(
    validations: List[EventValidation],
    gestures: List[GestureInstance],
    video_path: str,
    output_path: str,
) -> None:
    """Save v2 results to JSON."""
    # Count gesture types
    gesture_counts = {}
    for g in gestures:
        gesture_counts[g.gesture_type] = gesture_counts.get(g.gesture_type, 0) + 1
    
    output = {
        "video_path": video_path,
        "classifier_version": "v2",
        "num_events_validated": len(validations),
        "events_confirmed": sum(1 for v in validations if v.event_confirmed),
        "events_rejected": sum(1 for v in validations if not v.event_confirmed),
        "num_gestures_detected": len(gestures),
        "gesture_counts": gesture_counts,
        "event_validations": [asdict(v) for v in validations],
        "gestures": [asdict(g) for g in gestures],
    }
    
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"Saved: {output_path}")


def process_single_video_v2(
    video_path: str,
    events_json: str,
    output_path: str,
    api_key: str,
    model_name: str = "models/gemini-2.5-flash",
    max_events: int = 15,
) -> bool:
    """Process a single video with v2 classifier."""
    print(f"\n{'='*60}")
    print(f"Processing: {os.path.basename(events_json)}")
    print(f"{'='*60}")
    
    # Load events
    with open(events_json, 'r') as f:
        events_data = json.load(f)
    events = events_data.get("events", [])
    
    print(f"Loaded {len(events)} candidate events")
    
    if not events:
        print("No events to validate, but will still scan for gestures...")
    
    # Run v2 classification
    validations, gestures = classify_video_v2(
        video_path=video_path,
        events=events,
        api_key=api_key,
        model_name=model_name,
        max_events=max_events,
    )
    
    # Save results
    save_results_v2(validations, gestures, video_path, output_path)
    
    # Print summary
    print(f"\nSummary:")
    print(f"  Events confirmed: {sum(1 for v in validations if v.event_confirmed)}/{len(validations)}")
    print(f"  Gestures detected: {len(gestures)}")
    for g_type, count in sorted({g.gesture_type: sum(1 for x in gestures if x.gesture_type == g.gesture_type) for g in gestures}.items()):
        print(f"    - {g_type}: {count}")
    
    return True


def main():
    parser = argparse.ArgumentParser(description="Gemini Gesture Classifier V2 - Full Video Gesture Detection")
    
    # Single video mode
    parser.add_argument("--video_path", type=str, help="Path to video file")
    parser.add_argument("--events_json", type=str, help="Path to events JSON")
    
    # Batch mode
    parser.add_argument("--video_dir", type=str, help="Directory with videos for batch processing")
    parser.add_argument("--events_dir", type=str, help="Directory with events JSONs for batch")
    
    # Common args
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")
    parser.add_argument("--api_key", type=str, default=None, help="Gemini API key")
    parser.add_argument("--model", type=str, default="models/gemini-2.5-flash", help="Gemini model")
    parser.add_argument("--max_events", type=int, default=15, help="Max events to validate")
    parser.add_argument("--skip_existing", action="store_true", help="Skip if output exists")
    
    args = parser.parse_args()
    
    api_key = args.api_key or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("ERROR: No API key. Set GOOGLE_API_KEY or use --api_key")
        return
    
    if args.video_dir and args.events_dir:
        # Batch mode
        print(f"Batch mode: {args.video_dir}")
        
        import glob
        events_files = sorted(glob.glob(os.path.join(args.events_dir, "*_events.json")))
        
        for events_json in events_files:
            base = os.path.basename(events_json).replace("_sam3rf_gaze_events.json", "").replace("_events.json", "")
            
            # Find video
            video_candidates = [
                os.path.join(args.video_dir, f"{base}.mp4"),
                os.path.join(args.video_dir, f"{base}_sam3rf_viz.mp4"),
            ]
            video_path = None
            for c in video_candidates:
                if os.path.exists(c):
                    video_path = c
                    break
            
            if not video_path:
                print(f"Skipping {base}: no video found")
                continue
            
            output_path = os.path.join(args.output_dir or args.events_dir, f"{base}_gestures_v2.json")
            
            if args.skip_existing and os.path.exists(output_path):
                print(f"Skipping {base}: output exists")
                continue
            
            process_single_video_v2(
                video_path=video_path,
                events_json=events_json,
                output_path=output_path,
                api_key=api_key,
                model_name=args.model,
                max_events=args.max_events,
            )
    
    elif args.video_path and args.events_json:
        # Single video mode
        base = os.path.basename(args.video_path).replace(".mp4", "")
        output_path = os.path.join(args.output_dir or ".", f"{base}_gestures_v2.json")
        
        process_single_video_v2(
            video_path=args.video_path,
            events_json=args.events_json,
            output_path=output_path,
            api_key=api_key,
            model_name=args.model,
            max_events=args.max_events,
        )
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
