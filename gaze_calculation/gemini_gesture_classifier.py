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

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("Warning: google-generativeai not installed. Install: pip install google-generativeai")


# Social gesture taxonomy
GESTURE_TAXONOMY = {
    "deictic": {
        "pointing": "Directing attention to an object or location by pointing",
        "showing": "Presenting or displaying an object to another person",
        "giving": "Offering or handing over an object to another person",
        "reaching": "Extending arm to acquire or touch an object",
    },
    "social_attention": {
        "joint_attention": "Multiple people looking at the same target",
        "gaze_following": "One person looking where another person looked",
        "turn_taking": "Exchanging conversational floor (looking at new speaker)",
        "attention_capture": "One person attracting attention of others",
    },
    "other": {
        "no_gesture": "No clear social gesture occurring",
        "unclear": "Gesture is ambiguous or cannot be determined",
    }
}


@dataclass
class GestureClassification:
    """Classification result for a candidate event."""
    event_id: int
    gesture_category: str  # "deictic", "social_attention", "other"
    gesture_type: str  # Specific gesture from taxonomy
    confidence: float
    description: str
    initiator_id: Optional[int]
    responder_ids: List[int]
    target_region: Optional[Tuple[float, float]]  # Normalized (x, y)
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
    max_frames: int = 10,
    resize: Tuple[int, int] = (640, 360),
) -> List[str]:
    """
    Extract frames from video and convert to base64 for API.
    
    Args:
        video_path: Path to video
        max_frames: Maximum frames to extract
        resize: Resize frames to this size
    
    Returns:
        List of base64-encoded JPEG frames
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Sample frames evenly
    if total_frames <= max_frames:
        frame_indices = list(range(total_frames))
    else:
        frame_indices = np.linspace(0, total_frames - 1, max_frames, dtype=int).tolist()
    
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
    Build a structured prompt for Gemini to classify the gesture.
    """
    prompt = f"""Analyze these video frames for social gestures. This is research on non-verbal communication.

## Event Context
- Event type detected by gaze analysis: {event.get('event_type', 'unknown')}
- Time window: {event.get('start_time', 0):.2f}s - {event.get('end_time', 0):.2f}s
- Persons involved (IDs): {event.get('persons_involved', [])}
- Detection confidence: {event.get('confidence', 0):.2f}

## Gaze Data
{json.dumps(event.get('details', {}), indent=2)}

## Task
Classify what social gesture (if any) is occurring in these frames.

## Gesture Categories

### Deictic Gestures:
- **pointing**: Directing attention to object/location by pointing
- **showing**: Presenting/displaying object to another person
- **giving**: Offering/handing over object to another person
- **reaching**: Extending arm to acquire or touch object

### Social Attention Patterns:
- **joint_attention**: Multiple people looking at same target
- **gaze_following**: One person looking where another looked
- **turn_taking**: Exchanging conversational floor
- **attention_capture**: One person attracting attention of others

### Other:
- **no_gesture**: No clear social gesture
- **unclear**: Cannot determine

## Response Format (JSON only):
{{
    "gesture_category": "deictic" | "social_attention" | "other",
    "gesture_type": "<specific type from categories above>",
    "confidence": <0.0-1.0>,
    "description": "<brief description of what's happening>",
    "initiator_id": <person ID who initiated, or null>,
    "responder_ids": [<list of person IDs responding>],
    "target_region": [<x>, <y>] or null (normalized 0-1 coordinates of gesture target)
}}

Respond with ONLY the JSON object, no other text.
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
) -> List[GestureClassification]:
    """
    Classify all candidate events using Gemini.
    """
    if api_key:
        genai.configure(api_key=api_key)
    
    results = []
    events_to_process = events[:max_events] if max_events else events
    
    for i, event in enumerate(events_to_process):
        print(f"Classifying event {i+1}/{len(events_to_process)}: {event.get('event_type')}")
        
        # Extract video clip
        clip_path = extract_video_clip(
            video_path,
            event["start_frame"],
            event["end_frame"],
            context_frames=5,
            max_frames=20,
        )
        
        try:
            # Convert to frames
            frames_b64 = video_to_frames_base64(clip_path, max_frames=8)
            
            # Build prompt
            prompt = build_classification_prompt(event, gaze_data)
            
            # Query Gemini
            raw_response, parsed = classify_with_gemini(frames_b64, prompt, model_name)
            
            if parsed:
                result = GestureClassification(
                    event_id=event.get("event_id", i),
                    gesture_category=parsed.get("gesture_category", "other"),
                    gesture_type=parsed.get("gesture_type", "unclear"),
                    confidence=parsed.get("confidence", 0.0),
                    description=parsed.get("description", ""),
                    initiator_id=parsed.get("initiator_id"),
                    responder_ids=parsed.get("responder_ids", []),
                    target_region=tuple(parsed["target_region"]) if parsed.get("target_region") else None,
                    raw_response=raw_response,
                )
            else:
                result = GestureClassification(
                    event_id=event.get("event_id", i),
                    gesture_category="other",
                    gesture_type="unclear",
                    confidence=0.0,
                    description="Failed to parse Gemini response",
                    initiator_id=None,
                    responder_ids=[],
                    target_region=None,
                    raw_response=raw_response,
                )
            
            results.append(result)
            
        finally:
            # Cleanup temp file
            if os.path.exists(clip_path):
                os.remove(clip_path)
    
    return results


def save_classifications(
    results: List[GestureClassification],
    events_data: Dict,
    output_path: str,
) -> None:
    """Save classification results to JSON."""
    output = {
        "video_path": events_data.get("video_path"),
        "num_events_classified": len(results),
        "gesture_counts": {},
        "classifications": [],
    }
    
    # Count gestures
    for r in results:
        key = f"{r.gesture_category}/{r.gesture_type}"
        output["gesture_counts"][key] = output["gesture_counts"].get(key, 0) + 1
    
    # Convert to dicts
    for r in results:
        output["classifications"].append({
            "event_id": r.event_id,
            "gesture_category": r.gesture_category,
            "gesture_type": r.gesture_type,
            "confidence": r.confidence,
            "description": r.description,
            "initiator_id": r.initiator_id,
            "responder_ids": r.responder_ids,
            "target_region": list(r.target_region) if r.target_region else None,
        })
    
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"Saved {len(results)} classifications to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Classify social gestures using Gemini")
    parser.add_argument("--video_path", type=str, required=True, help="Path to video")
    parser.add_argument("--events_json", type=str, required=True, help="Path to candidate events JSON")
    parser.add_argument("--gaze_json", type=str, default=None, help="Path to gaze annotation JSON (optional)")
    parser.add_argument("--output_json", type=str, default=None, help="Output path")
    parser.add_argument("--api_key", type=str, default=None, help="Gemini API key (or set GOOGLE_API_KEY env)")
    parser.add_argument("--model", type=str, default="gemini-2.0-flash", help="Gemini model to use")
    parser.add_argument("--max_events", type=int, default=None, help="Max events to classify")
    
    args = parser.parse_args()
    
    if args.output_json is None:
        base = os.path.splitext(args.events_json)[0].replace("_events", "")
        args.output_json = f"{base}_gestures.json"
    
    # Get API key
    api_key = args.api_key or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("ERROR: No API key provided. Set GOOGLE_API_KEY env or use --api_key")
        return
    
    print(f"Loading events from: {args.events_json}")
    with open(args.events_json, 'r') as f:
        events_data = json.load(f)
    
    gaze_data = {}
    if args.gaze_json:
        print(f"Loading gaze data from: {args.gaze_json}")
        with open(args.gaze_json, 'r') as f:
            gaze_data = json.load(f)
    
    events = events_data.get("events", [])
    print(f"Found {len(events)} candidate events")
    
    print("Classifying with Gemini...")
    results = classify_events(
        args.video_path,
        events,
        gaze_data,
        api_key=api_key,
        model_name=args.model,
        max_events=args.max_events,
    )
    
    # Print summary
    print(f"\nClassification Summary:")
    gesture_counts = {}
    for r in results:
        key = f"{r.gesture_category}/{r.gesture_type}"
        gesture_counts[key] = gesture_counts.get(key, 0) + 1
    
    for key, count in sorted(gesture_counts.items()):
        print(f"  - {key}: {count}")
    
    save_classifications(results, events_data, args.output_json)


if __name__ == "__main__":
    main()
