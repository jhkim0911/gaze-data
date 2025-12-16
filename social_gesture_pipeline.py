#!/usr/bin/env python3
"""
Social Gesture Annotation Pipeline

Full pipeline orchestrator for social gesture detection:
1. Load existing gaze annotations (from sam3_retinaface_gaze_pipeline)
2. Extract gaze features
3. Detect candidate events
4. Classify gestures with Gemini

Usage:
    python social_gesture_pipeline.py \
        --video_path /path/to/video.mp4 \
        --gaze_json /path/to/gaze.json \
        --output_dir /path/to/output \
        --api_key YOUR_GEMINI_API_KEY
"""

import argparse
import json
import os
from typing import Dict, List, Optional

from gaze_feature_extractor import extract_gaze_features, save_features
from candidate_event_detector import detect_all_events, save_events
from gemini_gesture_classifier import classify_events, save_classifications


def run_full_pipeline(
    video_path: str,
    gaze_json: str,
    output_dir: str,
    api_key: Optional[str] = None,
    skip_gemini: bool = False,
    max_events: Optional[int] = None,
    model_name: str = "gemini-2.0-flash",
) -> Dict:
    """
    Run the full social gesture annotation pipeline.
    
    Args:
        video_path: Path to video file
        gaze_json: Path to gaze annotation JSON
        output_dir: Directory for output files
        api_key: Gemini API key (optional if skip_gemini=True)
        skip_gemini: If True, only run feature extraction and event detection
        max_events: Maximum events to classify with Gemini
        model_name: Gemini model to use
    
    Returns:
        Dict with paths to all output files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    
    output_paths = {
        "features": os.path.join(output_dir, f"{video_name}_features.json"),
        "events": os.path.join(output_dir, f"{video_name}_events.json"),
        "gestures": os.path.join(output_dir, f"{video_name}_gestures.json"),
    }
    
    # Step 1: Load gaze data
    print("=" * 60)
    print("STEP 1: Loading gaze annotations")
    print("=" * 60)
    with open(gaze_json, 'r') as f:
        gaze_data = json.load(f)
    print(f"  Loaded {gaze_data['processed_frames']} frames, {len(gaze_data['persons_summary'])} persons")
    
    # Step 2: Extract gaze features
    print("\n" + "=" * 60)
    print("STEP 2: Extracting gaze features")
    print("=" * 60)
    features = extract_gaze_features(gaze_data)
    save_features(features, output_paths["features"])
    
    print(f"  Mean convergence: {features.mean_convergence_score:.3f}")
    print(f"  Max convergence: {features.max_convergence_score:.3f}")
    print(f"  High velocity events: {len(features.high_velocity_events)}")
    
    # Step 3: Detect candidate events
    print("\n" + "=" * 60)
    print("STEP 3: Detecting candidate events")
    print("=" * 60)
    
    # Load features as dict for event detector
    with open(output_paths["features"], 'r') as f:
        features_dict = json.load(f)
    
    events = detect_all_events(features_dict)
    save_events(events, features_dict, output_paths["events"])
    
    event_counts = {}
    for e in events:
        event_counts[e.event_type] = event_counts.get(e.event_type, 0) + 1
    
    print(f"\n  Detected {len(events)} total events:")
    for t, c in sorted(event_counts.items()):
        print(f"    - {t}: {c}")
    
    # Step 4: Classify with Gemini (optional)
    if not skip_gemini:
        print("\n" + "=" * 60)
        print("STEP 4: Classifying gestures with Gemini")
        print("=" * 60)
        
        if not api_key:
            api_key = os.environ.get("GOOGLE_API_KEY")
        
        if not api_key:
            print("  WARNING: No API key provided, skipping Gemini classification")
            print("  Set GOOGLE_API_KEY or use --api_key")
        else:
            # Load events as dicts
            with open(output_paths["events"], 'r') as f:
                events_data = json.load(f)
            
            events_list = events_data.get("events", [])
            
            # Classify
            classifications = classify_events(
                video_path,
                events_list,
                gaze_data,
                api_key=api_key,
                model_name=model_name,
                max_events=max_events,
            )
            
            save_classifications(classifications, events_data, output_paths["gestures"])
            
            # Summary
            gesture_counts = {}
            for c in classifications:
                key = f"{c.gesture_category}/{c.gesture_type}"
                gesture_counts[key] = gesture_counts.get(key, 0) + 1
            
            print(f"\n  Classified {len(classifications)} events:")
            for key, count in sorted(gesture_counts.items()):
                print(f"    - {key}: {count}")
    else:
        print("\n  Skipping Gemini classification (--skip_gemini)")
    
    # Summary
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"Output files:")
    for name, path in output_paths.items():
        if os.path.exists(path):
            print(f"  - {name}: {path}")
    
    return output_paths


def main():
    parser = argparse.ArgumentParser(description="Social Gesture Annotation Pipeline")
    parser.add_argument("--video_path", type=str, required=True, help="Path to video")
    parser.add_argument("--gaze_json", type=str, required=True, help="Path to gaze annotation JSON")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")
    parser.add_argument("--api_key", type=str, default=None, help="Gemini API key")
    parser.add_argument("--skip_gemini", action="store_true", help="Skip Gemini classification")
    parser.add_argument("--max_events", type=int, default=None, help="Max events to classify")
    parser.add_argument("--model", type=str, default="gemini-2.0-flash", help="Gemini model")
    
    args = parser.parse_args()
    
    if args.output_dir is None:
        args.output_dir = os.path.dirname(args.gaze_json)
    
    run_full_pipeline(
        video_path=args.video_path,
        gaze_json=args.gaze_json,
        output_dir=args.output_dir,
        api_key=args.api_key,
        skip_gemini=args.skip_gemini,
        max_events=args.max_events,
        model_name=args.model,
    )


if __name__ == "__main__":
    main()
