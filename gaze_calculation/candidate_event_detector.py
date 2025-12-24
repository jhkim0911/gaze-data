#!/usr/bin/env python3
"""
Candidate Event Detector for Social Gesture Detection

Detects candidate temporal windows where social gestures may have occurred
based on gaze features extracted from gaze_feature_extractor.py.

Event types detected:
1. Sudden Gaze Shift - Large velocity spike for one or more persons
2. Joint Attention - Multiple people looking at same region
3. Gaze Following - Person B looks where Person A looked with temporal lag
4. Attention Capture - Sudden multi-person gaze shift
5. Mutual Gaze - Two people looking at each other (eye contact)

Usage:
    python candidate_event_detector.py --input_features /path/to/features.json --output_json /path/to/events.json
"""

import argparse
import json
import os
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
import numpy as np


@dataclass
class CandidateEvent:
    """A detected candidate event for social gesture classification."""
    event_id: int
    event_type: str  # "sudden_gaze_shift", "joint_attention", "gaze_following", "attention_capture"
    start_time: float
    end_time: float
    start_frame: int
    end_frame: int
    confidence: float  # 0-1 confidence score
    persons_involved: List[int]
    
    # Event-specific details
    details: Dict


def gaze_inside_bbox(gaze_point, bbox, margin=0.02):
    """
    Check if gaze point falls inside a bounding box with margin.
    
    Args:
        gaze_point: (x, y) normalized gaze coordinates
        bbox: [x1, y1, x2, y2] normalized bounding box
        margin: Tolerance margin (default 2% of frame)
        
    Returns:
        True if gaze is inside or very close to bbox
    """
    if gaze_point is None or bbox is None:
        return False
    x1, y1, x2, y2 = bbox
    gx, gy = gaze_point
    return (x1 - margin <= gx <= x2 + margin and 
            y1 - margin <= gy <= y2 + margin)


def detect_sudden_gaze_shifts(
    frame_features: List[Dict],
    velocity_threshold: float = 0.7,
    min_duration_sec: float = 0.5,  # At 2fps = 1 frame minimum
    max_duration_sec: float = 1.5,  # At 2fps = 3 frames max
) -> List[CandidateEvent]:
    """
    Detect sudden gaze shift events where a person's gaze moves rapidly.
    
    These could indicate:
    - Responding to pointing
    - Attention capture by external stimulus
    - Gaze following
    """
    events = []
    event_id = 0
    
    # Group consecutive high-velocity frames by person
    person_high_vel_frames: Dict[int, List[Dict]] = {}
    
    for ff in frame_features:
        for pid, vel in ff.get("person_velocities", {}).items():
            pid = int(pid)
            if vel > velocity_threshold:
                if pid not in person_high_vel_frames:
                    person_high_vel_frames[pid] = []
                person_high_vel_frames[pid].append({
                    "frame_idx": ff["frame_idx"],
                    "timestamp": ff["timestamp"],
                    "velocity": vel,
                })
    
    # Cluster consecutive high-velocity frames into events
    for pid, frames in person_high_vel_frames.items():
        if not frames:
            continue
        
        # Sort by timestamp
        frames = sorted(frames, key=lambda x: x["timestamp"])
        
        # Cluster consecutive frames (within 0.6s gap)
        clusters = []
        current_cluster = [frames[0]]
        
        for i in range(1, len(frames)):
            if frames[i]["timestamp"] - frames[i-1]["timestamp"] < 0.6:
                current_cluster.append(frames[i])
            else:
                if current_cluster:
                    clusters.append(current_cluster)
                current_cluster = [frames[i]]
        
        if current_cluster:
            clusters.append(current_cluster)
        
        # Create events from clusters
        for cluster in clusters:
            start_time = cluster[0]["timestamp"]
            end_time = cluster[-1]["timestamp"]
            duration = end_time - start_time
            
            if min_duration_sec <= duration <= max_duration_sec:
                max_vel = max(f["velocity"] for f in cluster)
                confidence = min(1.0, max_vel / (velocity_threshold * 2))
                
                events.append(CandidateEvent(
                    event_id=event_id,
                    event_type="sudden_gaze_shift",
                    start_time=start_time,
                    end_time=end_time,
                    start_frame=cluster[0]["frame_idx"],
                    end_frame=cluster[-1]["frame_idx"],
                    confidence=confidence,
                    persons_involved=[pid],
                    details={
                        "max_velocity": max_vel,
                        "duration": duration,
                        "num_frames": len(cluster),
                    }
                ))
                event_id += 1
    
    return events


def detect_joint_attention(
    frame_features: List[Dict],
    convergence_threshold: float = 0.6,
    min_persons: int = 2,
    min_duration_sec: float = 0.5,
) -> List[CandidateEvent]:
    """
    Detect joint attention events where multiple people look at the same region.
    
    These could indicate:
    - Shared attention to an object
    - Group focus on speaker
    - Response to pointing/showing gesture
    """
    events = []
    event_id = 1000  # Start from 1000 to distinguish from gaze shift events
    
    # Find frames with high convergence
    high_conv_frames = []
    for ff in frame_features:
        conv_score = ff.get("gaze_convergence_score", 0)
        num_faces = ff.get("num_faces_detected", 0)
        
        if conv_score >= convergence_threshold and num_faces >= min_persons:
            high_conv_frames.append({
                "frame_idx": ff["frame_idx"],
                "timestamp": ff["timestamp"],
                "convergence_score": conv_score,
                "center": ff.get("gaze_convergence_center"),
                "num_persons": num_faces,
                "person_gaze_points": ff.get("person_gaze_points", {}),
            })
    
    if not high_conv_frames:
        return events
    
    # Cluster consecutive high-convergence frames
    high_conv_frames = sorted(high_conv_frames, key=lambda x: x["timestamp"])
    
    clusters = []
    current_cluster = [high_conv_frames[0]]
    
    for i in range(1, len(high_conv_frames)):
        if high_conv_frames[i]["timestamp"] - high_conv_frames[i-1]["timestamp"] < 0.6:
            current_cluster.append(high_conv_frames[i])
        else:
            if current_cluster:
                clusters.append(current_cluster)
            current_cluster = [high_conv_frames[i]]
    
    if current_cluster:
        clusters.append(current_cluster)
    
    # Create events from clusters
    for cluster in clusters:
        start_time = cluster[0]["timestamp"]
        end_time = cluster[-1]["timestamp"]
        duration = end_time - start_time
        
        if duration >= min_duration_sec:
            mean_conv = np.mean([f["convergence_score"] for f in cluster])
            max_conv = max(f["convergence_score"] for f in cluster)
            
            # Find the peak convergence frame (most representative)
            peak_frame = max(cluster, key=lambda f: f["convergence_score"])
            
            # Get persons from peak frame only (not accumulated across all frames)
            all_persons = set()
            for pid, gaze in peak_frame.get("person_gaze_points", {}).items():
                if gaze is not None:
                    all_persons.add(int(pid))
            
            # Get convergence center from peak frame
            avg_center = peak_frame.get("center")
            
            # Filter to only persons converging toward the center
            if avg_center:
                converging_persons = set()
                DIST_THRESHOLD = 0.2  # 20% of frame
                
                for pid, gaze in peak_frame.get("person_gaze_points", {}).items():
                    if gaze is not None:
                        dist = np.sqrt((gaze[0] - avg_center[0])**2 + (gaze[1] - avg_center[1])**2)
                        if dist < DIST_THRESHOLD:
                            converging_persons.add(int(pid))
                
                # Only use converging persons if we have at least min_persons
                if len(converging_persons) >= min_persons:
                    all_persons = converging_persons
            
            events.append(CandidateEvent(
                event_id=event_id,
                event_type="joint_attention",
                start_time=start_time,
                end_time=end_time,
                start_frame=cluster[0]["frame_idx"],
                end_frame=cluster[-1]["frame_idx"],
                confidence=mean_conv,
                persons_involved=list(all_persons),
                details={
                    "max_convergence": max_conv,
                    "mean_convergence": mean_conv,
                    "duration": duration,
                    "attention_center": list(avg_center) if avg_center else None,
                    "num_persons": len(all_persons),
                }
            ))
            event_id += 1
    
    return events


def detect_gaze_following(
    frame_features: List[Dict],
    distance_threshold: float = 0.03,  # Very strict for obvious matches
    min_lag_sec: float = 1.0,
    max_lag_sec: float = 2.0,
    min_event_confidence: float = 0.9, # Only very high-confidence events
) -> List[CandidateEvent]:
    """
    Detect gaze following events where Person B looks where Person A looked.
    
    These could indicate:
    - Response to pointing gesture
    - Social referencing
    - Checking what someone is looking at
    """
    events = []
    event_id = 2000
    
    # Build gaze history per person (only high-confidence gaze)
    person_gaze_history: Dict[int, List[Tuple[float, int, Tuple[float, float]]]] = {}
    
    for ff in frame_features:
        confidences = ff.get("person_gaze_confidences", {})
        for pid_str, gaze in ff.get("person_gaze_points", {}).items():
            pid = int(pid_str)
            conf = confidences.get(str(pid), confidences.get(pid, 0))
            # Only use gaze points with confidence >= 0.5
            if gaze is not None and conf >= 0.5:
                if pid not in person_gaze_history:
                    person_gaze_history[pid] = []
                person_gaze_history[pid].append((ff["timestamp"], ff["frame_idx"], tuple(gaze)))
    
    # For each pair of persons, check if B's gaze matches A's previous gaze
    person_ids = list(person_gaze_history.keys())
    
    for i, pid_a in enumerate(person_ids):
        for pid_b in person_ids[i+1:]:
            history_a = person_gaze_history[pid_a]
            history_b = person_gaze_history[pid_b]
            
            # Track detected events for this pair to avoid duplicates
            pair_events = []  # [(start_time, end_time, confidence, details), ...]
            
            # For each gaze point of A, check if B later looked at same place
            for t_a, frame_a, gaze_a in history_a:
                for t_b, frame_b, gaze_b in history_b:
                    lag = t_b - t_a
                    
                    if min_lag_sec <= lag <= max_lag_sec:
                        dist = np.sqrt((gaze_a[0] - gaze_b[0])**2 + (gaze_a[1] - gaze_b[1])**2)
                        
                        if dist < distance_threshold:
                            confidence = 1.0 - (dist / distance_threshold)
                            
                            # Only keep high-confidence events
                            if confidence >= min_event_confidence:
                                pair_events.append({
                                    "start_time": t_a,
                                    "end_time": t_b,
                                    "start_frame": frame_a,
                                    "end_frame": frame_b,
                                    "confidence": confidence,
                                    "lag_seconds": lag,
                                    "gaze_distance": dist,
                                    "gaze_target": list(gaze_a),
                                })
            
            # Merge overlapping events for this pair
            if not pair_events:
                continue
                
            pair_events.sort(key=lambda x: x["start_time"])
            merged = [pair_events[0]]
            
            for e in pair_events[1:]:
                last = merged[-1]
                # If overlapping (new start <= last end + 1 sec), merge them
                if e["start_time"] <= last["end_time"] + 1.0:
                    # Extend end time, keep best confidence
                    last["end_time"] = max(last["end_time"], e["end_time"])
                    last["end_frame"] = max(last["end_frame"], e["end_frame"])
                    if e["confidence"] > last["confidence"]:
                        last["confidence"] = e["confidence"]
                        last["lag_seconds"] = e["lag_seconds"]
                        last["gaze_distance"] = e["gaze_distance"]
                        last["gaze_target"] = e["gaze_target"]
                else:
                    merged.append(e)
            
            # Create events from merged list
            for e in merged:
                events.append(CandidateEvent(
                    event_id=event_id,
                    event_type="gaze_following",
                    start_time=e["start_time"],
                    end_time=e["end_time"],
                    start_frame=e["start_frame"],
                    end_frame=e["end_frame"],
                    confidence=e["confidence"],
                    persons_involved=[pid_a, pid_b],
                    details={
                        "leader_id": pid_a,
                        "follower_id": pid_b,
                        "lag_seconds": e["lag_seconds"],
                        "gaze_distance": e["gaze_distance"],
                        "gaze_target": e["gaze_target"],
                    }
                ))
                event_id += 1
    
    return events


def detect_attention_capture(
    frame_features: List[Dict],
    velocity_threshold: float = 0.4,  # Lowered from 0.6 to detect more events
    min_persons: int = 3,  # Keep at 3 to reduce noise
    time_window_sec: float = 0.5,
) -> List[CandidateEvent]:
    """
    Detect attention capture events where multiple people suddenly shift gaze.
    
    These could indicate:
    - External stimulus (sound, movement)
    - Pointing/waving gesture
    - Something entering the scene
    """
    events = []
    event_id = 3000
    
    # Find frames where multiple people have high velocity
    multi_shift_frames = []
    
    for ff in frame_features:
        high_vel_persons = [
            int(pid) for pid, vel in ff.get("person_velocities", {}).items()
            if vel > velocity_threshold
        ]
        
        if len(high_vel_persons) >= min_persons:
            multi_shift_frames.append({
                "frame_idx": ff["frame_idx"],
                "timestamp": ff["timestamp"],
                "persons": high_vel_persons,
                "velocities": {int(k): v for k, v in ff.get("person_velocities", {}).items()},
            })
    
    if not multi_shift_frames:
        return events
    
    # Cluster by time
    multi_shift_frames = sorted(multi_shift_frames, key=lambda x: x["timestamp"])
    
    clusters = []
    current_cluster = [multi_shift_frames[0]]
    
    for i in range(1, len(multi_shift_frames)):
        if multi_shift_frames[i]["timestamp"] - multi_shift_frames[i-1]["timestamp"] < time_window_sec:
            current_cluster.append(multi_shift_frames[i])
        else:
            if current_cluster:
                clusters.append(current_cluster)
            current_cluster = [multi_shift_frames[i]]
    
    if current_cluster:
        clusters.append(current_cluster)
    
    for cluster in clusters:
        all_persons = set()
        all_velocities = {}
        
        for f in cluster:
            all_persons.update(f["persons"])
            for pid, vel in f["velocities"].items():
                if pid not in all_velocities or vel > all_velocities[pid]:
                    all_velocities[pid] = vel
        
        if len(all_persons) >= min_persons:
            mean_vel = np.mean(list(all_velocities.values()))
            confidence = min(1.0, mean_vel / velocity_threshold)
            
            events.append(CandidateEvent(
                event_id=event_id,
                event_type="attention_capture",
                start_time=cluster[0]["timestamp"],
                end_time=cluster[-1]["timestamp"],
                start_frame=cluster[0]["frame_idx"],
                end_frame=cluster[-1]["frame_idx"],
                confidence=confidence,
                persons_involved=list(all_persons),
                details={
                    "num_persons": len(all_persons),
                    "mean_velocity": mean_vel,
                    "person_velocities": all_velocities,
                }
            ))
            event_id += 1
    
    return events


def detect_mutual_gaze(
    frame_features: List[Dict],
    distance_threshold: float = 0.15,  # Legacy param, not used with bbox method
    min_duration_sec: float = 1.0,  # At 2fps = 2 frames minimum
    min_confidence: float = 0.5,
) -> List[CandidateEvent]:
    """
    Detect mutual gaze events where two people look at each other (eye contact).
    
    Logic:
    - Person A's gaze_point is near Person B's face_center
    - AND Person B's gaze_point is near Person A's face_center
    - Both conditions sustained for min_duration_sec
    
    These indicate:
    - Direct social interaction
    - Conversation engagement
    - Non-verbal communication
    """
    events = []
    event_id = 4000  # Start from 4000 for mutual gaze events
    
    # Build per-frame mutual gaze candidates
    mutual_gaze_frames: Dict[str, List[Dict]] = {}  # "pidA_pidB" -> list of frames
    
    for ff in frame_features:
        face_bboxes = ff.get("person_face_bboxes", {})
        gaze_points = ff.get("person_gaze_points", {})
        gaze_confidences = ff.get("person_gaze_confidences", {})
        gaze_methods = ff.get("person_gaze_methods", {})
        
        # Get all person IDs with valid face_bbox and MEASURED gaze_point
        # (Don't use interpolated gaze for mutual gaze detection)
        valid_persons = []
        for pid_str in face_bboxes.keys():
            pid = int(pid_str)
            face_bbox = face_bboxes.get(str(pid)) or face_bboxes.get(pid)
            gaze_point = gaze_points.get(str(pid)) or gaze_points.get(pid)
            gaze_conf = gaze_confidences.get(str(pid), gaze_confidences.get(pid, 0))
            gaze_method = gaze_methods.get(str(pid), gaze_methods.get(pid, ""))
            
            # Only use MEASURED gaze for mutual gaze (not interpolated)
            if face_bbox and gaze_point and gaze_conf >= min_confidence and gaze_method == "measured":
                valid_persons.append({
                    "pid": pid,
                    "face_bbox": face_bbox,  # [x1, y1, x2, y2]
                    "gaze_point": gaze_point,
                })
        
        # Check all pairs for mutual gaze using bbox containment
        for i, person_a in enumerate(valid_persons):
            for person_b in valid_persons[i+1:]:
                pid_a = person_a["pid"]
                pid_b = person_b["pid"]
                
                # Check if A's gaze is inside B's face bbox (with small margin)
                a_looks_at_b = gaze_inside_bbox(
                    person_a["gaze_point"], 
                    person_b["face_bbox"],
                    margin=0.02  # 2% margin for tolerance
                )
                
                # Check if B's gaze is inside A's face bbox
                b_looks_at_a = gaze_inside_bbox(
                    person_b["gaze_point"],
                    person_a["face_bbox"],
                    margin=0.02
                )
                
                # Both must be looking at each other's face
                if a_looks_at_b and b_looks_at_a:
                    pair_key = f"{min(pid_a, pid_b)}_{max(pid_a, pid_b)}"
                    if pair_key not in mutual_gaze_frames:
                        mutual_gaze_frames[pair_key] = []
                    
                    mutual_gaze_frames[pair_key].append({
                        "frame_idx": ff["frame_idx"],
                        "timestamp": ff["timestamp"],
                        "persons": [pid_a, pid_b],
                    })
    
    # Cluster consecutive mutual gaze frames into events
    for pair_key, frames in mutual_gaze_frames.items():
        if not frames:
            continue
        
        frames = sorted(frames, key=lambda x: x["timestamp"])
        
        clusters = []
        current_cluster = [frames[0]]
        
        for i in range(1, len(frames)):
            # Gap threshold: 1.5s for 2fps video (0.5s/frame + margin for dropped frames)
            if frames[i]["timestamp"] - frames[i-1]["timestamp"] < 1.5:
                current_cluster.append(frames[i])
            else:
                if current_cluster:
                    clusters.append(current_cluster)
                current_cluster = [frames[i]]
        
        if current_cluster:
            clusters.append(current_cluster)
        
        # Create events from clusters
        for cluster in clusters:
            duration = cluster[-1]["timestamp"] - cluster[0]["timestamp"]
            
            if duration >= min_duration_sec:
                # Confidence based on number of frames (bbox containment is binary)
                num_frames = len(cluster)
                confidence = min(1.0, 0.5 + 0.1 * num_frames)
                
                events.append(CandidateEvent(
                    event_id=event_id,
                    event_type="mutual_gaze",
                    start_time=cluster[0]["timestamp"],
                    end_time=cluster[-1]["timestamp"],
                    start_frame=cluster[0]["frame_idx"],
                    end_frame=cluster[-1]["frame_idx"],
                    confidence=confidence,
                    persons_involved=cluster[0]["persons"],
                    details={
                        "duration": duration,
                        "num_frames": num_frames,
                    }
                ))
                event_id += 1
    
    return events


def detect_all_events(features_data: Dict) -> List[CandidateEvent]:
    """Run all event detectors and merge results."""
    frame_features = features_data.get("frame_features", [])
    
    all_events = []
    
    print("Detecting sudden gaze shifts...")
    all_events.extend(detect_sudden_gaze_shifts(frame_features))
    
    print("Detecting joint attention...")
    all_events.extend(detect_joint_attention(frame_features))
    
    print("Detecting gaze following...")
    all_events.extend(detect_gaze_following(frame_features))
    
    print("Detecting attention capture...")
    all_events.extend(detect_attention_capture(frame_features))
    
    print("Detecting mutual gaze...")
    all_events.extend(detect_mutual_gaze(frame_features))
    
    # Sort by start time
    all_events.sort(key=lambda e: e.start_time)
    
    return all_events


def save_events(events: List[CandidateEvent], features_data: Dict, output_path: str) -> None:
    """Save detected events to JSON."""
    output = {
        "video_path": features_data.get("video_path"),
        "video_fps": features_data.get("video_fps"),
        "sample_fps": features_data.get("sample_fps"),
        "num_events": len(events),
        "event_counts": {},
        "events": [],
    }
    
    # Count by type
    for e in events:
        t = e.event_type
        output["event_counts"][t] = output["event_counts"].get(t, 0) + 1
    
    # Convert events to dicts
    for e in events:
        output["events"].append({
            "event_id": e.event_id,
            "event_type": e.event_type,
            "start_time": e.start_time,
            "end_time": e.end_time,
            "start_frame": e.start_frame,
            "end_frame": e.end_frame,
            "confidence": e.confidence,
            "persons_involved": e.persons_involved,
            "details": e.details,
        })
    
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"Saved {len(events)} events to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Detect candidate social gesture events")
    parser.add_argument("--input_features", type=str, default=None, help="Path to single gaze features JSON")
    parser.add_argument("--input_dir", type=str, default=None, help="Directory with *_features.json files for batch processing")
    parser.add_argument("--output_json", type=str, default=None, help="Output path for single file mode")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for batch mode (default: same as input)")
    
    args = parser.parse_args()
    
    if args.input_dir:
        # Batch processing mode
        import glob
        
        output_dir = args.output_dir or args.input_dir
        os.makedirs(output_dir, exist_ok=True)
        
        feature_files = glob.glob(os.path.join(args.input_dir, "*_features.json"))
        feature_files = sorted(set(feature_files))
        
        print(f"Found {len(feature_files)} features JSON files in: {args.input_dir}")
        print(f"Output directory: {output_dir}")
        
        processed, skipped, errors = 0, 0, 0
        total_events = 0
        
        for i, features_path in enumerate(feature_files):
            base_name = os.path.basename(features_path)
            out_name = base_name.replace("_features.json", "_events.json")
            output_path = os.path.join(output_dir, out_name)
            
            # Skip if already processed
            if os.path.exists(output_path):
                print(f"[{i+1}/{len(feature_files)}] {base_name} - SKIP (already exists)")
                skipped += 1
                continue
            
            print(f"[{i+1}/{len(feature_files)}] {base_name}")
            
            try:
                with open(features_path, 'r') as f:
                    features_data = json.load(f)
                
                events = detect_all_events(features_data)
                save_events(events, features_data, output_path)
                total_events += len(events)
                processed += 1
            except Exception as e:
                print(f"  ERROR: {e}")
                errors += 1
        
        print(f"\nDone! Processed: {processed}, Skipped: {skipped}, Errors: {errors}")
        print(f"Total events detected: {total_events}")
    
    elif args.input_features:
        # Single file mode
        if args.output_json is None:
            base = os.path.splitext(args.input_features)[0].replace("_features", "")
            args.output_json = f"{base}_events.json"
        
        print(f"Loading gaze features from: {args.input_features}")
        with open(args.input_features, 'r') as f:
            features_data = json.load(f)
        
        print("Detecting candidate events...")
        events = detect_all_events(features_data)
        
        # Print summary
        print(f"\nDetected {len(events)} candidate events:")
        event_types = {}
        for e in events:
            event_types[e.event_type] = event_types.get(e.event_type, 0) + 1
        
        for t, count in sorted(event_types.items()):
            print(f"  - {t}: {count}")
        
        save_events(events, features_data, args.output_json)
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
