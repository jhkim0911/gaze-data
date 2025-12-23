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


def detect_sudden_gaze_shifts(
    frame_features: List[Dict],
    velocity_threshold: float = 0.5,
    min_duration_sec: float = 0.3,
    max_duration_sec: float = 1.0,
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
            
            # Find all persons involved
            all_persons = set()
            for f in cluster:
                for pid, gaze in f.get("person_gaze_points", {}).items():
                    if gaze is not None:
                        all_persons.add(int(pid))
            
            # Get average convergence center
            centers = [f["center"] for f in cluster if f["center"]]
            if centers:
                avg_center = (
                    np.mean([c[0] for c in centers]),
                    np.mean([c[1] for c in centers])
                )
            else:
                avg_center = None
            
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
                                events.append(CandidateEvent(
                                    event_id=event_id,
                                    event_type="gaze_following",
                                    start_time=t_a,
                                    end_time=t_b,
                                    start_frame=frame_a,
                                    end_frame=frame_b,
                                    confidence=confidence,
                                    persons_involved=[pid_a, pid_b],
                                    details={
                                        "leader_id": pid_a,
                                        "follower_id": pid_b,
                                        "lag_seconds": lag,
                                        "gaze_distance": dist,
                                        "gaze_target": list(gaze_a),
                                    }
                                ))
                                event_id += 1
    
    return events


def detect_attention_capture(
    frame_features: List[Dict],
    velocity_threshold: float = 0.4,
    min_persons: int = 2,
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
    distance_threshold: float = 0.15,  # How close gaze must be to other person's face
    min_duration_sec: float = 0.5,
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
        face_centers = ff.get("person_face_centers", {})
        gaze_points = ff.get("person_gaze_points", {})
        gaze_confidences = ff.get("person_gaze_confidences", {})
        
        # Get all person IDs with valid face_center and gaze_point
        valid_persons = []
        for pid_str in face_centers.keys():
            pid = int(pid_str)
            face_center = face_centers.get(str(pid)) or face_centers.get(pid)
            gaze_point = gaze_points.get(str(pid)) or gaze_points.get(pid)
            gaze_conf = gaze_confidences.get(str(pid), gaze_confidences.get(pid, 0))
            
            if face_center and gaze_point and gaze_conf >= min_confidence:
                valid_persons.append({
                    "pid": pid,
                    "face_center": face_center,
                    "gaze_point": gaze_point,
                })
        
        # Check all pairs for mutual gaze
        for i, person_a in enumerate(valid_persons):
            for person_b in valid_persons[i+1:]:
                pid_a = person_a["pid"]
                pid_b = person_b["pid"]
                
                # Check if A looks at B's face
                dist_a_to_b = np.sqrt(
                    (person_a["gaze_point"][0] - person_b["face_center"][0])**2 +
                    (person_a["gaze_point"][1] - person_b["face_center"][1])**2
                )
                
                # Check if B looks at A's face
                dist_b_to_a = np.sqrt(
                    (person_b["gaze_point"][0] - person_a["face_center"][0])**2 +
                    (person_b["gaze_point"][1] - person_a["face_center"][1])**2
                )
                
                # Both must be looking at each other
                if dist_a_to_b < distance_threshold and dist_b_to_a < distance_threshold:
                    pair_key = f"{min(pid_a, pid_b)}_{max(pid_a, pid_b)}"
                    if pair_key not in mutual_gaze_frames:
                        mutual_gaze_frames[pair_key] = []
                    
                    mutual_gaze_frames[pair_key].append({
                        "frame_idx": ff["frame_idx"],
                        "timestamp": ff["timestamp"],
                        "dist_a_to_b": dist_a_to_b,
                        "dist_b_to_a": dist_b_to_a,
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
            duration = cluster[-1]["timestamp"] - cluster[0]["timestamp"]
            
            if duration >= min_duration_sec:
                mean_dist = np.mean([
                    (f["dist_a_to_b"] + f["dist_b_to_a"]) / 2 
                    for f in cluster
                ])
                confidence = max(0, 1.0 - (mean_dist / distance_threshold))
                
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
                        "mean_gaze_distance": mean_dist,
                        "num_frames": len(cluster),
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
    parser.add_argument("--input_features", type=str, required=True, help="Path to gaze features JSON")
    parser.add_argument("--output_json", type=str, default=None, help="Output path (default: input_events.json)")
    
    args = parser.parse_args()
    
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


if __name__ == "__main__":
    main()
