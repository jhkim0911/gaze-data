#!/usr/bin/env python3
"""
Gaze Feature Extractor for Social Gesture Detection

Extracts temporal gaze features from gaze annotation JSON files to enable
detection of social gesture candidate events.

Features extracted:
- Gaze velocity per person (sudden shifts)
- Pairwise gaze distance matrix (who looks at whom)
- Gaze convergence score (multiple people → same region)
- Per-person gaze statistics

Usage:
    python gaze_feature_extractor.py --input_json /path/to/gaze.json --output_json /path/to/features.json
"""

import argparse
import json
import math
import os
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
import numpy as np


@dataclass
class PersonGazeFeatures:
    """Gaze features for a single person across the video."""
    person_id: int
    track_length: int
    face_detection_pct: float
    inframe_gaze_pct: float
    
    # Velocity statistics
    mean_velocity: float  # Average gaze velocity (normalized units/sec)
    max_velocity: float   # Maximum gaze velocity
    velocity_std: float   # Velocity standard deviation
    
    # Gaze point statistics
    mean_gaze_x: float
    mean_gaze_y: float
    gaze_spread: float  # Spatial spread of gaze points


@dataclass
class FrameGazeFeatures:
    """Gaze features for a single frame."""
    frame_idx: int
    timestamp: float
    num_persons: int
    num_faces_detected: int
    
    # Per-person features (existing)
    person_velocities: Dict[int, float]  # person_id -> velocity at this frame (RAW gaze point velocity)
    person_gaze_points: Dict[int, Optional[Tuple[float, float]]]  # normalized gaze
    person_gaze_confidences: Dict[int, float]  # person_id -> gaze confidence (1.0=measured, <1.0=interpolated)
    person_gaze_methods: Dict[int, str]  # person_id -> method (measured, linear_interpolation, carry_forward, etc.)
    
    # NEW: Anchor-based features (physics-based)
    person_face_centers: Dict[int, Optional[Tuple[float, float]]]  # Anchor positions (face bbox center)
    person_face_bboxes: Dict[int, Optional[List[float]]]  # Face bounding boxes [x1,y1,x2,y2] for mutual gaze
    person_gaze_directions: Dict[int, Optional[Tuple[float, float]]]  # Gaze direction relative to face
    person_face_velocities: Dict[int, Optional[float]]  # How fast face/person moved (for scene cut detection)
    person_is_scene_cut: Dict[int, bool]  # Did scene cut occur for this person?
    person_gaze_dir_velocities: Dict[int, Optional[float]]  # Gaze DIRECTION velocity (physics-correct)
    
    # Social features
    gaze_convergence_score: float  # How clustered are gaze points
    gaze_convergence_center: Optional[Tuple[float, float]]  # Center of convergence
    pairwise_distances: Dict[str, float]  # "p0_p1" -> distance between gaze points


@dataclass
class GazeFeaturesResult:
    """Complete gaze features for a video."""
    video_path: str
    video_fps: float
    sample_fps: float
    total_frames: int
    processed_frames: int
    num_persons: int
    
    # Per-person features
    person_features: Dict[int, PersonGazeFeatures]
    
    # Frame-level features
    frame_features: List[FrameGazeFeatures]
    
    # Video-level statistics
    mean_convergence_score: float
    max_convergence_score: float
    high_velocity_events: List[Dict]  # List of sudden gaze shift events
    high_convergence_events: List[Dict]  # List of gaze convergence events


def compute_gaze_distance(gaze1: Tuple[float, float], gaze2: Tuple[float, float]) -> float:
    """Compute Euclidean distance between two gaze points (normalized)."""
    return math.sqrt((gaze1[0] - gaze2[0])**2 + (gaze1[1] - gaze2[1])**2)


def compute_gaze_velocity(
    gaze_prev: Tuple[float, float],
    gaze_curr: Tuple[float, float],
    dt: float
) -> float:
    """Compute gaze velocity between two frames."""
    if dt <= 0:
        return 0.0
    distance = compute_gaze_distance(gaze_prev, gaze_curr)
    return distance / dt


def compute_convergence_score(gaze_points: List[Tuple[float, float]]) -> Tuple[float, Optional[Tuple[float, float]]]:
    """
    Compute how clustered the gaze points are.
    
    Uses MEDIAN distance (instead of mean) for robustness to outliers.
    
    Returns:
        Tuple of (score, center) where:
        - score: 0.0 = maximally spread, 1.0 = all looking at same point
        - center: centroid of gaze points (if any)
    """
    if len(gaze_points) < 2:
        return 0.0, gaze_points[0] if gaze_points else None
    
    # Compute centroid
    cx = sum(p[0] for p in gaze_points) / len(gaze_points)
    cy = sum(p[1] for p in gaze_points) / len(gaze_points)
    center = (cx, cy)
    
    # Compute MEDIAN distance from centroid (robust to outliers)
    distances = [compute_gaze_distance(p, center) for p in gaze_points]
    median_dist = float(np.median(distances))
    
    # Convert to score (smaller distance = higher convergence)
    # Max possible distance in normalized space is sqrt(2) ≈ 1.414
    # Use exponential decay: score = exp(-k * median_dist)
    score = math.exp(-3.0 * median_dist)
    
    return score, center


# ============================================================================
# Anchor-based Gaze Features (Physics-correct)
# ============================================================================

# Scene cut detection threshold (30% of frame = likely scene cut or camera change)
SCENE_CUT_THRESHOLD = 0.3

# Maximum frame gap for computing velocity (gaps > 1 frame may indicate scene change)
MAX_VELOCITY_FRAME_GAP = 1


def compute_bbox_center(bbox: Tuple[float, float, float, float]) -> Tuple[float, float]:
    """
    Compute center of normalized bbox (x1, y1, x2, y2).
    
    Args:
        bbox: (x1, y1, x2, y2) in normalized coordinates [0, 1]
    
    Returns:
        (center_x, center_y) in normalized coordinates
    """
    return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)


def compute_gaze_direction(
    gaze_point: Tuple[float, float], 
    face_center: Tuple[float, float]
) -> Tuple[float, float]:
    """
    Compute gaze direction vector from face to gaze point.
    
    This is the "relative gaze" that tells where the person is looking
    independent of their position in the frame.
    
    Args:
        gaze_point: Where the person is looking (normalized)
        face_center: Where the person's face is (normalized)
    
    Returns:
        (dx, dy) direction vector from face to gaze point
    """
    return (gaze_point[0] - face_center[0], gaze_point[1] - face_center[1])


def is_scene_cut(
    face_center_prev: Tuple[float, float],
    face_center_curr: Tuple[float, float],
    threshold: float = SCENE_CUT_THRESHOLD
) -> bool:
    """
    Detect if face jumped (scene cut, camera change, or person teleported).
    
    In continuous video, a person's face doesn't move more than ~10-20% of frame
    between adjacent frames. Larger jumps indicate discontinuity.
    
    Args:
        face_center_prev: Face center in previous frame
        face_center_curr: Face center in current frame
        threshold: Maximum displacement before flagging as scene cut
    
    Returns:
        True if displacement exceeds threshold (likely scene cut)
    """
    displacement = math.sqrt(
        (face_center_curr[0] - face_center_prev[0])**2 +
        (face_center_curr[1] - face_center_prev[1])**2
    )
    return displacement > threshold


# ============================================================================
# Gaze Interpolation with Confidence
# ============================================================================

# Interpolation thresholds (tuned from data analysis)
INTERPOLATE_MAX_GAP = 3      # Linear interpolation for gaps ≤ 3 frames
CARRY_FORWARD_MAX_GAP = 10   # Carry forward for gaps 4-10 frames
MAX_TIME_GAP_SEC = 3.0       # Never interpolate across gaps > 3 seconds (scene cuts) - 6 frames at 2fps
# Gaps > 10 frames or > 3 seconds: Keep as null


def interpolate_gaze_for_person(
    gaze_history: List[Tuple[float, Optional[Tuple[float, float]]]]
) -> List[Dict]:
    """
    Interpolate missing gaze points for a single person.
    
    3-tier strategy:
    - Gap ≤ 3 frames: Linear interpolation, confidence = 1.0 - 0.1 * gap
    - Gap 4-10 frames: Carry forward, confidence = 0.5 * exp(-0.2 * gap_position)
    - Gap > 10 frames: Keep as null, confidence = 0.0
    
    Args:
        gaze_history: List of (timestamp, gaze_point or None)
        
    Returns:
        List of dicts with interpolated gaze data
    """
    n = len(gaze_history)
    if n == 0:
        return []
    
    result = []
    
    # First pass: identify all valid gaze indices
    valid_indices = [i for i, (_, g) in enumerate(gaze_history) if g is not None]
    
    for i, (timestamp, gaze) in enumerate(gaze_history):
        if gaze is not None:
            # Original valid gaze
            result.append({
                "timestamp": timestamp,
                "gaze_point": gaze,
                "gaze_confidence": 1.0,
                "gaze_method": "measured",
                "original_inout": True,
            })
        else:
            # Find nearest valid gaze before and after
            prev_valid = None
            next_valid = None
            
            for vi in valid_indices:
                if vi < i:
                    prev_valid = vi
                elif vi > i and next_valid is None:
                    next_valid = vi
                    break
            
            # Determine interpolation method
            if prev_valid is not None and next_valid is not None:
                gap = next_valid - prev_valid - 1
                position_in_gap = i - prev_valid
                
                # Check timestamp gap - don't interpolate across scene cuts
                time_gap = gaze_history[next_valid][0] - gaze_history[prev_valid][0]
                if time_gap > MAX_TIME_GAP_SEC:
                    # Scene cut detected - don't interpolate
                    result.append({
                        "timestamp": timestamp,
                        "gaze_point": None,
                        "gaze_confidence": 0.0,
                        "gaze_method": "null_scene_cut",
                        "original_inout": False,
                    })
                elif gap <= INTERPOLATE_MAX_GAP:
                    # Tier 1: Linear interpolation
                    t = position_in_gap / (gap + 1)
                    prev_gaze = gaze_history[prev_valid][1]
                    next_gaze = gaze_history[next_valid][1]
                    
                    interp_x = prev_gaze[0] + t * (next_gaze[0] - prev_gaze[0])
                    interp_y = prev_gaze[1] + t * (next_gaze[1] - prev_gaze[1])
                    confidence = 1.0 - (0.1 * gap)
                    
                    result.append({
                        "timestamp": timestamp,
                        "gaze_point": (interp_x, interp_y),
                        "gaze_confidence": confidence,
                        "gaze_method": "linear_interpolation",
                        "original_inout": False,
                    })
                elif gap <= CARRY_FORWARD_MAX_GAP:
                    # Tier 2: Carry forward with decay
                    prev_gaze = gaze_history[prev_valid][1]
                    confidence = 0.5 * math.exp(-0.2 * position_in_gap)
                    
                    result.append({
                        "timestamp": timestamp,
                        "gaze_point": prev_gaze,
                        "gaze_confidence": confidence,
                        "gaze_method": "carry_forward",
                        "original_inout": False,
                    })
                else:
                    # Tier 3: Gap too long
                    result.append({
                        "timestamp": timestamp,
                        "gaze_point": None,
                        "gaze_confidence": 0.0,
                        "gaze_method": "null_gap_too_long",
                        "original_inout": False,
                    })
            elif prev_valid is not None:
                # Only have previous, carry forward if close enough
                gap = i - prev_valid
                # Check timestamp gap for scene cuts
                time_gap = gaze_history[i][0] - gaze_history[prev_valid][0]
                if time_gap > MAX_TIME_GAP_SEC:
                    # Scene cut - don't carry forward
                    result.append({
                        "timestamp": timestamp,
                        "gaze_point": None,
                        "gaze_confidence": 0.0,
                        "gaze_method": "null_scene_cut",
                        "original_inout": False,
                    })
                elif gap <= CARRY_FORWARD_MAX_GAP:
                    prev_gaze = gaze_history[prev_valid][1]
                    confidence = 0.5 * math.exp(-0.2 * gap)
                    result.append({
                        "timestamp": timestamp,
                        "gaze_point": prev_gaze,
                        "gaze_confidence": confidence,
                        "gaze_method": "carry_forward",
                        "original_inout": False,
                    })
                else:
                    result.append({
                        "timestamp": timestamp,
                        "gaze_point": None,
                        "gaze_confidence": 0.0,
                        "gaze_method": "null_no_valid",
                        "original_inout": False,
                    })
            elif next_valid is not None:
                # Only have next, carry backward if close enough
                gap = next_valid - i
                # Check timestamp gap for scene cuts
                time_gap = gaze_history[next_valid][0] - gaze_history[i][0]
                if time_gap > MAX_TIME_GAP_SEC:
                    # Scene cut - don't carry backward
                    result.append({
                        "timestamp": timestamp,
                        "gaze_point": None,
                        "gaze_confidence": 0.0,
                        "gaze_method": "null_scene_cut",
                        "original_inout": False,
                    })
                elif gap <= CARRY_FORWARD_MAX_GAP:
                    next_gaze = gaze_history[next_valid][1]
                    confidence = 0.5 * math.exp(-0.2 * gap)
                    result.append({
                        "timestamp": timestamp,
                        "gaze_point": next_gaze,
                        "gaze_confidence": confidence,
                        "gaze_method": "carry_backward",
                        "original_inout": False,
                    })
                else:
                    result.append({
                        "timestamp": timestamp,
                        "gaze_point": None,
                        "gaze_confidence": 0.0,
                        "gaze_method": "null_no_valid",
                        "original_inout": False,
                    })
            else:
                # No valid gaze at all
                result.append({
                    "timestamp": timestamp,
                    "gaze_point": None,
                    "gaze_confidence": 0.0,
                    "gaze_method": "null_no_valid",
                    "original_inout": False,
                })
    
    return result


def compute_weighted_convergence_score(
    gaze_points: List[Tuple[float, float]],
    confidences: List[float],
    min_confidence: float = 0.3
) -> Tuple[float, Optional[Tuple[float, float]]]:
    """
    Compute convergence score with confidence weighting.
    
    Uses MEDIAN distance for robustness to outliers.
    Only considers gaze points with confidence >= min_confidence.
    Uses confidence as weight for centroid calculation.
    """
    # Filter by minimum confidence
    valid = [(g, c) for g, c in zip(gaze_points, confidences) if g is not None and c >= min_confidence]
    
    if len(valid) < 2:
        return 0.0, valid[0][0] if valid else None
    
    # Weighted centroid
    total_weight = sum(c for _, c in valid)
    cx = sum(g[0] * c for g, c in valid) / total_weight
    cy = sum(g[1] * c for g, c in valid) / total_weight
    center = (cx, cy)
    
    # MEDIAN distance (robust to outliers)
    distances = [compute_gaze_distance(g, center) for g, c in valid]
    median_dist = float(np.median(distances))
    
    # Score with confidence penalty
    avg_confidence = total_weight / len(valid)
    score = math.exp(-3.0 * median_dist) * avg_confidence
    
    return score, center


def extract_gaze_features(gaze_data: Dict) -> GazeFeaturesResult:
    """
    Extract gaze features from a gaze annotation JSON.
    
    Args:
        gaze_data: Loaded gaze annotation JSON
        
    Returns:
        GazeFeaturesResult with all extracted features
    """
    video_path = gaze_data["video_path"]
    video_fps = gaze_data["video_fps"]
    sample_fps = gaze_data.get("sample_fps", video_fps)
    total_frames = gaze_data["total_frames"]
    processed_frames = gaze_data["processed_frames"]
    
    persons_summary = gaze_data["persons_summary"]
    num_persons = len(persons_summary)
    
    frames = gaze_data["frames"]
    
    # Build per-person gaze history (raw, before interpolation)
    person_gaze_history_raw: Dict[int, List[Tuple[float, Optional[Tuple[float, float]]]]] = {}
    for pid in persons_summary.keys():
        person_gaze_history_raw[int(pid)] = []
    
    # Process all frames to build history
    for frame_data in frames:
        timestamp = frame_data["timestamp"]
        for person in frame_data["persons"]:
            pid = person["person_id"]
            gaze = None
            if person.get("gaze_point") and person.get("inout") is True:
                gaze = tuple(person["gaze_point"])
            
            if pid not in person_gaze_history_raw:
                person_gaze_history_raw[pid] = []
            person_gaze_history_raw[pid].append((timestamp, gaze))
    
    # Apply interpolation to each person's gaze history
    person_gaze_interpolated: Dict[int, List[Dict]] = {}
    for pid, raw_history in person_gaze_history_raw.items():
        person_gaze_interpolated[pid] = interpolate_gaze_for_person(raw_history)
    
    # Compute per-person velocity features (using interpolated gaze)
    person_velocities_over_time: Dict[int, List[Tuple[float, float, float]]] = {}  # pid -> [(timestamp, velocity, confidence)]
    
    for pid, interp_history in person_gaze_interpolated.items():
        velocities = []
        for i in range(1, len(interp_history)):
            prev = interp_history[i-1]
            curr = interp_history[i]
            dt = curr["timestamp"] - prev["timestamp"]
            
            gaze_prev = prev["gaze_point"]
            gaze_curr = curr["gaze_point"]
            conf_prev = prev["gaze_confidence"]
            conf_curr = curr["gaze_confidence"]
            
            if gaze_prev is not None and gaze_curr is not None and dt > 0:
                vel = compute_gaze_velocity(gaze_prev, gaze_curr, dt)
                # Velocity confidence = min of both endpoint confidences
                vel_conf = min(conf_prev, conf_curr)
                velocities.append((curr["timestamp"], vel, vel_conf))
            else:
                velocities.append((curr["timestamp"], 0.0, 0.0))
        
        person_velocities_over_time[pid] = velocities
    
    # Compute per-person aggregate features
    person_features: Dict[int, PersonGazeFeatures] = {}
    
    for pid_str, summary in persons_summary.items():
        pid = int(pid_str)
        interp_history = person_gaze_interpolated.get(pid, [])
        velocities = [v for _, v, _ in person_velocities_over_time.get(pid, [])]
        
        # Gaze point statistics (using interpolated gaze with confidence > 0.5)
        valid_gaze_points = [
            d["gaze_point"] for d in interp_history 
            if d["gaze_point"] is not None and d["gaze_confidence"] > 0.5
        ]
        
        if valid_gaze_points:
            mean_x = sum(p[0] for p in valid_gaze_points) / len(valid_gaze_points)
            mean_y = sum(p[1] for p in valid_gaze_points) / len(valid_gaze_points)
            spread = np.std([compute_gaze_distance(p, (mean_x, mean_y)) for p in valid_gaze_points])
        else:
            mean_x, mean_y, spread = 0.0, 0.0, 0.0
        
        person_features[pid] = PersonGazeFeatures(
            person_id=pid,
            track_length=summary["track_length"],
            face_detection_pct=summary["face_detection_pct"],
            inframe_gaze_pct=summary["inframe_gaze_pct"],
            mean_velocity=float(np.mean(velocities)) if velocities else 0.0,
            max_velocity=float(np.max(velocities)) if velocities else 0.0,
            velocity_std=float(np.std(velocities)) if velocities else 0.0,
            mean_gaze_x=mean_x,
            mean_gaze_y=mean_y,
            gaze_spread=float(spread),
        )
    
    # Compute frame-level features
    frame_features: List[FrameGazeFeatures] = []
    frame_idx_to_velocity_idx: Dict[int, int] = {}  # Map frame index to velocity array index
    
    # NEW: Build per-person anchor history (face_center per frame) for physics-based velocity
    person_anchor_history: Dict[int, List[Dict]] = {}  # pid -> [{frame_idx, timestamp, face_center, gaze_point, gaze_direction}, ...]
    
    for idx, frame_data in enumerate(frames):
        frame_idx = frame_data["frame_idx"]
        timestamp = frame_data["timestamp"]
        
        for person in frame_data["persons"]:
            pid = person["person_id"]
            
            if pid not in person_anchor_history:
                person_anchor_history[pid] = []
            
            # Extract face center (anchor) if face detected
            face_center = None
            if person.get("face_detected") and person.get("face_bbox"):
                face_bbox = person["face_bbox"]
                face_center = compute_bbox_center(tuple(face_bbox))
            
            # Get gaze point
            gaze_point = None
            interp_history = person_gaze_interpolated.get(pid, [])
            if idx < len(interp_history):
                interp_data = interp_history[idx]
                gaze_point = interp_data["gaze_point"]
            
            # Compute gaze direction (relative to face) if both available
            gaze_direction = None
            if face_center and gaze_point:
                gaze_direction = compute_gaze_direction(gaze_point, face_center)
            
            person_anchor_history[pid].append({
                "frame_list_idx": idx,
                "frame_idx": frame_idx,
                "timestamp": timestamp,
                "face_center": face_center,
                "face_bbox": person.get("face_bbox"),  # [x1, y1, x2, y2] for mutual gaze
                "gaze_point": gaze_point,
                "gaze_direction": gaze_direction,
            })
    
    # Compute anchor-based velocities for each person
    person_anchor_velocities: Dict[int, List[Dict]] = {}  # pid -> [{frame_idx, face_vel, gaze_dir_vel, is_scene_cut}, ...]
    
    for pid, history in person_anchor_history.items():
        velocities = []
        for i in range(len(history)):
            curr = history[i]
            result = {
                "frame_idx": curr["frame_idx"],
                "face_velocity": None,
                "gaze_dir_velocity": None,
                "is_scene_cut": False,
            }
            
            if i == 0:
                # First frame - no velocity
                velocities.append(result)
                continue
            
            prev = history[i - 1]
            dt = curr["timestamp"] - prev["timestamp"]
            
            # Check if we can compute velocity
            if prev["face_center"] is None or curr["face_center"] is None:
                # Missing anchor - can't compute
                velocities.append(result)
                continue
            
            # Compute face displacement
            face_disp = compute_gaze_distance(prev["face_center"], curr["face_center"])
            
            # Check for scene cut
            if is_scene_cut(prev["face_center"], curr["face_center"]):
                result["is_scene_cut"] = True
                velocities.append(result)
                continue
            
            # Compute face velocity
            result["face_velocity"] = face_disp / dt if dt > 0 else 0.0
            result["is_scene_cut"] = False
            
            # Compute gaze direction velocity (physics-correct!)
            if prev["gaze_direction"] and curr["gaze_direction"]:
                gaze_dir_disp = compute_gaze_distance(prev["gaze_direction"], curr["gaze_direction"])
                result["gaze_dir_velocity"] = gaze_dir_disp / dt if dt > 0 else 0.0
            
            velocities.append(result)
        
        person_anchor_velocities[pid] = velocities
    
    # Now build frame features with all the new data
    for idx, frame_data in enumerate(frames):
        frame_idx = frame_data["frame_idx"]
        timestamp = frame_data["timestamp"]
        frame_idx_to_velocity_idx[frame_idx] = idx
        
        persons = frame_data["persons"]
        num_faces = sum(1 for p in persons if p.get("face_detected"))
        
        # Get interpolated gaze points, confidences, and velocities for this frame
        person_gaze_points = {}
        person_gaze_confidences = {}
        person_gaze_methods = {}
        person_velocities_frame = {}
        
        # NEW: Anchor-based features
        person_face_centers = {}
        person_face_bboxes = {}  # For mutual gaze bbox containment check
        person_gaze_directions = {}
        person_face_velocities = {}
        person_is_scene_cut = {}
        person_gaze_dir_velocities = {}
        
        for person in persons:
            pid = person["person_id"]
            
            # IMPORTANT: Only include gaze data for persons with face_detected=True
            # Skip interpolated gaze for persons without face bbox (not actually in frame)
            if not person.get("face_detected"):
                # Still need to add to dicts with None for consistency, but mark as not present
                person_gaze_points[pid] = None
                person_gaze_confidences[pid] = 0.0
                person_gaze_methods[pid] = "no_face"
            elif person.get("gaze_point") and person.get("inout") is True:
                # Valid face detected AND valid gaze point
                person_gaze_points[pid] = tuple(person["gaze_point"])
                person_gaze_confidences[pid] = 1.0
                person_gaze_methods[pid] = "measured"
            else:
                # Face detected but no valid gaze - try interpolation for temporary loss
                interp_history = person_gaze_interpolated.get(pid, [])
                found = False
                for idata in interp_history:
                    if abs(idata.get("timestamp", 0) - timestamp) < 0.01:
                        person_gaze_points[pid] = idata["gaze_point"]
                        person_gaze_confidences[pid] = idata["gaze_confidence"]
                        person_gaze_methods[pid] = idata["gaze_method"]
                        found = True
                        break
                if not found:
                    person_gaze_points[pid] = None
                    person_gaze_confidences[pid] = 0.0
                    person_gaze_methods[pid] = "missing"
            
            # Get RAW velocity at this frame (original method)
            vel_history = person_velocities_over_time.get(pid, [])
            if idx > 0 and idx - 1 < len(vel_history):
                person_velocities_frame[pid] = vel_history[idx - 1][1]  # [1] is velocity
            else:
                person_velocities_frame[pid] = 0.0
            
            # Get anchor-based features
            anchor_history = person_anchor_history.get(pid, [])
            anchor_velocities = person_anchor_velocities.get(pid, [])
            
            # Get face_bbox directly from current frame's person data (not from anchor history)
            # because anchor_history index doesn't match frame list index when persons enter/exit
            person_face_bboxes[pid] = person.get("face_bbox") if person.get("face_detected") else None
            
            # For face Center and gaze direction, look up by frame_idx in anchor history
            anchor_entry = None
            for ah in anchor_history:
                if ah.get("frame_idx") == frame_idx:
                    anchor_entry = ah
                    break
            
            if anchor_entry:
                person_face_centers[pid] = anchor_entry["face_center"]
                person_gaze_directions[pid] = anchor_entry["gaze_direction"]
            else:
                person_face_centers[pid] = None
                person_gaze_directions[pid] = None
            
            # For velocities, also look up by frame_idx
            vel_entry = None
            for av in anchor_velocities:
                if av.get("frame_idx") == frame_idx:
                    vel_entry = av
                    break
            
            if vel_entry:
                person_face_velocities[pid] = vel_entry["face_velocity"]
                person_is_scene_cut[pid] = vel_entry["is_scene_cut"]
                person_gaze_dir_velocities[pid] = vel_entry["gaze_dir_velocity"]
            else:
                person_face_velocities[pid] = None
                person_is_scene_cut[pid] = False
                person_gaze_dir_velocities[pid] = None
        
        # Compute weighted convergence (uses confidence)
        # IMPORTANT: Only include persons with valid face_bbox (actually present in frame)
        # Don't use interpolated gaze from absent persons
        present_pids = [pid for pid in person_gaze_points 
                        if person_face_bboxes.get(pid) is not None]
        gaze_list = [person_gaze_points[pid] for pid in present_pids]
        conf_list = [person_gaze_confidences[pid] for pid in present_pids]
        conv_score, conv_center = compute_weighted_convergence_score(gaze_list, conf_list)
        
        # Compute pairwise distances (only for persons actually present with good confidence)
        pairwise_distances = {}
        # Only use persons with valid face_bbox
        present_person_ids = [p["person_id"] for p in persons 
                              if person_face_bboxes.get(p["person_id"]) is not None]
        for i, pid1 in enumerate(present_person_ids):
            for pid2 in present_person_ids[i+1:]:
                g1 = person_gaze_points.get(pid1)
                g2 = person_gaze_points.get(pid2)
                c1 = person_gaze_confidences.get(pid1, 0)
                c2 = person_gaze_confidences.get(pid2, 0)
                if g1 is not None and g2 is not None and c1 >= 0.3 and c2 >= 0.3:
                    dist = compute_gaze_distance(g1, g2)
                    pairwise_distances[f"p{pid1}_p{pid2}"] = dist
        
        frame_features.append(FrameGazeFeatures(
            frame_idx=frame_idx,
            timestamp=timestamp,
            num_persons=len(persons),
            num_faces_detected=num_faces,
            person_velocities=person_velocities_frame,
            person_gaze_points=person_gaze_points,
            person_gaze_confidences=person_gaze_confidences,
            person_gaze_methods=person_gaze_methods,
            # NEW fields:
            person_face_centers=person_face_centers,
            person_face_bboxes=person_face_bboxes,  # For mutual gaze bbox check
            person_gaze_directions=person_gaze_directions,
            person_face_velocities=person_face_velocities,
            person_is_scene_cut=person_is_scene_cut,
            person_gaze_dir_velocities=person_gaze_dir_velocities,
            # Social features:
            gaze_convergence_score=conv_score,
            gaze_convergence_center=conv_center,
            pairwise_distances=pairwise_distances,
        ))
    
    # Detect high-velocity events (sudden gaze shifts)
    VELOCITY_THRESHOLD = 0.5  # Normalized units per second
    MIN_VELOCITY_CONFIDENCE = 0.5  # Only count high-confidence velocity measurements
    high_velocity_events = []
    
    for pid, vel_history in person_velocities_over_time.items():
        for t, vel, vel_conf in vel_history:
            if vel > VELOCITY_THRESHOLD and vel_conf >= MIN_VELOCITY_CONFIDENCE:
                high_velocity_events.append({
                    "timestamp": t,
                    "person_id": pid,
                    "velocity": vel,
                    "confidence": vel_conf,
                })
    
    # Sort by velocity (descending)
    high_velocity_events.sort(key=lambda x: x["velocity"], reverse=True)
    
    # Detect high-convergence events
    CONVERGENCE_THRESHOLD = 0.7
    high_convergence_events = []
    
    for ff in frame_features:
        if ff.gaze_convergence_score > CONVERGENCE_THRESHOLD and ff.num_faces_detected >= 2:
            high_convergence_events.append({
                "timestamp": ff.timestamp,
                "frame_idx": ff.frame_idx,
                "convergence_score": ff.gaze_convergence_score,
                "center": ff.gaze_convergence_center,
                "num_persons_looking": ff.num_faces_detected,
            })
    
    # Video-level statistics
    convergence_scores = [ff.gaze_convergence_score for ff in frame_features if ff.num_faces_detected >= 2]
    mean_convergence = float(np.mean(convergence_scores)) if convergence_scores else 0.0
    max_convergence = float(np.max(convergence_scores)) if convergence_scores else 0.0
    
    return GazeFeaturesResult(
        video_path=video_path,
        video_fps=video_fps,
        sample_fps=sample_fps,
        total_frames=total_frames,
        processed_frames=processed_frames,
        num_persons=num_persons,
        person_features=person_features,
        frame_features=frame_features,
        mean_convergence_score=mean_convergence,
        max_convergence_score=max_convergence,
        high_velocity_events=high_velocity_events[:50],  # Top 50
        high_convergence_events=high_convergence_events,
    )


def save_features(result: GazeFeaturesResult, output_path: str) -> None:
    """Save extracted features to JSON file."""
    
    def to_dict(obj):
        if hasattr(obj, '__dict__'):
            return {k: to_dict(v) for k, v in obj.__dict__.items()}
        elif isinstance(obj, dict):
            return {k: to_dict(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [to_dict(v) for v in obj]
        elif isinstance(obj, tuple):
            return list(obj)
        elif isinstance(obj, (np.floating, np.integer)):
            return float(obj) if isinstance(obj, np.floating) else int(obj)
        else:
            return obj
    
    output_data = to_dict(result)
    
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"Saved gaze features to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Extract gaze features for social gesture detection")
    parser.add_argument("--input_json", type=str, default=None, help="Path to single gaze annotation JSON")
    parser.add_argument("--input_dir", type=str, default=None, help="Directory with *_gaze.json files for batch processing")
    parser.add_argument("--output_json", type=str, default=None, help="Output path for single file mode")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for batch mode (default: same as input)")
    
    args = parser.parse_args()
    
    if args.input_dir:
        # Batch processing mode
        import glob
        
        output_dir = args.output_dir or args.input_dir
        os.makedirs(output_dir, exist_ok=True)
        
        gaze_files = glob.glob(os.path.join(args.input_dir, "*.json"))
        # Exclude output files (features, events, gestures, qa)
        gaze_files = [f for f in gaze_files if not any(
            f.endswith(suffix) for suffix in ["_features.json", "_events.json", "_gestures.json", "_qa_dataset.json"]
        )]
        gaze_files = sorted(set(gaze_files))
        
        print(f"Found {len(gaze_files)} gaze JSON files in: {args.input_dir}")
        print(f"Output directory: {output_dir}")
        
        processed, skipped, errors = 0, 0, 0
        
        for i, gaze_path in enumerate(gaze_files):
            base_name = os.path.basename(gaze_path)
            out_name = base_name.replace(".json", "_features.json")
            output_path = os.path.join(output_dir, out_name)
            
            # Skip if already processed
            if os.path.exists(output_path):
                print(f"[{i+1}/{len(gaze_files)}] {base_name} - SKIP (already exists)")
                skipped += 1
                continue
            
            print(f"[{i+1}/{len(gaze_files)}] {base_name}")
            
            try:
                with open(gaze_path, 'r') as f:
                    gaze_data = json.load(f)
                
                result = extract_gaze_features(gaze_data)
                save_features(result, output_path)
                processed += 1
            except Exception as e:
                print(f"  ERROR: {e}")
                errors += 1
        
        print(f"\nDone! Processed: {processed}, Skipped: {skipped}, Errors: {errors}")
    
    elif args.input_json:
        # Single file mode
        if args.output_json is None:
            base = os.path.splitext(args.input_json)[0]
            args.output_json = f"{base}_features.json"
        
        print(f"Loading gaze data from: {args.input_json}")
        with open(args.input_json, 'r') as f:
            gaze_data = json.load(f)
        
        print("Extracting gaze features...")
        result = extract_gaze_features(gaze_data)
        
        print(f"  - {result.num_persons} persons")
        print(f"  - {result.processed_frames} frames")
        print(f"  - Mean convergence: {result.mean_convergence_score:.3f}")
        print(f"  - High velocity events: {len(result.high_velocity_events)}")
        print(f"  - High convergence events: {len(result.high_convergence_events)}")
        
        save_features(result, args.output_json)
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
