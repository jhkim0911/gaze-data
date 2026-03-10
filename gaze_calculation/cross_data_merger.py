#!/usr/bin/env python3
"""
Cross-Data Merger - Merge gaze events and gestures into a single time-sorted JSON per video.

Applies the same preprocessing as the graph generators (confidence filtering,
gesture timestamp snapping, duration filtering) but without Gemini calls or
temporal relation computation.

Output: {dataset}/cross_data/{video_name}_cross_data.json

Usage:
    # Single dataset
    python -m gaze_calculation.cross_data_merger --dataset social_gesture

    # All datasets
    python -m gaze_calculation.cross_data_merger --all

    # Sample 10 files
    python -m gaze_calculation.cross_data_merger --dataset social_gesture --max_files 10
"""

import argparse
import glob
import json
import math
import os
import re
import subprocess
import sys
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

sys.stdout.reconfigure(line_buffering=True)

DATASETS = [
    'avsbench', 'embody3d', 'friendsmmc', 'social_gesture',
    'social-iq', 'tvqa', 'werewolf',
]
DEFAULT_BASE_PATH = "/projects/illinois/eng/cs/jrehg/users/arkimjh/gaze_social"

GAZE_CONFIDENCE_THRESHOLD = 0.9
GESTURE_CONFIDENCE_THRESHOLD = 0.85


def _parse_timestamp(t) -> float:
    """Convert timestamp to float seconds. Handles float, str('19.5'), str('0:45.50'), str('1:02:30')."""
    if isinstance(t, (int, float)):
        return float(t)
    s = str(t).strip()
    if ':' in s:
        parts = s.split(':')
        secs = 0.0
        for p in parts:
            secs = secs * 60 + float(p)
        return secs
    return float(s)


def _snap_to_half(t, ceil: bool = False) -> float:
    """Snap timestamp to nearest 0.5 boundary (floor or ceil)."""
    try:
        t = float(t)
    except (ValueError, TypeError):
        t = _parse_timestamp(t)
    if ceil:
        return math.ceil(t * 2) / 2
    return math.floor(t * 2) / 2


def _get_video_duration(video_path: str) -> Optional[float]:
    """Get video duration in seconds via ffprobe."""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
             "-of", "csv=p=0", video_path],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            return float(result.stdout.strip())
    except (subprocess.TimeoutExpired, ValueError):
        pass
    return None


def _resolve_video_path(video_name: str, base_dir: str, dataset_name: str) -> Optional[str]:
    """Resolve video_name → nbbox_videos/{name}_bbox_viz.mp4."""
    video_path = os.path.join(base_dir, dataset_name, "nbbox_videos", f"{video_name}_bbox_viz.mp4")
    if os.path.exists(video_path):
        return video_path
    return None


def load_gaze_events(events_json_path: str) -> List[dict]:
    """Load gaze events, filter by confidence, add source tag."""
    with open(events_json_path, 'r') as f:
        raw = json.load(f)

    events = []
    for ev in raw.get("events", []):
        if ev.get("confidence", 0) < GAZE_CONFIDENCE_THRESHOLD:
            continue
        entry = {
            "source": "gaze",
            "event_id": ev.get("event_id"),
            "event_type": ev.get("event_type"),
            "start_time": ev.get("start_time", 0),
            "end_time": ev.get("end_time", 0),
            "confidence": ev.get("confidence"),
            "persons_involved": ev.get("persons_involved", []),
        }
        # Preserve social details for downstream description generation
        details = ev.get("details", {})
        etype = ev.get("event_type", "")
        if etype == "gaze_following":
            entry["leader_id"] = details.get("leader_id")
            entry["follower_id"] = details.get("follower_id")
            if details.get("lag_seconds"):
                entry["lag_seconds"] = details["lag_seconds"]
        elif etype == "mutual_gaze":
            if details.get("duration"):
                entry["gaze_duration"] = details["duration"]
        elif etype in ("joint_attention", "attention_capture"):
            if details.get("num_persons"):
                entry["num_persons"] = details["num_persons"]
        events.append(entry)
    return events


def load_gestures(gestures_json_path: str, max_duration: float = 120.0) -> List[dict]:
    """Load gestures, snap timestamps, filter by confidence & duration, add source tag."""
    with open(gestures_json_path, 'r') as f:
        raw = json.load(f)

    gestures = []
    for i, g in enumerate(raw.get("gestures", [])):
        if g.get("confidence", 0) < GESTURE_CONFIDENCE_THRESHOLD:
            continue

        start = _snap_to_half(g.get("start_time", 0), ceil=False)
        end = _snap_to_half(g.get("end_time", 0), ceil=True)

        if end > max_duration:
            continue

        gestures.append({
            "source": "gesture",
            "gesture_index": i,
            "gesture_type": g.get("gesture_type"),
            "start_time": start,
            "end_time": end,
            "confidence": g.get("confidence"),
            "initiator_id": g.get("initiator_id"),
            "target_type": g.get("target_type"),
            "target_person_id": g.get("target_person_id"),
            "target_description": g.get("target_description"),
        })
    return gestures


# ------------------------------------------------------------------
# GT matching for social_gesture dataset
# ------------------------------------------------------------------

def _parse_segment_name(video_name: str) -> Tuple[Optional[str], int]:
    """Parse 'video_id_NNN' → (video_id, segment_num). E.g. '152_1_001' → ('152_1', 1)."""
    match = re.match(r'^(.+)_(\d{3})$', video_name)
    if match:
        return match.group(1), int(match.group(2))
    return None, 0


def _build_segment_offsets(
    video_names: List[str], base_dir: str, dataset_name: str,
) -> Dict[str, Dict[int, float]]:
    """Build cumulative segment offsets per video_id from actual video durations."""
    segments_by_video = defaultdict(list)
    for vname in video_names:
        video_id, seg_num = _parse_segment_name(vname)
        if video_id:
            segments_by_video[video_id].append((seg_num, vname))

    offsets = {}
    for video_id, segs in segments_by_video.items():
        segs.sort()
        cumulative = 0.0
        offsets[video_id] = {}
        for seg_num, vname in segs:
            offsets[video_id][seg_num] = cumulative
            vpath = _resolve_video_path(vname, base_dir, dataset_name)
            dur = 120.0
            if vpath:
                d = _get_video_duration(vpath)
                if d:
                    dur = d
            cumulative += dur
    return offsets


def _load_gt_annotations(gt_dir: str) -> Dict[str, List[dict]]:
    """Load all GT annotation files. Returns gt_key → list of gesture dicts."""
    gt_map = {}
    for f in os.listdir(gt_dir):
        if not f.endswith('.mp4.json'):
            continue
        video_id = f.replace('.mp4.json', '')
        with open(os.path.join(gt_dir, f)) as fh:
            data = json.load(fh)
        gt_map[video_id] = data.get('gestures', [])
    return gt_map


def _find_gt_key(video_id: str, gt_map: Dict[str, List[dict]]) -> Optional[str]:
    """Find matching GT key, handling underscore/space/dot differences."""
    if video_id in gt_map:
        return video_id
    space_id = video_id.replace('_', ' ')
    if space_id in gt_map:
        return space_id
    normalized = video_id.replace(' ', '_').replace('.', '_')
    for gt_vid in gt_map:
        if gt_vid.replace(' ', '_').replace('.', '_') == normalized:
            return gt_vid
    return None


MAX_FALLBACK_GESTURES = 3


def filter_gestures_by_gt(
    gestures: List[dict], gt_gestures: List[dict], offset: float,
) -> List[dict]:
    """Keep only predicted gestures that match a GT gesture (same type + temporal overlap).

    Fallback: if no GT match, sample up to MAX_FALLBACK_GESTURES from predictions,
    picking one per gesture_type (diversity) then filling by highest confidence.
    """
    matched = []
    used_gt = set()

    for g in gestures:
        global_start = g['start_time'] + offset
        global_end = g['end_time'] + offset
        gtype = g['gesture_type']

        for j, gt in enumerate(gt_gestures):
            if j in used_gt:
                continue
            if gt.get('gesture_type') != gtype:
                continue
            gt_start = gt.get('start_time', 0)
            gt_end = gt.get('end_time', 0)
            # Any temporal overlap counts as a match
            if not (global_end < gt_start or gt_end < global_start):
                matched.append(g)
                used_gt.add(j)
                break

    if matched:
        return matched

    # Fallback: no GT match — sample diverse gestures from predictions
    if not gestures:
        return []

    # Pick 1 per gesture_type (highest confidence), then fill remaining slots
    by_type: Dict[str, list] = {}
    for g in gestures:
        by_type.setdefault(g['gesture_type'], []).append(g)

    sampled = []
    for gtype, gs in by_type.items():
        best = max(gs, key=lambda x: x.get('confidence', 0))
        sampled.append(best)

    if len(sampled) > MAX_FALLBACK_GESTURES:
        sampled.sort(key=lambda x: x.get('confidence', 0), reverse=True)
        sampled = sampled[:MAX_FALLBACK_GESTURES]
    elif len(sampled) < MAX_FALLBACK_GESTURES:
        used = set(id(s) for s in sampled)
        remaining = [g for g in gestures if id(g) not in used]
        remaining.sort(key=lambda x: x.get('confidence', 0), reverse=True)
        for g in remaining:
            if len(sampled) >= MAX_FALLBACK_GESTURES:
                break
            sampled.append(g)

    return sampled


def merge_video(
    video_name: str,
    gaze_path: Optional[str],
    gesture_path: Optional[str],
    max_duration: float,
    gt_gestures: Optional[List[dict]] = None,
    segment_offset: float = 0.0,
) -> dict:
    """Merge gaze events and gestures for a single video into one time-sorted list.

    If gt_gestures is provided, only gestures matching GT (same type + temporal overlap) are kept.
    segment_offset is the cumulative time offset for this segment within the full video.
    """
    gaze_events = load_gaze_events(gaze_path) if gaze_path else []
    gesture_events = load_gestures(gesture_path, max_duration=max_duration) if gesture_path else []

    # GT filtering for social_gesture
    num_before_gt = len(gesture_events)
    gt_filter_type = None
    if gt_gestures is not None and gesture_events:
        matched = filter_gestures_by_gt(gesture_events, gt_gestures, segment_offset)
        # Determine if result came from GT match or fallback sampling
        if num_before_gt > 0 and len(matched) > 0:
            # Check if any gesture was actually GT-matched (not fallback)
            # filter_gestures_by_gt returns matched first, falls back if empty
            # We can detect fallback by re-checking overlap
            has_real_match = False
            for g in matched:
                gs = g['start_time'] + segment_offset
                ge = g['end_time'] + segment_offset
                for gt in gt_gestures:
                    if gt.get('gesture_type') == g['gesture_type']:
                        if not (ge < gt.get('start_time', 0) or gt.get('end_time', 0) < gs):
                            has_real_match = True
                            break
                if has_real_match:
                    break
            gt_filter_type = "gt_match" if has_real_match else "fallback_sample"
        gesture_events = matched

    merged = gaze_events + gesture_events
    merged.sort(key=lambda x: (x["start_time"], x["end_time"]))

    result = {
        "video_name": video_name,
        "num_gaze_events": len(gaze_events),
        "num_gestures": len(gesture_events),
        "num_total": len(merged),
        "gaze_confidence_threshold": GAZE_CONFIDENCE_THRESHOLD,
        "gesture_confidence_threshold": GESTURE_CONFIDENCE_THRESHOLD,
        "timeline": merged,
    }

    if gt_gestures is not None:
        result["num_gestures_before_gt_filter"] = num_before_gt
        result["gt_filter_type"] = gt_filter_type or "no_gestures"

    return result


def process_dataset(
    base_dir: str,
    dataset_name: str,
    skip_existing: bool = False,
    max_files: int = 0,
) -> Tuple[int, int]:
    """Process all videos in a dataset, merging gaze + gesture data."""
    event_dir = os.path.join(base_dir, dataset_name, "event_data")
    gesture_dir = os.path.join(base_dir, dataset_name, "gesture_info")
    output_dir = os.path.join(base_dir, dataset_name, "cross_data")

    # Collect video names from both sources
    video_names = set()

    gaze_map = {}
    if os.path.isdir(event_dir):
        for f in glob.glob(os.path.join(event_dir, "*_sam3rf_gaze_events.json")):
            name = os.path.basename(f).replace("_sam3rf_gaze_events.json", "")
            gaze_map[name] = f
            video_names.add(name)

    gesture_map = {}
    if os.path.isdir(gesture_dir):
        for f in glob.glob(os.path.join(gesture_dir, "*_gestures_v2.json")):
            name = os.path.basename(f).replace("_gestures_v2.json", "")
            gesture_map[name] = f
            video_names.add(name)

    if not video_names:
        print(f"  No data found for {dataset_name}")
        return 0, 0

    video_names = sorted(video_names)
    total = len(video_names)

    if max_files > 0:
        video_names = video_names[:max_files]
        print(f"  Sampling {len(video_names)}/{total} files")

    # GT matching for social_gesture
    gt_map = None
    seg_offsets = None
    if dataset_name == "social_gesture":
        gt_dir = os.path.join(base_dir, dataset_name, "annotation")
        if os.path.isdir(gt_dir):
            gt_map = _load_gt_annotations(gt_dir)
            seg_offsets = _build_segment_offsets(video_names, base_dir, dataset_name)
            print(f"  GT loaded: {len(gt_map)} videos, filtering gestures by GT match")

    processed, skipped = 0, 0

    for i, vname in enumerate(video_names):
        output_path = os.path.join(output_dir, f"{vname}_cross_data.json")

        if skip_existing and os.path.exists(output_path):
            skipped += 1
            continue

        # Get video duration for gesture filtering
        vpath = _resolve_video_path(vname, base_dir, dataset_name)
        max_dur = 120.0
        if vpath:
            dur = _get_video_duration(vpath)
            if dur:
                max_dur = dur

        # Resolve GT for this segment
        gt_gestures = None
        offset = 0.0
        if gt_map is not None:
            video_id, seg_num = _parse_segment_name(vname)
            if video_id:
                gt_key = _find_gt_key(video_id, gt_map)
                if gt_key:
                    gt_gestures = gt_map[gt_key]
                    if seg_offsets and video_id in seg_offsets:
                        offset = seg_offsets[video_id].get(seg_num, 0.0)

        result = merge_video(
            video_name=vname,
            gaze_path=gaze_map.get(vname),
            gesture_path=gesture_map.get(vname),
            max_duration=max_dur,
            gt_gestures=gt_gestures,
            segment_offset=offset,
        )

        os.makedirs(output_dir, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2)

        processed += 1

        if (i + 1) % 500 == 0:
            print(f"  [{i+1}/{len(video_names)}] processed...")

    return processed, skipped


def main():
    parser = argparse.ArgumentParser(
        description="Merge gaze events and gestures into time-sorted cross_data JSON per video"
    )
    parser.add_argument("--base_dir", type=str, default=DEFAULT_BASE_PATH)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--max_files", type=int, default=0)
    args = parser.parse_args()

    if args.all:
        datasets = DATASETS
    elif args.dataset:
        datasets = [args.dataset]
    else:
        parser.print_help()
        return

    total_processed, total_skipped = 0, 0

    for ds in datasets:
        event_dir = os.path.join(args.base_dir, ds, "event_data")
        gesture_dir = os.path.join(args.base_dir, ds, "gesture_info")
        output_dir = os.path.join(args.base_dir, ds, "cross_data")

        has_gaze = os.path.isdir(event_dir)
        has_gesture = os.path.isdir(gesture_dir)

        if not has_gaze and not has_gesture:
            print(f"\n{ds}: no event_data or gesture_info, skipping")
            continue

        print(f"\n{'='*60}")
        print(f"Dataset: {ds}")
        print(f"  Gaze:    {event_dir} {'(found)' if has_gaze else '(not found)'}")
        print(f"  Gesture: {gesture_dir} {'(found)' if has_gesture else '(not found)'}")
        print(f"  Output:  {output_dir}")
        print(f"{'='*60}")

        p, s = process_dataset(args.base_dir, ds, args.skip_existing, args.max_files)
        total_processed += p
        total_skipped += s

        print(f"\n{ds} done: {p} processed, {s} skipped")

    print(f"\n{'='*60}")
    print(f"All done: {total_processed} processed, {total_skipped} skipped")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
