"""
Gesture Evaluation Script

Compares Gemini gesture predictions against ground truth annotations.

Key challenges addressed:
1. Video segmentation: Gemini processes 152_1_001, 152_1_002, etc. but GT is 152_1.mp4.json
2. FPS difference: Segments are at 5fps (same as GT), but Gemini event times are based on 2fps sampling
3. Temporal alignment: Need to map Gemini event times to GT frame numbers

GT annotation structure (5fps):
- Frame-level annotations with gesture_types per person
- frame status: "start frame", "active", "representative frame", "end frame", null
- 4 gesture types: pointing, showing, giving, reaching

Gemini output structure:
- Event-based with deictic_gestures array
- Each gesture has: gesture_type, initiator_id, target_type, target_person_id, target_description
- Times are in seconds relative to segment start
"""

import json
import os
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
from pathlib import Path


SEGMENT_DURATION_SEC = 120.0  # Each segment is 120 seconds
GT_FPS = 5.0  # Ground truth annotation fps
GEMINI_SAMPLE_FPS = 2.0  # Gemini event detection fps


@dataclass
class GestureInstance:
    """A gesture instance with temporal bounds."""
    gesture_type: str
    person_id: int
    start_frame: int  # GT frame number (1-indexed, 5fps)
    end_frame: int
    start_time: float  # in seconds from video start
    end_time: float
    target_person_ids: List[int] = field(default_factory=list)
    related_objects: List[str] = field(default_factory=list)
    representative_frame: Optional[int] = None
    source: str = "gt"  # "gt" or "gemini"


@dataclass 
class EvaluationResult:
    """Evaluation metrics for gesture detection."""
    gesture_type: str
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    
    @property
    def precision(self) -> float:
        if self.true_positives + self.false_positives == 0:
            return 0.0
        return self.true_positives / (self.true_positives + self.false_positives)
    
    @property
    def recall(self) -> float:
        if self.true_positives + self.false_negatives == 0:
            return 0.0
        return self.true_positives / (self.true_positives + self.false_negatives)
    
    @property
    def f1(self) -> float:
        if self.precision + self.recall == 0:
            return 0.0
        return 2 * self.precision * self.recall / (self.precision + self.recall)


def parse_segment_name(filename: str) -> Tuple[str, int]:
    """
    Parse segment filename to get video_id and segment number.
    
    Examples:
        "152_1_001_gestures.json" -> ("152_1", 1)
        "152_1_001_sam3rf_viz_gestures_v2.json" -> ("152_1", 1)
    """
    # Match pattern: {video_id}_{segment_num}_*.json (handles both v1 and v2)
    match = re.match(r'^(.+?)_(\d{3})(?:_.*)?\.json$', filename)
    if match:
        video_id = match.group(1)
        segment_num = int(match.group(2))
        return video_id, segment_num
    return None, 0


def get_segment_offset(segment_num: int, segment_duration: float = SEGMENT_DURATION_SEC) -> float:
    """Get time offset for a segment (1-indexed)."""
    return (segment_num - 1) * segment_duration


def time_to_gt_frame(time_sec: float, fps: float = GT_FPS) -> int:
    """Convert time in seconds to GT frame number (1-indexed)."""
    return int(time_sec * fps) + 1


def gt_frame_to_time(frame: int, fps: float = GT_FPS) -> float:
    """Convert GT frame number (1-indexed) to time in seconds."""
    return (frame - 1) / fps


def load_gt_annotations(gt_path: str) -> Dict:
    """Load ground truth annotations from a single video file."""
    with open(gt_path, 'r') as f:
        return json.load(f)


def extract_gt_gestures(gt_data: Dict) -> List[GestureInstance]:
    """
    Extract gesture instances from GT annotations.
    
    Groups consecutive frames with the same gesture type and person into instances.
    Uses 'start frame', 'end frame', 'representative frame' markers.
    """
    gestures = []
    annotations = gt_data.get('annotations', [])
    
    # Track active gestures: (person_id, gesture_type) -> GestureInstance
    active_gestures: Dict[Tuple[int, str], GestureInstance] = {}
    
    for ann in annotations:
        frame = ann.get('frame')
        segments_info = ann.get('segments_info', [])
        
        # Track which gestures are active in this frame
        current_frame_gestures: Set[Tuple[int, str]] = set()
        
        for seg in segments_info:
            if seg.get('category_id') != 'person':
                continue
                
            person_id = seg.get('id')
            attrs = seg.get('attributes', {})
            gesture_type = attrs.get('gesture_types', 'None')
            
            if gesture_type and gesture_type != 'None':
                key = (person_id, gesture_type)
                current_frame_gestures.add(key)
                frame_status = attrs.get('frame')
                
                if key not in active_gestures:
                    # Start new gesture
                    active_gestures[key] = GestureInstance(
                        gesture_type=gesture_type,
                        person_id=person_id,
                        start_frame=frame,
                        end_frame=frame,
                        start_time=gt_frame_to_time(frame),
                        end_time=gt_frame_to_time(frame),
                        target_person_ids=[int(x) for x in attrs.get('target_person ID', []) if x],
                        related_objects=attrs.get('ralated objects', []),
                        source="gt"
                    )
                else:
                    # Update existing gesture
                    active_gestures[key].end_frame = frame
                    active_gestures[key].end_time = gt_frame_to_time(frame)
                    
                    # Update target info if available
                    target_ids = [int(x) for x in attrs.get('target_person ID', []) if x]
                    if target_ids:
                        active_gestures[key].target_person_ids = target_ids
                    
                if frame_status == 'representative frame':
                    active_gestures[key].representative_frame = frame
        
        # Close gestures that are no longer active
        keys_to_remove = []
        for key in active_gestures:
            if key not in current_frame_gestures:
                gestures.append(active_gestures[key])
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del active_gestures[key]
    
    # Add any remaining active gestures
    gestures.extend(active_gestures.values())
    
    return gestures


def load_gemini_gestures(gesture_dir: str, video_id: str) -> List[GestureInstance]:
    """
    Load and merge Gemini gesture predictions from all segments of a video.
    
    Supports both v1 (classifications) and v2 (gestures array) formats.
    
    Args:
        gesture_dir: Directory containing *_gestures.json or *_gestures_v2.json files
        video_id: Video ID (e.g., "152_1")
    
    Returns:
        List of GestureInstance with times adjusted to full video timeline
    """
    gestures = []
    
    # Find all segment files for this video (both v1 and v2)
    segment_files = []
    for fname in os.listdir(gesture_dir):
        if not (fname.endswith('_gestures.json') or fname.endswith('_gestures_v2.json')):
            continue
        seg_video_id, seg_num = parse_segment_name(fname)
        if seg_video_id == video_id:
            segment_files.append((seg_num, fname))
    
    segment_files.sort(key=lambda x: x[0])
    
    for seg_num, fname in segment_files:
        fpath = os.path.join(gesture_dir, fname)
        with open(fpath, 'r') as f:
            data = json.load(f)
        
        offset = get_segment_offset(seg_num)
        
        # Check if v2 format (has 'gestures' array directly)
        if data.get('classifier_version') == 'v2' or 'gestures' in data:
            # V2 format: gestures are directly in 'gestures' array with start/end frames
            sample_fps = 2.0  # Video is sampled at 2fps
            for g in data.get('gestures', []):
                # Support both old (start_time) and new (start_frame) format
                if 'start_frame' in g:
                    gesture_start = g.get('start_frame', 0) / sample_fps
                    gesture_end = g.get('end_frame', 0) / sample_fps
                else:
                    gesture_start = g.get('start_time', 0)
                    gesture_end = g.get('end_time', 0)
                
                # Convert to global time
                global_start = offset + gesture_start
                global_end = offset + gesture_end
                
                gesture = GestureInstance(
                    gesture_type=g.get('gesture_type', 'unknown'),
                    person_id=g.get('initiator_id', -1),
                    start_frame=time_to_gt_frame(global_start),
                    end_frame=time_to_gt_frame(global_end),
                    start_time=global_start,
                    end_time=global_end,
                    target_person_ids=[g['target_person_id']] if g.get('target_person_id') is not None else [],
                    source="gemini"
                )
                gestures.append(gesture)
        else:
            # V1 format: gestures are nested in 'classifications' -> 'deictic_gestures'
            for cls in data.get('classifications', []):
                deictic_gestures = cls.get('deictic_gestures', [])
                if not deictic_gestures:
                    continue
                
                # Event time range (in segment-local time)
                event_start = cls.get('start_time', 0)
                event_end = cls.get('end_time', 0)
                
                # Convert to global time
                global_start = offset + event_start
                global_end = offset + event_end
                
                for g in deictic_gestures:
                    gesture = GestureInstance(
                        gesture_type=g.get('gesture_type', 'unknown'),
                        person_id=g.get('initiator_id', -1),
                        start_frame=time_to_gt_frame(global_start),
                        end_frame=time_to_gt_frame(global_end),
                        start_time=global_start,
                        end_time=global_end,
                        target_person_ids=[g['target_person_id']] if g.get('target_person_id') is not None else [],
                        source="gemini"
                    )
                    gestures.append(gesture)
    
    return gestures


def check_temporal_overlap(g1: GestureInstance, g2: GestureInstance, 
                           tolerance_sec: float = 2.0) -> bool:
    """
    Check if two gesture instances have temporal overlap within tolerance.
    """
    # Expand ranges by tolerance
    g1_start = g1.start_time - tolerance_sec
    g1_end = g1.end_time + tolerance_sec
    g2_start = g2.start_time - tolerance_sec
    g2_end = g2.end_time + tolerance_sec
    
    # Check overlap
    return not (g1_end < g2_start or g2_end < g1_start)


def match_gestures(gt_gestures: List[GestureInstance], 
                   pred_gestures: List[GestureInstance],
                   tolerance_sec: float = 2.0,
                   require_person_match: bool = False) -> Dict[str, EvaluationResult]:
    """
    Match predicted gestures to ground truth and compute metrics.
    
    Args:
        gt_gestures: Ground truth gesture instances
        pred_gestures: Predicted gesture instances from Gemini
        tolerance_sec: Time tolerance for matching (seconds)
        require_person_match: If True, also require person_id to match
    
    Returns:
        Dict mapping gesture_type to EvaluationResult
    """
    # Group by gesture type
    gt_by_type = defaultdict(list)
    pred_by_type = defaultdict(list)
    
    for g in gt_gestures:
        gt_by_type[g.gesture_type].append(g)
    for g in pred_gestures:
        pred_by_type[g.gesture_type].append(g)
    
    # All gesture types
    all_types = set(gt_by_type.keys()) | set(pred_by_type.keys())
    
    results = {}
    
    for g_type in all_types:
        result = EvaluationResult(gesture_type=g_type)
        gt_list = gt_by_type[g_type]
        pred_list = pred_by_type[g_type]
        
        # Track which GTs have been matched
        matched_gt = set()
        matched_pred = set()
        
        # Greedy matching
        for i, pred in enumerate(pred_list):
            for j, gt in enumerate(gt_list):
                if j in matched_gt:
                    continue
                
                # Check temporal overlap
                if not check_temporal_overlap(pred, gt, tolerance_sec):
                    continue
                
                # Optionally check person match
                if require_person_match and pred.person_id != gt.person_id:
                    continue
                
                # Match found
                matched_gt.add(j)
                matched_pred.add(i)
                result.true_positives += 1
                break
        
        result.false_positives = len(pred_list) - len(matched_pred)
        result.false_negatives = len(gt_list) - len(matched_gt)
        
        results[g_type] = result
    
    return results


def evaluate_video(gt_path: str, gesture_dir: str, video_id: str,
                   tolerance_sec: float = 2.0) -> Dict[str, EvaluationResult]:
    """
    Evaluate gesture detection for a single video.
    """
    # Load GT
    gt_data = load_gt_annotations(gt_path)
    gt_gestures = extract_gt_gestures(gt_data)
    
    # Load predictions
    pred_gestures = load_gemini_gestures(gesture_dir, video_id)
    
    # Match and evaluate
    results = match_gestures(gt_gestures, pred_gestures, tolerance_sec)
    
    return results, gt_gestures, pred_gestures


def print_evaluation_report(results: Dict[str, EvaluationResult],
                            gt_gestures: List[GestureInstance],
                            pred_gestures: List[GestureInstance],
                            video_id: str):
    """Print a formatted evaluation report."""
    print(f"\n{'='*60}")
    print(f"Gesture Evaluation Report: {video_id}")
    print(f"{'='*60}")
    
    print(f"\nGround Truth: {len(gt_gestures)} gesture instances")
    for g_type in ['pointing', 'showing', 'giving', 'reaching']:
        count = sum(1 for g in gt_gestures if g.gesture_type == g_type)
        if count > 0:
            print(f"  - {g_type}: {count}")
    
    print(f"\nPredictions: {len(pred_gestures)} gesture instances")
    for g_type in ['pointing', 'showing', 'giving', 'reaching']:
        count = sum(1 for g in pred_gestures if g.gesture_type == g_type)
        if count > 0:
            print(f"  - {g_type}: {count}")
    
    print(f"\n{'─'*60}")
    print(f"{'Gesture Type':<15} {'TP':<5} {'FP':<5} {'FN':<5} {'Prec':<8} {'Rec':<8} {'F1':<8}")
    print(f"{'─'*60}")
    
    total_tp, total_fp, total_fn = 0, 0, 0
    
    for g_type in ['pointing', 'showing', 'giving', 'reaching']:
        if g_type in results:
            r = results[g_type]
            print(f"{g_type:<15} {r.true_positives:<5} {r.false_positives:<5} {r.false_negatives:<5} "
                  f"{r.precision:.4f}  {r.recall:.4f}  {r.f1:.4f}")
            total_tp += r.true_positives
            total_fp += r.false_positives
            total_fn += r.false_negatives
    
    print(f"{'─'*60}")
    
    # Overall metrics
    overall_prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    overall_rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    overall_f1 = 2 * overall_prec * overall_rec / (overall_prec + overall_rec) if (overall_prec + overall_rec) > 0 else 0
    
    print(f"{'OVERALL':<15} {total_tp:<5} {total_fp:<5} {total_fn:<5} "
          f"{overall_prec:.4f}  {overall_rec:.4f}  {overall_f1:.4f}")
    print()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate Gemini gesture predictions against GT')
    parser.add_argument('--gt_dir', type=str, 
                        default='/projects/illinois/eng/cs/jrehg/datasets-social/SocialGesture/annotations_v3',
                        help='Directory containing GT annotation files')
    parser.add_argument('--gesture_dir', type=str,
                        default='/projects/illinois/eng/cs/jrehg/users/arkimjh/gaze_social/social_gesture/gesture_data',
                        help='Directory containing Gemini gesture predictions')
    parser.add_argument('--video_id', type=str, default=None,
                        help='Video ID(s) to evaluate, comma-separated (e.g., "152_1,152_2"). If not provided, evaluates all.')
    parser.add_argument('--tolerance', type=float, default=2.0,
                        help='Time tolerance for matching in seconds')
    parser.add_argument('--verbose', action='store_true',
                        help='Print detailed gesture lists')
    
    args = parser.parse_args()
    
    # Find videos to evaluate
    if args.video_id:
        # Support comma-separated video IDs
        video_ids = [v.strip() for v in args.video_id.split(',')]
    else:
        # Find all videos with both GT and predictions
        gt_videos = set()
        for f in os.listdir(args.gt_dir):
            if f.endswith('.mp4.json'):
                video_id = f.replace('.mp4.json', '')
                gt_videos.add(video_id)
        
        pred_videos = set()
        for f in os.listdir(args.gesture_dir):
            if f.endswith('_gestures.json'):
                video_id, _ = parse_segment_name(f)
                if video_id:
                    pred_videos.add(video_id)
        
        video_ids = sorted(gt_videos & pred_videos)
        print(f"Found {len(video_ids)} videos with both GT and predictions")
    
    # Aggregate results
    all_results = defaultdict(lambda: EvaluationResult(gesture_type=''))
    
    for video_id in video_ids:
        gt_path = os.path.join(args.gt_dir, f"{video_id}.mp4.json")
        
        if not os.path.exists(gt_path):
            print(f"Warning: GT not found for {video_id}")
            continue
        
        results, gt_gestures, pred_gestures = evaluate_video(
            gt_path, args.gesture_dir, video_id, args.tolerance
        )
        
        print_evaluation_report(results, gt_gestures, pred_gestures, video_id)
        
        if args.verbose:
            print("\nGT Gestures:")
            for g in gt_gestures[:10]:
                print(f"  {g.gesture_type} by P{g.person_id} @ frames {g.start_frame}-{g.end_frame} "
                      f"({g.start_time:.1f}s-{g.end_time:.1f}s)")
            
            print("\nPredicted Gestures:")
            for g in pred_gestures[:10]:
                print(f"  {g.gesture_type} by P{g.person_id} @ {g.start_time:.1f}s-{g.end_time:.1f}s")
        
        # Aggregate
        for g_type, result in results.items():
            all_results[g_type].gesture_type = g_type
            all_results[g_type].true_positives += result.true_positives
            all_results[g_type].false_positives += result.false_positives
            all_results[g_type].false_negatives += result.false_negatives
    
    # Print overall summary if multiple videos
    if len(video_ids) > 1:
        print(f"\n{'='*60}")
        print(f"AGGREGATE RESULTS ({len(video_ids)} videos)")
        print(f"{'='*60}")
        print(f"\n{'Gesture Type':<15} {'TP':<5} {'FP':<5} {'FN':<5} {'Prec':<8} {'Rec':<8} {'F1':<8}")
        print(f"{'─'*60}")
        
        total_tp, total_fp, total_fn = 0, 0, 0
        for g_type in ['pointing', 'showing', 'giving', 'reaching']:
            if g_type in all_results:
                r = all_results[g_type]
                print(f"{g_type:<15} {r.true_positives:<5} {r.false_positives:<5} {r.false_negatives:<5} "
                      f"{r.precision:.4f}  {r.recall:.4f}  {r.f1:.4f}")
                total_tp += r.true_positives
                total_fp += r.false_positives
                total_fn += r.false_negatives
        
        print(f"{'─'*60}")
        overall_prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        overall_rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        overall_f1 = 2 * overall_prec * overall_rec / (overall_prec + overall_rec) if (overall_prec + overall_rec) > 0 else 0
        print(f"{'OVERALL':<15} {total_tp:<5} {total_fp:<5} {total_fn:<5} "
              f"{overall_prec:.4f}  {overall_rec:.4f}  {overall_f1:.4f}")


if __name__ == '__main__':
    main()
