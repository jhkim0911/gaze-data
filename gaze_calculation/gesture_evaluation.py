"""
Gesture Evaluation Script

Compares generated gesture predictions against ground truth annotations using IoU matching.

Key features:
1. Uses new gesture event format from converted annotations (2fps aligned)
2. Handles video segmentation with time shifts
3. Uses temporal IoU for matching (ignores initiator_id)
4. Supports both old frame-based and new event-based GT formats

Ground Truth (GT):
    Path: /projects/illinois/eng/cs/jrehg/users/arkimjh/gaze_social/social_gesture/annotation
    Format: Event-based with gesture_type, start_time, end_time (2fps aligned to 0.5s)

Generated Data:
    Path: /projects/illinois/eng/cs/jrehg/users/arkimjh/gaze_social/social_gesture/gesture_data
    Format: Segment files (e.g., 152_1_001_gestures_v2.json) with gestures array
"""

import json
import os
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
from pathlib import Path


SEGMENT_DURATION_SEC = 120.0  # Each segment is 120 seconds
GT_FPS = 2.0  # Ground truth annotation fps (2fps aligned)
GEMINI_SAMPLE_FPS = 2.0  # Generated data is also at 2fps


@dataclass
class GestureInstance:
    """A gesture instance with temporal bounds."""
    gesture_type: str
    start_time: float  # in seconds
    end_time: float
    initiator_id: int = -1  # Not used for matching
    target_type: str = "unknown"
    target_person_id: Optional[int] = None
    target_description: str = ""
    confidence: float = 1.0
    source: str = "gt"  # "gt" or "pred"

    @property
    def duration(self) -> float:
        return max(self.end_time - self.start_time, 0.0)


@dataclass 
class EvaluationResult:
    """Evaluation metrics for gesture detection."""
    gesture_type: str
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    iou_sum: float = 0.0  # Sum of IoUs for TP matches
    
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
    
    @property
    def avg_iou(self) -> float:
        if self.true_positives == 0:
            return 0.0
        return self.iou_sum / self.true_positives


def parse_segment_name(filename: str) -> Tuple[str, int]:
    """
    Parse segment filename to get video_id and segment number.
    
    Examples:
        "152_1_001_gestures_v2.json" -> ("152_1", 1)
        "Giving Strangers Christmas Gifts!_001_gestures_v2.json" -> ("Giving Strangers Christmas Gifts!", 1)
    """
    # Remove _gestures_v2.json or _gestures.json suffix
    base = re.sub(r'_gestures(_v2)?\.json$', '', filename)
    
    # Match trailing _NNN segment number
    match = re.match(r'^(.+)_(\d{3})$', base)
    if match:
        video_id = match.group(1)
        segment_num = int(match.group(2))
        return video_id, segment_num
    return None, 0


def get_segment_offset(segment_num: int, segment_duration: float = SEGMENT_DURATION_SEC) -> float:
    """Get time offset for a segment (1-indexed)."""
    return (segment_num - 1) * segment_duration


def has_temporal_overlap(g1: GestureInstance, g2: GestureInstance) -> bool:
    """
    Check if two gestures have any temporal overlap.
    
    Simple overlap: returns True if there's any intersection at all.
    """
    # Check if intervals overlap: NOT (g1 ends before g2 starts OR g2 ends before g1 starts)
    return not (g1.end_time < g2.start_time or g2.end_time < g1.start_time)


def compute_overlap_duration(g1: GestureInstance, g2: GestureInstance) -> float:
    """Compute the duration of overlap between two gestures."""
    inter_start = max(g1.start_time, g2.start_time)
    inter_end = min(g1.end_time, g2.end_time)
    return max(0, inter_end - inter_start)


def load_gt_annotations_event_format(gt_path: str) -> List[GestureInstance]:
    """
    Load ground truth annotations from the converted event-based format.
    
    Args:
        gt_path: Path to the JSON file with gesture events
    
    Returns:
        List of GestureInstance objects
    """
    with open(gt_path, 'r') as f:
        data = json.load(f)
    
    gestures = []
    for g in data.get('gestures', []):
        gesture = GestureInstance(
            gesture_type=g.get('gesture_type', 'unknown'),
            start_time=g.get('start_time', 0),
            end_time=g.get('end_time', 0),
            initiator_id=g.get('initiator_id', -1),
            target_type=g.get('target_type', 'unknown'),
            target_person_id=g.get('target_person_id'),
            target_description=g.get('target_description', ''),
            confidence=g.get('confidence', 1.0),
            source="gt"
        )
        gestures.append(gesture)
    
    return gestures


def load_generated_gestures(gesture_dir: str, video_id: str) -> List[GestureInstance]:
    """
    Load and merge generated gesture predictions from all segments of a video.
    
    Args:
        gesture_dir: Directory containing *_gestures_v2.json files
        video_id: Video ID (e.g., "152_1")
    
    Returns:
        List of GestureInstance with times adjusted to full video timeline
    """
    gestures = []
    
    # Find all segment files for this video
    segment_files = []
    for fname in os.listdir(gesture_dir):
        if not fname.endswith('_gestures_v2.json'):
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
        
        for g in data.get('gestures', []):
            # Get times (already in seconds at 2fps)
            gesture_start = g.get('start_time', 0)
            gesture_end = g.get('end_time', 0)
            
            # Apply segment time offset
            global_start = offset + gesture_start
            global_end = offset + gesture_end
            
            gesture = GestureInstance(
                gesture_type=g.get('gesture_type', 'unknown'),
                start_time=global_start,
                end_time=global_end,
                initiator_id=g.get('initiator_id', -1),
                target_type=g.get('target_type', 'unknown'),
                target_person_id=g.get('target_person_id'),
                target_description=g.get('target_description', ''),
                confidence=g.get('confidence', 1.0),
                source="pred"
            )
            gestures.append(gesture)
    
    return gestures


def match_gestures_overlap(gt_gestures: List[GestureInstance], 
                            pred_gestures: List[GestureInstance]) -> Dict[str, EvaluationResult]:
    """
    Match predicted gestures to ground truth using simple overlap.
    
    Any temporal overlap between prediction and GT counts as a match.
    Matching is done by gesture_type only (initiator_id is ignored).
    Uses greedy matching: for each prediction, find any overlapping unmatched GT.
    
    Args:
        gt_gestures: Ground truth gesture instances
        pred_gestures: Predicted gesture instances
    
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
        
        # For each prediction, find any overlapping GT
        for i, pred in enumerate(pred_list):
            for j, gt in enumerate(gt_list):
                if j in matched_gt:
                    continue
                
                # Simple overlap check - any overlap counts as a match
                if has_temporal_overlap(pred, gt):
                    matched_gt.add(j)
                    matched_pred.add(i)
                    result.true_positives += 1
                    result.iou_sum += compute_overlap_duration(pred, gt)  # Track overlap duration
                    break
        
        result.false_positives = len(pred_list) - len(matched_pred)
        result.false_negatives = len(gt_list) - len(matched_gt)
        
        results[g_type] = result
    
    return results


def evaluate_video(gt_path: str, gesture_dir: str, video_id: str) -> Tuple[Dict[str, EvaluationResult], 
                                                         List[GestureInstance], 
                                                         List[GestureInstance]]:
    """
    Evaluate gesture detection for a single video.
    
    Args:
        gt_path: Path to GT annotation file
        gesture_dir: Directory containing generated gesture files
        video_id: Video ID
    
    Returns:
        Tuple of (results dict, gt_gestures list, pred_gestures list)
    """
    # Load GT
    gt_gestures = load_gt_annotations_event_format(gt_path)
    
    # Load predictions
    pred_gestures = load_generated_gestures(gesture_dir, video_id)
    
    # Match and evaluate using simple overlap
    results = match_gestures_overlap(gt_gestures, pred_gestures)
    
    return results, gt_gestures, pred_gestures


def print_evaluation_report(results: Dict[str, EvaluationResult],
                            gt_gestures: List[GestureInstance],
                            pred_gestures: List[GestureInstance],
                            video_id: str):
    """Print a formatted evaluation report."""
    print(f"\n{'='*70}")
    print(f"Gesture Evaluation Report: {video_id}")
    print(f"{'='*70}")
    
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
    
    print(f"\n{'─'*70}")
    print(f"{'Gesture Type':<15} {'TP':<5} {'FP':<5} {'FN':<5} {'Prec':<8} {'Rec':<8} {'F1':<8} {'AvgIoU':<8}")
    print(f"{'─'*70}")
    
    total_tp, total_fp, total_fn = 0, 0, 0
    total_iou_sum = 0.0
    
    for g_type in ['pointing', 'showing', 'giving', 'reaching']:
        if g_type in results:
            r = results[g_type]
            print(f"{g_type:<15} {r.true_positives:<5} {r.false_positives:<5} {r.false_negatives:<5} "
                  f"{r.precision:.4f}  {r.recall:.4f}  {r.f1:.4f}  {r.avg_iou:.4f}")
            total_tp += r.true_positives
            total_fp += r.false_positives
            total_fn += r.false_negatives
            total_iou_sum += r.iou_sum
    
    print(f"{'─'*70}")
    
    # Overall metrics
    overall_prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    overall_rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    overall_f1 = 2 * overall_prec * overall_rec / (overall_prec + overall_rec) if (overall_prec + overall_rec) > 0 else 0
    overall_avg_iou = total_iou_sum / total_tp if total_tp > 0 else 0
    
    print(f"{'OVERALL':<15} {total_tp:<5} {total_fp:<5} {total_fn:<5} "
          f"{overall_prec:.4f}  {overall_rec:.4f}  {overall_f1:.4f}  {overall_avg_iou:.4f}")
    print()


def find_matching_gt_file(gt_dir: str, video_id: str) -> Optional[str]:
    """
    Find matching GT annotation file for a video ID.
    
    Handles both:
    - Simple format: "152_1.mp4.json" for video_id "152_1"  
    - Complex format: "Giving Strangers Christmas Gifts!.mp4.json" 
    """
    # Try direct match
    simple_path = os.path.join(gt_dir, f"{video_id}.mp4.json")
    if os.path.exists(simple_path):
        return simple_path
    
    # Try with underscores converted to spaces
    space_id = video_id.replace('_', ' ')
    space_path = os.path.join(gt_dir, f"{space_id}.mp4.json")
    if os.path.exists(space_path):
        return space_path
    
    # Search for partial matches
    for f in os.listdir(gt_dir):
        if f.endswith('.mp4.json'):
            gt_name = f.replace('.mp4.json', '')
            # Normalize for comparison
            normalized_gt = gt_name.replace(' ', '_').replace('.', '_')
            normalized_vid = video_id.replace(' ', '_').replace('.', '_')
            if normalized_gt == normalized_vid:
                return os.path.join(gt_dir, f)
    
    return None


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate generated gesture predictions against GT using simple overlap')
    parser.add_argument('--gt_dir', type=str, 
                        default='/projects/illinois/eng/cs/jrehg/users/arkimjh/gaze_social/social_gesture/annotation',
                        help='Directory containing GT annotation files')
    parser.add_argument('--gesture_dir', type=str,
                        default='/projects/illinois/eng/cs/jrehg/users/arkimjh/gaze_social/social_gesture/gesture_data',
                        help='Directory containing generated gesture predictions')
    parser.add_argument('--video_id', type=str, default=None,
                        help='Video ID(s) to evaluate, comma-separated. If not provided, evaluates all.')
    parser.add_argument('--verbose', action='store_true',
                        help='Print detailed gesture lists')
    
    args = parser.parse_args()
    
    # Find videos to evaluate
    if args.video_id:
        video_ids = [v.strip() for v in args.video_id.split(',')]
    else:
        # Find all videos with predictions
        pred_videos = set()
        for f in os.listdir(args.gesture_dir):
            if f.endswith('_gestures_v2.json'):
                video_id, _ = parse_segment_name(f)
                if video_id:
                    pred_videos.add(video_id)
        
        video_ids = sorted(pred_videos)
        print(f"Found {len(video_ids)} videos with predictions")
    
    # Aggregate results
    all_results = defaultdict(lambda: EvaluationResult(gesture_type=''))
    evaluated_count = 0
    
    for video_id in video_ids:
        gt_path = find_matching_gt_file(args.gt_dir, video_id)
        
        if not gt_path:
            print(f"Warning: GT not found for {video_id}")
            continue
        
        results, gt_gestures, pred_gestures = evaluate_video(
            gt_path, args.gesture_dir, video_id
        )
        
        print_evaluation_report(results, gt_gestures, pred_gestures, video_id)
        evaluated_count += 1
        
        if args.verbose:
            print("\nGT Gestures:")
            for g in gt_gestures[:10]:
                print(f"  {g.gesture_type} @ {g.start_time:.1f}s-{g.end_time:.1f}s")
            
            print("\nPredicted Gestures:")
            for g in pred_gestures[:10]:
                print(f"  {g.gesture_type} @ {g.start_time:.1f}s-{g.end_time:.1f}s")
        
        # Aggregate
        for g_type, result in results.items():
            all_results[g_type].gesture_type = g_type
            all_results[g_type].true_positives += result.true_positives
            all_results[g_type].false_positives += result.false_positives
            all_results[g_type].false_negatives += result.false_negatives
            all_results[g_type].iou_sum += result.iou_sum
    
    # Print overall summary if multiple videos
    if evaluated_count > 1:
        print(f"\n{'='*70}")
        print(f"AGGREGATE RESULTS ({evaluated_count} videos)")
        print(f"{'='*70}")
        print(f"\n{'Gesture Type':<15} {'TP':<5} {'FP':<5} {'FN':<5} {'Prec':<8} {'Rec':<8} {'F1':<8} {'AvgIoU':<8}")
        print(f"{'─'*70}")
        
        total_tp, total_fp, total_fn = 0, 0, 0
        total_iou_sum = 0.0
        
        for g_type in ['pointing', 'showing', 'giving', 'reaching']:
            if g_type in all_results:
                r = all_results[g_type]
                print(f"{g_type:<15} {r.true_positives:<5} {r.false_positives:<5} {r.false_negatives:<5} "
                      f"{r.precision:.4f}  {r.recall:.4f}  {r.f1:.4f}  {r.avg_iou:.4f}")
                total_tp += r.true_positives
                total_fp += r.false_positives
                total_fn += r.false_negatives
                total_iou_sum += r.iou_sum
        
        print(f"{'─'*70}")
        overall_prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        overall_rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        overall_f1 = 2 * overall_prec * overall_rec / (overall_prec + overall_rec) if (overall_prec + overall_rec) > 0 else 0
        overall_avg_iou = total_iou_sum / total_tp if total_tp > 0 else 0
        
        print(f"{'OVERALL':<15} {total_tp:<5} {total_fp:<5} {total_fn:<5} "
              f"{overall_prec:.4f}  {overall_rec:.4f}  {overall_f1:.4f}  {overall_avg_iou:.4f}")


if __name__ == '__main__':
    main()
