#!/usr/bin/env python3
"""
Gaze Event Evaluator

Evaluates predicted gaze events against ground truth annotations.
Maps GT taxonomy (share, single, mutual, refer) to prediction taxonomy
(sudden_gaze_shift, joint_attention, gaze_following, attention_capture, mutual_gaze).

Usage:
    python gaze_event_evaluator.py \
        --gt_dir /path/to/annotation \
        --pred_dir /path/to/event_data \
        --output_dir /path/to/eval_results
"""

import argparse
import json
import os
import re
from collections import defaultdict
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
import glob


# =============================================================================
# Taxonomy Mapping
# =============================================================================

# GT atomic attributes -> Predicted event types they can match
TAXONOMY_MAPPING = {
    'mutual': ['mutual_gaze'],
    'share': ['joint_attention'],
    'refer': ['joint_attention', 'gaze_following'],  # Can match either
    'single': [],  # No direct mapping (one person looking at something)
}

# Reverse mapping: prediction type -> GT types it can match
PRED_TO_GT_MAPPING = {
    'mutual_gaze': ['mutual'],
    'joint_attention': ['share', 'refer'],
    'gaze_following': ['refer'],
    'sudden_gaze_shift': [],  # No direct GT equivalent
    'attention_capture': [],  # No direct GT equivalent
}


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class GTEvent:
    """Ground truth event segment."""
    video_id: str
    event_type: str  # atomic attribute: share, single, mutual, refer
    start_frame: int
    end_frame: int
    
    @property
    def duration_frames(self) -> int:
        return self.end_frame - self.start_frame + 1


@dataclass 
class PredEvent:
    """Predicted event."""
    video_id: str
    event_type: str
    start_frame: int
    end_frame: int
    confidence: float
    
    @property
    def duration_frames(self) -> int:
        return self.end_frame - self.start_frame + 1


@dataclass
class EventMatch:
    """A matched pair of GT and predicted events."""
    gt_event: GTEvent
    pred_event: PredEvent
    temporal_iou: float


# =============================================================================
# GT Annotation Parser
# =============================================================================

def parse_gt_annotation(filepath: str) -> List[GTEvent]:
    """
    Parse a GT annotation file into event segments.
    
    Format: bbx ID | xmin | ymin| xmax | ymax | frame ID | lost | occluded | generated | bbx label | event attribute | atomic attribute | attention focus
    
    Returns:
        List of GTEvent segments (consecutive frames with same atomic attribute)
    """
    video_id = os.path.basename(filepath).replace('NewAnt_', '').replace('.txt', '')
    
    # Collect per-frame atomic attributes (only for Person entries)
    frame_events: Dict[int, set] = defaultdict(set)
    
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 12:
                continue
            
            # Check if this is a Person entry (not Object)
            bbx_label = parts[9]
            if not bbx_label.startswith('Person'):
                continue
            
            frame_id = int(parts[5])
            
            # Get atomic attribute (column 11, 0-indexed)
            if len(parts) >= 12:
                atomic_attr = parts[11].lower()
                if atomic_attr in TAXONOMY_MAPPING:
                    frame_events[frame_id].add(atomic_attr)
    
    # Group consecutive frames with same event type into segments
    events = []
    sorted_frames = sorted(frame_events.keys())
    
    if not sorted_frames:
        return events
    
    # For each event type, find contiguous segments
    for event_type in ['mutual', 'share', 'refer', 'single']:
        i = 0
        while i < len(sorted_frames):
            frame = sorted_frames[i]
            if event_type in frame_events[frame]:
                # Start of a segment
                start_frame = frame
                end_frame = frame
                
                # Extend segment while consecutive frames have this event
                j = i + 1
                while j < len(sorted_frames):
                    next_frame = sorted_frames[j]
                    # Allow small gaps (up to 5 frames for 2fps = 2.5s)
                    if next_frame - end_frame <= 5 and event_type in frame_events[next_frame]:
                        end_frame = next_frame
                        j += 1
                    else:
                        break
                
                # Only keep segments with at least 2 frames
                if end_frame - start_frame >= 1:
                    events.append(GTEvent(
                        video_id=video_id,
                        event_type=event_type,
                        start_frame=start_frame,
                        end_frame=end_frame,
                    ))
                
                i = j
            else:
                i += 1
    
    return events


def load_all_gt_annotations(gt_dir: str) -> Dict[str, List[GTEvent]]:
    """Load all GT annotations from directory."""
    gt_files = glob.glob(os.path.join(gt_dir, 'NewAnt_*.txt'))
    
    all_events = {}
    for filepath in sorted(gt_files):
        video_id = os.path.basename(filepath).replace('NewAnt_', '').replace('.txt', '')
        events = parse_gt_annotation(filepath)
        all_events[video_id] = events
    
    return all_events


# =============================================================================
# Prediction Loader
# =============================================================================

def load_predictions(pred_dir: str, min_confidence: float = 0.0) -> Dict[str, List[PredEvent]]:
    """Load all predicted events from directory, optionally filtering by confidence."""
    pred_files = glob.glob(os.path.join(pred_dir, '*_sam3rf_gaze_events.json'))
    
    all_events = {}
    for filepath in sorted(pred_files):
        # Extract video ID from filename
        basename = os.path.basename(filepath)
        video_id = basename.replace('_sam3rf_gaze_events.json', '')
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        events = []
        for e in data.get('events', []):
            conf = e.get('confidence', 1.0)
            # Filter by confidence threshold
            if conf < min_confidence:
                continue
            events.append(PredEvent(
                video_id=video_id,
                event_type=e['event_type'],
                start_frame=e['start_frame'],
                end_frame=e['end_frame'],
                confidence=conf,
            ))
        
        all_events[video_id] = events
    
    return all_events


# =============================================================================
# Evaluation Metrics
# =============================================================================

def compute_temporal_iou(event1_start: int, event1_end: int, 
                         event2_start: int, event2_end: int) -> float:
    """Compute temporal IoU between two event segments."""
    intersection_start = max(event1_start, event2_start)
    intersection_end = min(event1_end, event2_end)
    
    if intersection_end < intersection_start:
        return 0.0
    
    intersection = intersection_end - intersection_start + 1
    union = (event1_end - event1_start + 1) + (event2_end - event2_start + 1) - intersection
    
    return intersection / union if union > 0 else 0.0


def match_events(gt_events: List[GTEvent], pred_events: List[PredEvent],
                 iou_threshold: float = 0.3) -> Tuple[List[EventMatch], List[GTEvent], List[PredEvent]]:
    """
    Match GT events to predicted events using temporal IoU.
    
    Returns:
        Tuple of (matched_pairs, unmatched_gt, unmatched_pred)
    """
    matched = []
    matched_gt_indices = set()
    matched_pred_indices = set()
    
    # For each GT event, find best matching prediction
    for gi, gt in enumerate(gt_events):
        best_match = None
        best_iou = 0.0
        best_pi = -1
        
        # Get compatible prediction types for this GT type
        compatible_types = TAXONOMY_MAPPING.get(gt.event_type, [])
        if not compatible_types:
            continue
        
        for pi, pred in enumerate(pred_events):
            if pi in matched_pred_indices:
                continue
            
            # Check if prediction type is compatible
            if pred.event_type not in compatible_types:
                continue
            
            iou = compute_temporal_iou(gt.start_frame, gt.end_frame,
                                       pred.start_frame, pred.end_frame)
            
            if iou > best_iou and iou >= iou_threshold:
                best_iou = iou
                best_match = pred
                best_pi = pi
        
        if best_match is not None:
            matched.append(EventMatch(gt_event=gt, pred_event=best_match, temporal_iou=best_iou))
            matched_gt_indices.add(gi)
            matched_pred_indices.add(best_pi)
    
    # Collect unmatched events
    unmatched_gt = [gt_events[i] for i in range(len(gt_events)) 
                    if i not in matched_gt_indices and TAXONOMY_MAPPING.get(gt_events[i].event_type)]
    unmatched_pred = [pred_events[i] for i in range(len(pred_events))
                      if i not in matched_pred_indices and PRED_TO_GT_MAPPING.get(pred_events[i].event_type)]
    
    return matched, unmatched_gt, unmatched_pred


def evaluate_video(gt_events: List[GTEvent], pred_events: List[PredEvent],
                   iou_threshold: float = 0.3) -> Dict:
    """Evaluate predictions for a single video."""
    matched, unmatched_gt, unmatched_pred = match_events(gt_events, pred_events, iou_threshold)
    
    # Count by event type
    gt_counts = defaultdict(int)
    pred_counts = defaultdict(int)
    matched_counts = defaultdict(int)
    
    for gt in gt_events:
        if TAXONOMY_MAPPING.get(gt.event_type):  # Only count mappable events
            gt_counts[gt.event_type] += 1
    
    for pred in pred_events:
        if PRED_TO_GT_MAPPING.get(pred.event_type):  # Only count mappable events
            pred_counts[pred.event_type] += 1
    
    for m in matched:
        matched_counts[m.gt_event.event_type] += 1
    
    # Compute metrics
    total_gt_mappable = sum(gt_counts.values())
    total_pred_mappable = sum(pred_counts.values())
    total_matched = len(matched)
    
    recall = total_matched / total_gt_mappable if total_gt_mappable > 0 else 0.0
    precision = total_matched / total_pred_mappable if total_pred_mappable > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    avg_iou = sum(m.temporal_iou for m in matched) / len(matched) if matched else 0.0
    
    return {
        'gt_counts': dict(gt_counts),
        'pred_counts': dict(pred_counts),
        'matched_counts': dict(matched_counts),
        'total_gt_mappable': total_gt_mappable,
        'total_pred_mappable': total_pred_mappable,
        'total_matched': total_matched,
        'recall': recall,
        'precision': precision,
        'f1': f1,
        'avg_iou': avg_iou,
        'unmatched_gt': len(unmatched_gt),
        'unmatched_pred': len(unmatched_pred),
    }


def evaluate_all(gt_annotations: Dict[str, List[GTEvent]], 
                 predictions: Dict[str, List[PredEvent]],
                 iou_threshold: float = 0.3) -> Dict:
    """Evaluate all videos and compute aggregate metrics."""
    per_video_results = {}
    
    # Aggregate counters
    total_gt = defaultdict(int)
    total_pred = defaultdict(int)
    total_matched = defaultdict(int)
    all_ious = []
    
    common_videos = set(gt_annotations.keys()) & set(predictions.keys())
    
    for video_id in sorted(common_videos):
        gt_events = gt_annotations[video_id]
        pred_events = predictions[video_id]
        
        result = evaluate_video(gt_events, pred_events, iou_threshold)
        per_video_results[video_id] = result
        
        # Aggregate
        for k, v in result['gt_counts'].items():
            total_gt[k] += v
        for k, v in result['pred_counts'].items():
            total_pred[k] += v
        for k, v in result['matched_counts'].items():
            total_matched[k] += v
        if result['avg_iou'] > 0:
            all_ious.append(result['avg_iou'])
    
    # Compute aggregate metrics
    total_gt_sum = sum(total_gt.values())
    total_pred_sum = sum(total_pred.values())
    total_matched_sum = sum(total_matched.values())
    
    aggregate_recall = total_matched_sum / total_gt_sum if total_gt_sum > 0 else 0.0
    aggregate_precision = total_matched_sum / total_pred_sum if total_pred_sum > 0 else 0.0
    aggregate_f1 = (2 * aggregate_precision * aggregate_recall / 
                    (aggregate_precision + aggregate_recall) 
                    if (aggregate_precision + aggregate_recall) > 0 else 0.0)
    
    return {
        'iou_threshold': iou_threshold,
        'num_videos_evaluated': len(common_videos),
        'num_gt_only_videos': len(gt_annotations) - len(common_videos),
        'num_pred_only_videos': len(predictions) - len(common_videos),
        'aggregate': {
            'gt_counts': dict(total_gt),
            'pred_counts': dict(total_pred),
            'matched_counts': dict(total_matched),
            'total_gt': total_gt_sum,
            'total_pred': total_pred_sum,
            'total_matched': total_matched_sum,
            'recall': aggregate_recall,
            'precision': aggregate_precision,
            'f1': aggregate_f1,
            'avg_iou': sum(all_ious) / len(all_ious) if all_ious else 0.0,
        },
        'per_video': per_video_results,
    }


# =============================================================================
# Output Formatting
# =============================================================================

def print_summary(results: Dict):
    """Print evaluation summary."""
    agg = results['aggregate']
    
    print("=" * 70)
    print("GAZE EVENT EVALUATION SUMMARY")
    print("=" * 70)
    print(f"IoU Threshold: {results['iou_threshold']}")
    print(f"Videos Evaluated: {results['num_videos_evaluated']}")
    print()
    
    print("TAXONOMY MAPPING:")
    print("-" * 50)
    for gt_type, pred_types in TAXONOMY_MAPPING.items():
        if pred_types:
            print(f"  GT '{gt_type}' -> Pred {pred_types}")
    print()
    
    print("GROUND TRUTH EVENT COUNTS:")
    print("-" * 50)
    for event_type, count in sorted(agg['gt_counts'].items()):
        matched = agg['matched_counts'].get(event_type, 0)
        pct = matched / count * 100 if count > 0 else 0
        print(f"  {event_type:15s}: {count:5d} events | {matched:4d} matched ({pct:5.1f}%)")
    print(f"  {'TOTAL':15s}: {agg['total_gt']:5d} events | {agg['total_matched']:4d} matched")
    print()
    
    print("PREDICTED EVENT COUNTS (Mappable):")
    print("-" * 50)
    for event_type, count in sorted(agg['pred_counts'].items()):
        print(f"  {event_type:20s}: {count:5d} events")
    print(f"  {'TOTAL':20s}: {agg['total_pred']:5d} events")
    print()
    
    print("AGGREGATE METRICS:")
    print("-" * 50)
    print(f"  Recall:    {agg['recall']:.3f} ({agg['total_matched']}/{agg['total_gt']} GT events matched)")
    print(f"  Precision: {agg['precision']:.3f} ({agg['total_matched']}/{agg['total_pred']} predicted events matched)")
    print(f"  F1 Score:  {agg['f1']:.3f}")
    print(f"  Avg IoU:   {agg['avg_iou']:.3f} (temporal overlap of matched events)")
    print("=" * 70)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Evaluate gaze event predictions against GT")
    parser.add_argument("--gt_dir", type=str, required=True,
                        help="Directory containing NewAnt_*.txt GT files")
    parser.add_argument("--pred_dir", type=str, required=True,
                        help="Directory containing *_sam3rf_gaze_events.json prediction files")
    parser.add_argument("--iou_threshold", type=float, default=0.3,
                        help="IoU threshold for event matching (default: 0.3)")
    parser.add_argument("--min_confidence", type=float, default=0.0,
                        help="Minimum confidence for predictions (default: 0.0, no filtering)")
    
    args = parser.parse_args()
    
    print(f"Loading GT annotations from: {args.gt_dir}")
    gt_annotations = load_all_gt_annotations(args.gt_dir)
    print(f"  Found {len(gt_annotations)} videos with GT annotations")
    
    print(f"Loading predictions from: {args.pred_dir}")
    if args.min_confidence > 0:
        print(f"  Filtering predictions with confidence >= {args.min_confidence}")
    predictions = load_predictions(args.pred_dir, args.min_confidence)
    print(f"  Found {len(predictions)} videos with predictions")
    
    print(f"\nEvaluating with IoU threshold = {args.iou_threshold}...")
    results = evaluate_all(gt_annotations, predictions, args.iou_threshold)
    
    print_summary(results)


if __name__ == "__main__":
    main()
