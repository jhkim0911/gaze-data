#!/usr/bin/env python3
"""
Event Statistics Analyzer

Analyzes event statistics across multiple datasets and confidence thresholds.

Usage:
    python event_stats_analyzer.py --base_dir /path/to/gaze_social
"""

import argparse
import json
import os
import glob
from collections import defaultdict
from typing import Dict, List


DATASETS = [
    'avsbench',
    'embody3d', 
    'friendsmmc',
    'social_gesture',
    'social-iq',
    'tvqa',
    'werewolf',
]

CONFIDENCE_THRESHOLDS = [0.0, 0.6, 0.7, 0.8, 0.9]

EVENT_TYPES = [
    'sudden_gaze_shift',
    'joint_attention',
    'gaze_following',
    'attention_capture',
    'mutual_gaze',
]


def load_events(event_dir: str, min_confidence: float = 0.0) -> Dict:
    """Load all events from a directory, optionally filtering by confidence."""
    event_files = glob.glob(os.path.join(event_dir, '*_events.json'))
    
    stats = {
        'num_videos': 0,
        'total_events': 0,
        'event_counts': defaultdict(int),
        'videos_with_events': 0,
    }
    
    for filepath in event_files:
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError):
            continue
        
        stats['num_videos'] += 1
        
        video_event_count = 0
        for event in data.get('events', []):
            conf = event.get('confidence', 1.0)
            if conf >= min_confidence:
                event_type = event.get('event_type', 'unknown')
                stats['event_counts'][event_type] += 1
                stats['total_events'] += 1
                video_event_count += 1
        
        if video_event_count > 0:
            stats['videos_with_events'] += 1
    
    return stats


def analyze_all_datasets(base_dir: str) -> Dict:
    """Analyze all datasets across all confidence thresholds."""
    results = {}
    
    for dataset in DATASETS:
        event_dir = os.path.join(base_dir, dataset, 'event_data')
        
        if not os.path.exists(event_dir):
            print(f"  Skipping {dataset}: event_data not found")
            continue
        
        results[dataset] = {}
        
        for conf_thresh in CONFIDENCE_THRESHOLDS:
            stats = load_events(event_dir, conf_thresh)
            results[dataset][conf_thresh] = stats
    
    return results


def print_summary(results: Dict):
    """Print formatted summary of results."""
    
    # Header
    print("\n" + "=" * 120)
    print("EVENT STATISTICS ACROSS DATASETS AND CONFIDENCE THRESHOLDS")
    print("=" * 120)
    
    # Summary table header
    conf_labels = [f"conf≥{c}" if c > 0 else "all" for c in CONFIDENCE_THRESHOLDS]
    header = f"{'Dataset':<20} {'Videos':>8} | " + " | ".join([f"{l:>10}" for l in conf_labels])
    print("\n" + header)
    print("-" * len(header))
    
    # Per dataset rows
    totals = {c: 0 for c in CONFIDENCE_THRESHOLDS}
    total_videos = 0
    
    for dataset in DATASETS:
        if dataset not in results:
            continue
        
        num_videos = results[dataset][0.0]['num_videos']
        total_videos += num_videos
        
        counts = []
        for conf in CONFIDENCE_THRESHOLDS:
            total = results[dataset][conf]['total_events']
            counts.append(f"{total:>10}")
            totals[conf] += total
        
        row = f"{dataset:<20} {num_videos:>8} | " + " | ".join(counts)
        print(row)
    
    # Totals row
    print("-" * len(header))
    total_counts = [f"{totals[c]:>10}" for c in CONFIDENCE_THRESHOLDS]
    print(f"{'TOTAL':<20} {total_videos:>8} | " + " | ".join(total_counts))
    
    # Per event type breakdown for each dataset
    print("\n" + "=" * 120)
    print("PER EVENT TYPE STATISTICS")
    print("=" * 120)
    
    for dataset in DATASETS:
        if dataset not in results:
            continue
        
        num_videos = results[dataset][0.0]['num_videos']
        print(f"\n{'─' * 80}")
        print(f" {dataset.upper()} ({num_videos} videos)")
        print(f"{'─' * 80}")
        
        # Header for event types
        print(f"\n  {'Event Type':<22} | " + " | ".join([f"{l:>8}" for l in conf_labels]))
        print("  " + "-" * 75)
        
        for event_type in EVENT_TYPES:
            counts = []
            for conf in CONFIDENCE_THRESHOLDS:
                count = results[dataset][conf]['event_counts'].get(event_type, 0)
                counts.append(f"{count:>8}")
            
            print(f"  {event_type:<22} | " + " | ".join(counts))
        
        # Total row for this dataset
        print("  " + "-" * 75)
        totals_row = []
        for conf in CONFIDENCE_THRESHOLDS:
            total = results[dataset][conf]['total_events']
            totals_row.append(f"{total:>8}")
        print(f"  {'TOTAL':<22} | " + " | ".join(totals_row))
        
        # Average per video
        avg_row = []
        for conf in CONFIDENCE_THRESHOLDS:
            total = results[dataset][conf]['total_events']
            avg = total / num_videos if num_videos > 0 else 0
            avg_row.append(f"{avg:>8.2f}")
        print(f"  {'Avg per video':<22} | " + " | ".join(avg_row))
    
    # Grand total by event type
    print("\n" + "=" * 120)
    print("GRAND TOTAL BY EVENT TYPE")
    print("=" * 120)
    
    grand_totals = {conf: defaultdict(int) for conf in CONFIDENCE_THRESHOLDS}
    grand_total_videos = 0
    
    for dataset in DATASETS:
        if dataset not in results:
            continue
        grand_total_videos += results[dataset][0.0]['num_videos']
        for conf in CONFIDENCE_THRESHOLDS:
            for event_type in EVENT_TYPES:
                grand_totals[conf][event_type] += results[dataset][conf]['event_counts'].get(event_type, 0)
    
    print(f"\n  {'Event Type':<22} | " + " | ".join([f"{l:>8}" for l in conf_labels]))
    print("  " + "-" * 75)
    
    for event_type in EVENT_TYPES:
        counts = [f"{grand_totals[conf][event_type]:>8}" for conf in CONFIDENCE_THRESHOLDS]
        print(f"  {event_type:<22} | " + " | ".join(counts))
    
    print("  " + "-" * 75)
    grand_sums = [f"{sum(grand_totals[conf].values()):>8}" for conf in CONFIDENCE_THRESHOLDS]
    print(f"  {'TOTAL':<22} | " + " | ".join(grand_sums))
    
    avg_sums = [f"{sum(grand_totals[conf].values()) / grand_total_videos:>8.2f}" for conf in CONFIDENCE_THRESHOLDS]
    print(f"  {'Avg per video':<22} | " + " | ".join(avg_sums))
    
    print("\n" + "=" * 120)


def main():
    parser = argparse.ArgumentParser(description="Analyze event statistics across datasets")
    parser.add_argument("--base_dir", type=str, 
                        default="/projects/illinois/eng/cs/jrehg/users/arkimjh/gaze_social",
                        help="Base directory containing dataset folders")
    
    args = parser.parse_args()
    
    print(f"Analyzing datasets in: {args.base_dir}")
    print(f"Datasets: {', '.join(DATASETS)}")
    print(f"Confidence thresholds: {CONFIDENCE_THRESHOLDS}")
    
    results = analyze_all_datasets(args.base_dir)
    print_summary(results)


if __name__ == "__main__":
    main()
