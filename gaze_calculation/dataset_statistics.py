#!/usr/bin/env python3
"""
Dataset Statistics Generator

Analyzes gaze annotation datasets to generate comprehensive statistics:
- Number of videos
- Average duration
- Total hours
- Event counts by type (5 taxonomies)

Usage:
    python dataset_statistics.py --output stats.txt
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
import numpy as np


def analyze_gaze_data(gaze_dir: str) -> Tuple[int, List[float]]:
    """
    Analyze gaze JSON files to count videos and get durations.
    
    Returns:
        Tuple of (num_videos, list_of_durations_in_seconds)
    """
    gaze_files = list(Path(gaze_dir).glob("*_sam3rf_gaze.json"))
    print(f"  Found {len(gaze_files)} gaze JSON files")
    durations = []
    
    for gaze_file in gaze_files:
        try:
            with open(gaze_file) as f:
                data = json.load(f)
            
            # Get duration from video metadata
            video_fps = data.get("video_fps", 30.0)
            total_frames = data.get("total_frames", 0)
            
            if video_fps > 0:
                duration_sec = total_frames / video_fps
                durations.append(duration_sec)
        except Exception as e:
            print(f"  Warning: Could not read {gaze_file.name}: {e}")
    
    return len(gaze_files), durations


def analyze_event_data(event_dir: str) -> Dict[str, int]:
    """
    Analyze event JSON files to count events by type.
    
    Returns:
        Dict mapping event_type to total count
    """
    event_counts = defaultdict(int)
    event_files = list(Path(event_dir).glob("*_events.json"))
    print(f"  Found {len(event_files)} event JSON files")
    
    for event_file in event_files:
        try:
            with open(event_file) as f:
                data = json.load(f)
            
            # Count events by type
            events = data.get("events", [])
            for event in events:
                event_type = event.get("event_type", "unknown")
                event_counts[event_type] += 1
        except Exception as e:
            print(f"  Warning: Could not read {event_file.name}: {e}")
    
    return dict(event_counts)


def format_duration(seconds: float) -> str:
    """Format duration in human-readable format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def analyze_dataset(base_path: str, dataset_name: str) -> Dict:
    """
    Analyze a single dataset.
    
    Returns:
        Dict with statistics
    """
    print(f"\n{'='*60}")
    print(f"Analyzing {dataset_name}...")
    print(f"{'='*60}")
    
    gaze_videos_dir = os.path.join(base_path, "gaze_videos")
    gaze_data_dir = os.path.join(base_path, "gaze_data")
    event_data_dir = os.path.join(base_path, "event_data")
    
    # Try gaze_videos first, fall back to gaze_data
    if os.path.exists(gaze_videos_dir):
        num_videos, durations = analyze_gaze_data(gaze_videos_dir)
    elif os.path.exists(gaze_data_dir):
        num_videos, durations = analyze_gaze_data(gaze_data_dir)
    else:
        print(f"  Warning: No gaze data directory found for {dataset_name}")
        num_videos, durations = 0, []
    
    # Analyze events
    print(f"  Analyzing event data...")
    if os.path.exists(event_data_dir):
        event_counts = analyze_event_data(event_data_dir)
    else:
        print(f"  Warning: No event data directory found for {dataset_name}")
        event_counts = {}
    
    print(f"  ✓ Completed analysis for {dataset_name}")
    
    # Calculate statistics
    total_duration = sum(durations)
    avg_duration = np.mean(durations) if durations else 0.0
    total_hours = total_duration / 3600.0
    
    return {
        "dataset": dataset_name,
        "num_videos": num_videos,
        "durations": durations,
        "avg_duration_sec": avg_duration,
        "total_duration_sec": total_duration,
        "total_hours": total_hours,
        "event_counts": event_counts,
    }


def print_statistics(stats: List[Dict]):
    """Print formatted statistics to console."""
    
    # Event type order
    event_types = [
        "joint_attention",
        "mutual_gaze", 
        "gaze_following",
        "sudden_gaze_shift",
        "attention_capture"
    ]
    
    lines = []
    lines.append("=" * 120)
    lines.append("GAZE ANNOTATION DATASET STATISTICS")
    lines.append("=" * 120)
    lines.append("")
    
    # Header
    header = f"{'Dataset':<20} {'#Videos':<10} {'Avg Duration':<15} {'Total Hours':<12}"
    for event_type in event_types:
        header += f" {event_type.replace('_', ' ').title():<20}"
    lines.append(header)
    lines.append("-" * 120)
    
    # Per-dataset stats
    total_videos = 0
    total_hours = 0.0
    total_events = defaultdict(int)
    
    for stat in stats:
        dataset = stat["dataset"]
        num_videos = stat["num_videos"]
        avg_dur = stat["avg_duration_sec"]
        total_hrs = stat["total_hours"]
        event_counts = stat["event_counts"]
        
        total_videos += num_videos
        total_hours += total_hrs
        
        line = f"{dataset:<20} {num_videos:<10} {format_duration(avg_dur):<15} {total_hrs:<12.2f}"
        
        for event_type in event_types:
            count = event_counts.get(event_type, 0)
            total_events[event_type] += count
            line += f" {count:<20}"
        
        lines.append(line)
    
    # Total row
    lines.append("-" * 120)
    total_line = f"{'TOTAL':<20} {total_videos:<10} {'-':<15} {total_hours:<12.2f}"
    for event_type in event_types:
        total_line += f" {total_events[event_type]:<20}"
    lines.append(total_line)
    lines.append("=" * 120)
    
    # Summary statistics
    lines.append("")
    lines.append("SUMMARY:")
    lines.append(f"  Total Datasets:        {len(stats)}")
    lines.append(f"  Total Videos:          {total_videos}")
    lines.append(f"  Total Hours:           {total_hours:.2f} hours ({format_duration(total_hours * 3600)})")
    lines.append(f"  Avg Videos/Dataset:    {total_videos / len(stats):.1f}")
    lines.append(f"  Avg Hours/Dataset:     {total_hours / len(stats):.2f}")
    lines.append("")
    lines.append("EVENT BREAKDOWN:")
    total_all_events = sum(total_events.values())
    for event_type in event_types:
        count = total_events[event_type]
        pct = (count / total_all_events * 100) if total_all_events > 0 else 0
        lines.append(f"  {event_type.replace('_', ' ').title():<25} {count:>8} ({pct:>5.1f}%)")
    lines.append(f"  {'TOTAL EVENTS':<25} {total_all_events:>8}")
    lines.append("=" * 120)
    
    # Print to console only
    print("\n".join(lines))


def main():
    print("\n" + "="*80)
    print("DATASET STATISTICS GENERATOR")
    print("="*80)
    
    parser = argparse.ArgumentParser(description="Generate dataset statistics")
    args = parser.parse_args()
    
    # Dataset paths
    datasets = [
        ("/projects/illinois/eng/cs/jrehg/users/arkimjh/gaze_social/avsbench", "avsbench"),
        ("/projects/illinois/eng/cs/jrehg/users/arkimjh/gaze_social/embody3d", "embody3d"),
        ("/projects/illinois/eng/cs/jrehg/users/arkimjh/gaze_social/friendsmmc", "friendsmmc"),
        ("/projects/illinois/eng/cs/jrehg/users/arkimjh/gaze_social/social_gesture", "social_gesture"),
        ("/projects/illinois/eng/cs/jrehg/users/arkimjh/gaze_social/social-iq", "social-iq"),
        ("/projects/illinois/eng/cs/jrehg/users/arkimjh/gaze_social/tvqa", "tvqa"),
    ]
    
    all_stats = []
    
    print(f"\nProcessing {len(datasets)} datasets...\n")
    
    for base_path, dataset_name in datasets:
        if os.path.exists(base_path):
            stat = analyze_dataset(base_path, dataset_name)
            all_stats.append(stat)
        else:
            print(f"\nWarning: Dataset path not found: {base_path}")
    
    # Print statistics
    print(f"\n{'='*80}")
    print("GENERATING FINAL STATISTICS TABLE...")
    print(f"{'='*80}\n")
    print_statistics(all_stats)


if __name__ == "__main__":
    main()
