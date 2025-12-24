#!/usr/bin/env python3
"""
Event Statistics Calculator

Aggregates event counts from event JSON files and prints statistics.

Usage:
    python event_stats.py --event_dir /path/to/events
"""

import argparse
import glob
import json
import os
from collections import defaultdict


def analyze_events(event_dir: str):
    """Analyze all event JSON files in a directory."""
    event_files = glob.glob(os.path.join(event_dir, "*_events.json"))
    
    if not event_files:
        print(f"No *_events.json files found in: {event_dir}")
        return
    
    # Aggregate counts
    total_counts = defaultdict(int)
    total_events = 0
    files_processed = 0
    files_with_errors = 0
    
    # Per-file stats
    events_per_file = []
    
    for filepath in sorted(event_files):
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            event_counts = data.get("event_counts", {})
            num_events = data.get("num_events", sum(event_counts.values()))
            
            for event_type, count in event_counts.items():
                total_counts[event_type] += count
            
            total_events += num_events
            files_processed += 1
            events_per_file.append(num_events)
            
        except Exception as e:
            files_with_errors += 1
    
    # Print results
    print("=" * 60)
    print("EVENT STATISTICS")
    print("=" * 60)
    print(f"Event directory: {event_dir}")
    print(f"Files processed: {files_processed}")
    print(f"Files with errors: {files_with_errors}")
    print(f"Total events: {total_events}")
    print()
    
    print("Event counts by type:")
    print("-" * 40)
    for event_type, count in sorted(total_counts.items(), key=lambda x: -x[1]):
        pct = count / total_events * 100 if total_events > 0 else 0
        print(f"  {event_type:25s}: {count:6d} ({pct:5.1f}%)")
    print("-" * 40)
    
    if events_per_file:
        import statistics
        print()
        print("Events per file:")
        print(f"  Mean:   {statistics.mean(events_per_file):.1f}")
        print(f"  Median: {statistics.median(events_per_file):.1f}")
        print(f"  Min:    {min(events_per_file)}")
        print(f"  Max:    {max(events_per_file)}")
    
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Calculate event statistics")
    parser.add_argument("--event_dir", type=str, required=True, 
                        help="Directory containing *_events.json files")
    
    args = parser.parse_args()
    
    if not os.path.isdir(args.event_dir):
        print(f"ERROR: Directory not found: {args.event_dir}")
        return
    
    analyze_events(args.event_dir)


if __name__ == "__main__":
    main()
