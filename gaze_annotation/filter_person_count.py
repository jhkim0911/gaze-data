#!/usr/bin/env python3
"""
Person Count Filtering Script for Gaze Annotation Outputs

Filters gaze annotation JSON files based on the number of unique persons detected.
Useful for removing videos with too many people for practical annotation.

Usage:
    # Analyze person count distribution
    python filter_person_count.py --input_dir /path/to/gaze_videos --mode analyze
    
    # Filter files with > 6 persons to separate directory
    python filter_person_count.py --input_dir /path/to/gaze_videos \\
        --output_dir /path/to/filtered_out --max_persons 6 --mode filter
    
    # Delete files with > 8 persons (requires confirmation)
    python filter_person_count.py --input_dir /path/to/gaze_videos \\
        --max_persons 8 --mode delete --yes
"""

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import Counter
from tqdm import tqdm
import sys


def count_persons(json_path: str, min_face_detection_pct: Optional[float] = None) -> int:
    """
    Count unique persons in gaze annotation JSON.
    
    Args:
        json_path: Path to gaze annotation JSON file
        min_face_detection_pct: If provided, only count persons with 
                                face_detection_pct >= this threshold
    
    Returns:
        Number of unique persons
    """
    try:
        with open(json_path) as f:
            data = json.load(f)
        
        persons_summary = data.get("persons_summary", {})
        
        if min_face_detection_pct is None:
            return len(persons_summary)
        
        # Filter by face detection percentage
        valid_persons = 0
        for person_id, info in persons_summary.items():
            face_pct = info.get("face_detection_pct", 0.0)
            if face_pct >= min_face_detection_pct:
                valid_persons += 1
        
        return valid_persons
        
    except (json.JSONDecodeError, FileNotFoundError, KeyError) as e:
        print(f"  WARNING: Could not read {json_path}: {e}")
        return 0


def get_associated_files(json_path: str) -> List[str]:
    """
    Find associated files for a gaze annotation JSON.
    
    Looks for:
    - Visualization video (*_sam3rf_viz.mp4)
    
    Args:
        json_path: Path to JSON file
    
    Returns:
        List of existing associated file paths
    """
    base = json_path.replace("_sam3rf_gaze.json", "")
    
    associated = []
    viz_video = f"{base}_sam3rf_viz.mp4"
    if os.path.exists(viz_video):
        associated.append(viz_video)
    
    return associated


def analyze_person_counts(
    input_dir: str,
    pattern: str = "*_sam3rf_gaze.json",
    min_face_detection_pct: Optional[float] = None,
    recursive: bool = True
) -> Dict[str, int]:
    """
    Analyze person count distribution across all JSON files.
    
    Args:
        input_dir: Directory containing JSON files
        pattern: Filename pattern to match
        min_face_detection_pct: Minimum face detection % to count as valid person
        recursive: Search subdirectories
    
    Returns:
        Dict mapping file paths to person counts
    """
    input_path = Path(input_dir)
    
    if recursive:
        json_files = list(input_path.rglob(pattern))
    else:
        json_files = list(input_path.glob(pattern))
    
    print(f"Found {len(json_files)} JSON files in {input_dir}")
    
    if min_face_detection_pct:
        print(f"  (Filtering persons with face_detection_pct >= {min_face_detection_pct}%)")
    
    counts = {}
    for json_file in tqdm(json_files, desc="Analyzing JSON files", unit="file"):
        person_count = count_persons(str(json_file), min_face_detection_pct)
        counts[str(json_file)] = person_count
    
    return counts


def print_statistics(counts: Dict[str, int], max_persons: Optional[int] = None):
    """Print statistics about person count distribution."""
    if not counts:
        print("\nNo files found.")
        return
    
    values = list(counts.values())
    total = len(values)
    
    print("\n" + "=" * 70)
    print("PERSON COUNT STATISTICS")
    print("=" * 70)
    print(f"Total files:           {total}")
    print(f"Min persons:           {min(values)}")
    print(f"Max persons:           {max(values)}")
    print(f"Mean persons:          {sum(values) / total:.1f}")
    print(f"Median persons:        {sorted(values)[total // 2]}")
    
    if max_persons:
        exceeding = sum(1 for v in values if v > max_persons)
        within = total - exceeding
        print(f"\nWith max_persons={max_persons}:")
        print(f"  Files to keep:       {within} ({100 * within / total:.1f}%)")
        print(f"  Files to filter:     {exceeding} ({100 * exceeding / total:.1f}%)")
    
    # Distribution histogram
    print("\nDistribution:")
    counter = Counter(values)
    for count in sorted(counter.keys()):
        bar = "█" * (counter[count] // max(1, total // 50))
        print(f"  {count:2d} persons: {counter[count]:4d} files {bar}")
    
    # Show files exceeding threshold
    if max_persons:
        exceeding_files = [(path, count) for path, count in counts.items() if count > max_persons]
        if exceeding_files:
            print(f"\nFiles exceeding {max_persons} persons:")
            for path, count in sorted(exceeding_files, key=lambda x: -x[1])[:20]:
                filename = Path(path).name
                print(f"  {count:2d} persons: {filename}")
            if len(exceeding_files) > 20:
                print(f"  ... and {len(exceeding_files) - 20} more")
    
    print("=" * 70)


def filter_files(
    input_dir: str,
    output_dir: Optional[str],
    max_persons: int,
    mode: str,
    min_face_detection_pct: Optional[float] = None,
    pattern: str = "*_sam3rf_gaze.json",
    recursive: bool = True,
    require_confirm: bool = True
) -> Tuple[List[str], List[Tuple[str, int]]]:
    """
    Filter JSON files based on person count.
    
    Args:
        input_dir: Directory containing JSON files
        output_dir: Directory to move filtered files (for 'filter' mode)
        max_persons: Maximum number of persons to keep
        mode: 'analyze', 'filter', or 'delete'
        min_face_detection_pct: Minimum face detection % to count as valid person
        pattern: Filename pattern to match
        recursive: Search subdirectories
        require_confirm: Require user confirmation for delete mode
    
    Returns:
        Tuple of (kept_files, filtered_files_with_counts)
    """
    counts = analyze_person_counts(input_dir, pattern, min_face_detection_pct, recursive)
    
    kept = []
    filtered = []
    
    for path, count in counts.items():
        if count <= max_persons:
            kept.append(path)
        else:
            filtered.append((path, count))
    
    print_statistics(counts, max_persons)
    
    if mode == "analyze":
        print(f"\n[ANALYZE MODE] No files modified.")
        return kept, filtered
    
    if not filtered:
        print(f"\nNo files exceed {max_persons} persons. Nothing to do.")
        return kept, filtered
    
    # Confirm deletion if required
    if mode == "delete" and require_confirm:
        print(f"\n⚠️  WARNING: About to DELETE {len(filtered)} files and their associated videos!")
        response = input("Type 'yes' to confirm deletion: ")
        if response.lower() != "yes":
            print("Deletion cancelled.")
            return kept, filtered
    
    # Process filtered files
    print(f"\nProcessing {len(filtered)} files...")
    
    for path, count in filtered:
        json_path = Path(path)
        associated = get_associated_files(str(json_path))
        all_files = [json_path] + [Path(f) for f in associated]
        
        if mode == "filter":
            # Move to output directory
            output_path = Path(output_dir)
            
            # Preserve relative directory structure
            rel_path = json_path.relative_to(input_dir)
            dest_dir = output_path / rel_path.parent
            dest_dir.mkdir(parents=True, exist_ok=True)
            
            for file_path in all_files:
                if file_path.exists():
                    rel_file = file_path.relative_to(input_dir)
                    dest_file = output_path / rel_file
                    dest_file.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(file_path), str(dest_file))
                    print(f"  Moved: {file_path.name} → {dest_file}")
        
        elif mode == "delete":
            # Delete files
            for file_path in all_files:
                if file_path.exists():
                    file_path.unlink()
                    print(f"  Deleted: {file_path.name}")
    
    print(f"\n✓ Processed {len(filtered)} files ({mode} mode)")
    return kept, filtered


def export_to_csv(counts: Dict[str, int], output_path: str):
    """Export person counts to CSV for analysis."""
    import csv
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "person_count"])
        
        for path, count in sorted(counts.items()):
            filename = Path(path).name
            writer.writerow([filename, count])
    
    print(f"\nExported counts to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Filter gaze annotation JSON files by person count",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze distribution
  python filter_person_count.py --input_dir ./gaze_videos --mode analyze
  
  # Filter to separate directory
  python filter_person_count.py --input_dir ./gaze_videos \\
      --output_dir ./filtered_out --max_persons 6 --mode filter
  
  # Only count persons with good face detection
  python filter_person_count.py --input_dir ./gaze_videos \\
      --min_face_pct 50 --mode analyze
        """)
    
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Directory containing gaze annotation JSON files")
    parser.add_argument("--output_dir", type=str,
                        help="Directory to move filtered files (required for 'filter' mode)")
    parser.add_argument("--max_persons", type=int, default=6,
                        help="Maximum number of persons to keep (default: 6)")
    parser.add_argument("--mode", type=str, default="analyze",
                        choices=["analyze", "filter", "delete"],
                        help="Operation mode: analyze (default), filter, or delete")
    parser.add_argument("--min_face_pct", type=float, default=None,
                        help="Minimum face_detection_pct to count as valid person (0-100)")
    parser.add_argument("--pattern", type=str, default="*_sam3rf_gaze.json",
                        help="Filename pattern to match (default: *_sam3rf_gaze.json)")
    parser.add_argument("--no_recursive", action="store_true",
                        help="Don't search subdirectories")
    parser.add_argument("--yes", action="store_true",
                        help="Skip confirmation prompt for delete mode")
    parser.add_argument("--export_csv", type=str,
                        help="Export person counts to CSV file")
    
    args = parser.parse_args()
    
    # Validate arguments
    if not os.path.isdir(args.input_dir):
        print(f"Error: Input directory not found: {args.input_dir}")
        sys.exit(1)
    
    if args.mode == "filter" and not args.output_dir:
        print("Error: --output_dir required for 'filter' mode")
        sys.exit(1)
    
    if args.mode == "filter" and os.path.exists(args.output_dir):
        if os.path.samefile(args.input_dir, args.output_dir):
            print("Error: output_dir cannot be the same as input_dir")
            sys.exit(1)
    
    # Run filtering
    kept, filtered = filter_files(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        max_persons=args.max_persons,
        mode=args.mode,
        min_face_detection_pct=args.min_face_pct,
        pattern=args.pattern,
        recursive=not args.no_recursive,
        require_confirm=not args.yes
    )
    
    # Export to CSV if requested
    if args.export_csv:
        counts = analyze_person_counts(
            args.input_dir, 
            args.pattern, 
            args.min_face_pct,
            not args.no_recursive
        )
        export_to_csv(counts, args.export_csv)


if __name__ == "__main__":
    main()
