#!/usr/bin/env python3
"""
Video Clip Generator - Clip videos per QA pair based on time_range.

Reads qa_data JSON files, clips bbox_videos around each QA pair's
time_range with random 1-3s buffer (snapped to 2fps).

Input:
  {dataset}/qa_data/{video}_qa.json       (train, root level)
  {dataset}/qa_data/test/{video}_qa.json  (test split)
  {dataset}/bbox_videos/{video}_bbox_viz.mp4

Output:
  {dataset}/video_info/{video}_qa{idx:04d}.mp4
  {dataset}/video_info/test/{video}_qa{idx:04d}.mp4
  {dataset}/final_qa/{video}_qa{idx:04d}.json
  {dataset}/final_qa/test/{video}_qa{idx:04d}.json

Usage:
    python -m gaze_calculation.video_clip_generator --dataset social_gesture
    python -m gaze_calculation.video_clip_generator --dataset social_gesture --split test
    python -m gaze_calculation.video_clip_generator --all --skip_existing
"""

import argparse
import concurrent.futures
import json
import os
import random
import re
import subprocess
import sys
from typing import Tuple

sys.stdout.reconfigure(line_buffering=True)

DATASETS = [
    'avsbench', 'embody3d', 'friendsmmc', 'social_gesture',
    'social-iq', 'tvqa', 'werewolf',
]
DEFAULT_BASE_PATH = "/projects/illinois/eng/cs/jrehg/users/arkimjh/gaze_social"

BUFFER_MIN = 1.0
BUFFER_MAX = 3.0
FPS = 2.0  # all videos are sampled at 2fps


def snap_to_fps(t: float) -> float:
    """Snap timestamp to nearest 2fps frame boundary (0.5s intervals)."""
    return round(round(t * FPS) / FPS, 2)


def get_video_duration(video_path: str) -> float:
    """Get video duration using ffprobe."""
    cmd = [
        "ffprobe", "-v", "quiet", "-show_entries", "format=duration",
        "-of", "csv=p=0", video_path,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return float(result.stdout.strip())
    except Exception:
        return 0.0


def clip_video(input_path: str, output_path: str, start: float, end: float) -> bool:
    """Clip video segment using ffmpeg."""
    duration = end - start
    if duration <= 0:
        return False
    cmd = [
        "ffmpeg", "-y", "-ss", f"{start:.3f}",
        "-i", input_path, "-t", f"{duration:.3f}",
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-an", "-loglevel", "error",
        output_path,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        return result.returncode == 0
    except Exception:
        return False


def adjust_timestamps_in_text(text: str, offset: float) -> str:
    """Adjust float timestamps in text by subtracting offset."""
    def replace_ts(match):
        val = round(float(match.group(1)) - offset, 1)
        val = max(0.0, val)
        if val == int(val):
            return f"{int(val)}.0"
        return f"{val:.1f}"

    return re.sub(
        r'(?<![A-Za-z])(\d+\.\d+)(?=\s*(?:s(?:econds?)?|,|\s|and\s|$|\?))',
        replace_ts,
        text,
    )


def process_qa_file(
    qa_path: str,
    video_dir: str,
    output_video_dir: str,
    output_qa_dir: str,
    skip_existing: bool,
) -> Tuple[int, int, int]:
    """Process all QA pairs in one qa_data file.

    Returns (clips_made, skipped, errors).
    """
    with open(qa_path) as f:
        qa_data = json.load(f)

    video_name = qa_data.get("video_id", "")
    qa_pairs = qa_data.get("qa_pairs", [])

    if not qa_pairs:
        return 0, 0, 0

    video_path = os.path.join(video_dir, f"{video_name}_bbox_viz.mp4")
    if not os.path.exists(video_path):
        return 0, len(qa_pairs), 0

    vid_duration = get_video_duration(video_path)
    if vid_duration <= 0:
        print(f"  WARN: can't read duration for {video_name}")
        return 0, len(qa_pairs), 0

    os.makedirs(output_video_dir, exist_ok=True)
    os.makedirs(output_qa_dir, exist_ok=True)

    made, skipped, errors = 0, 0, 0

    for idx, qa in enumerate(qa_pairs):
        clip_name = f"{video_name}_qa{idx:04d}"
        out_vid = os.path.join(output_video_dir, f"{clip_name}.mp4")
        out_qa = os.path.join(output_qa_dir, f"{clip_name}.json")

        if skip_existing and os.path.exists(out_vid) and os.path.exists(out_qa):
            skipped += 1
            continue

        tr = qa.get("time_range")
        if not tr or not isinstance(tr, list) or len(tr) < 2:
            skipped += 1
            continue

        t_start, t_end = float(tr[0]), float(tr[1])

        # Random buffer snapped to 0.5s
        buf_before = snap_to_fps(random.uniform(BUFFER_MIN, BUFFER_MAX))
        buf_after = snap_to_fps(random.uniform(BUFFER_MIN, BUFFER_MAX))

        clip_start = snap_to_fps(max(0.0, t_start - buf_before))
        clip_end = snap_to_fps(min(vid_duration, t_end + buf_after))

        if clip_end <= clip_start:
            skipped += 1
            continue

        if not clip_video(video_path, out_vid, clip_start, clip_end):
            errors += 1
            continue

        # Build adjusted QA with timestamps relative to clip
        offset = clip_start
        adj = dict(qa)
        adj["time_range"] = [
            round(t_start - offset, 2),
            round(t_end - offset, 2),
        ]
        adj["question"] = adjust_timestamps_in_text(qa["question"], offset)
        if qa.get("format") == "open_ended":
            if "answer_text" in qa:
                adj["answer_text"] = adjust_timestamps_in_text(qa["answer_text"], offset)
            if "answer" in qa:
                adj["answer"] = adjust_timestamps_in_text(qa["answer"], offset)
        if "options" in qa:
            adj["options"] = [
                adjust_timestamps_in_text(o, offset) for o in qa["options"]
            ]

        final_qa = {
            "video_name": video_name,
            "clip_file": f"{clip_name}.mp4",
            "clip_offset": round(offset, 2),
            "clip_duration": round(clip_end - clip_start, 2),
            "qa": adj,
        }

        with open(out_qa, 'w') as f:
            json.dump(final_qa, f, indent=2)

        made += 1

    return made, skipped, errors


def collect_qa_files(qa_dir: str, split: str):
    """Collect qa_data JSON files for the given split."""
    if split == "train":
        # Root-level JSON files only (not in subdirs)
        return [
            os.path.join(qa_dir, f)
            for f in os.listdir(qa_dir)
            if f.endswith("_qa.json") and os.path.isfile(os.path.join(qa_dir, f))
        ]
    else:
        sub = os.path.join(qa_dir, split)
        if not os.path.isdir(sub):
            return []
        return [
            os.path.join(sub, f)
            for f in os.listdir(sub)
            if f.endswith("_qa.json")
        ]


def process_dataset(
    base_dir: str,
    dataset: str,
    split: str,
    skip_existing: bool,
    max_files: int,
    num_workers: int,
) -> Tuple[int, int, int]:
    """Process a dataset split. Returns (made, skipped, errors)."""
    qa_dir = os.path.join(base_dir, dataset, "qa_data")
    video_dir = os.path.join(base_dir, dataset, "bbox_videos")

    # Output dirs: train at root, test in test/ subdir
    if split == "train":
        out_vid_dir = os.path.join(base_dir, dataset, "video_info")
        out_qa_dir = os.path.join(base_dir, dataset, "final_qa")
    else:
        out_vid_dir = os.path.join(base_dir, dataset, "video_info", split)
        out_qa_dir = os.path.join(base_dir, dataset, "final_qa", split)

    qa_files = sorted(collect_qa_files(qa_dir, split))
    if not qa_files:
        print(f"No qa_data files for {dataset}/{split}")
        return 0, 0, 0

    if max_files > 0:
        random.seed(42)
        qa_files = random.sample(qa_files, min(max_files, len(qa_files)))

    print(f"\n{'='*60}")
    print(f"Dataset: {dataset} | Split: {split} | Files: {len(qa_files)}")
    print(f"Video src: {video_dir}")
    print(f"Output:    {out_vid_dir}")
    print(f"           {out_qa_dir}")
    print(f"{'='*60}")

    total_made, total_skip, total_err = 0, 0, 0

    def _do(args):
        i, qf = args
        if i % 100 == 0:
            print(f"  [{i+1}/{len(qa_files)}] {os.path.basename(qf)}")
        return process_qa_file(qf, video_dir, out_vid_dir, out_qa_dir, skip_existing)

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as ex:
        for m, s, e in ex.map(_do, enumerate(qa_files)):
            total_made += m
            total_skip += s
            total_err += e

    print(f"\n{dataset}/{split}: {total_made} clips, {total_skip} skipped, {total_err} errors")
    return total_made, total_skip, total_err


def main():
    parser = argparse.ArgumentParser(description="Clip videos per QA pair")
    parser.add_argument("--base_dir", type=str, default=DEFAULT_BASE_PATH)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--split", type=str, default="all",
                        choices=["train", "test", "all"],
                        help="Which split to process (default: all)")
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--max_files", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=8)

    args = parser.parse_args()

    if args.all:
        datasets = DATASETS
    elif args.dataset:
        datasets = [args.dataset]
    else:
        parser.print_help()
        sys.exit(1)

    splits = ["train", "test"] if args.split == "all" else [args.split]

    grand_made, grand_skip, grand_err = 0, 0, 0
    for ds in datasets:
        for sp in splits:
            m, s, e = process_dataset(
                args.base_dir, ds, sp, args.skip_existing,
                args.max_files, args.num_workers,
            )
            grand_made += m
            grand_skip += s
            grand_err += e

    print(f"\n{'='*60}")
    print(f"All done: {grand_made} clips, {grand_skip} skipped, {grand_err} errors")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
