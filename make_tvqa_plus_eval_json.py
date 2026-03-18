"""Create TVQA+ evaluation JSON with original character names and subtitles.

Reads the preprocessed validation JSON, computes absolute timestamps for each
clip by summing frame counts, extracts matching SRT subtitles, and outputs a
clean eval JSON.

Output fields per entry:
  qid, vid_name, ts, q, a0-a4, answer_idx, subtitle
"""

import json
import os
import re
from collections import defaultdict

DATA_PATH = "/projects/illinois/eng/cs/jrehg/users/arkimjh/gaze_social/tvqa/penxiang/tvqa_plus_valid_preprocessed.json"
FRAMES_BASE = "/projects/illinois/eng/cs/jrehg/users/arkimjh/gaze_social/tvqa/penxiang/frames_hq/bbt_frames"
SRT_DIR = "/projects/illinois/eng/cs/jrehg/users/arkimjh/gaze_social/benchmark/TVQA+/tvqa_subtitles/bbt"
OUT_PATH = "/projects/illinois/eng/cs/jrehg/users/arkimjh/gaze_social/benchmark/original_tvqa+/tvqa_plus_eval.json"


def parse_srt(path):
    """Parse SRT file into list of (start_sec, end_sec, text)."""
    entries = []
    with open(path, encoding="utf-8", errors="replace") as f:
        content = f.read()
    blocks = re.split(r"\n\s*\n", content.strip())
    for block in blocks:
        lines = block.strip().split("\n")
        if len(lines) < 2:
            continue
        # Find timestamp line
        ts_line = None
        for i, line in enumerate(lines):
            if "-->" in line:
                ts_line = line
                text_lines = lines[i + 1:]
                break
        if not ts_line:
            continue
        m = re.match(
            r"(\d+):(\d+):(\d+)[,.](\d+)\s*-->\s*(\d+):(\d+):(\d+)[,.](\d+)",
            ts_line.strip(),
        )
        if not m:
            continue
        g = [int(x) for x in m.groups()]
        start = g[0] * 3600 + g[1] * 60 + g[2] + g[3] / 1000
        end = g[4] * 3600 + g[5] * 60 + g[6] + g[7] / 1000
        text = " ".join(l.strip() for l in text_lines if l.strip())
        # Strip HTML tags
        text = re.sub(r"<[^>]+>", "", text)
        if text:
            entries.append((start, end, text))
    return entries


def get_srt_path(season, episode):
    """Get SRT file path for a BBT episode."""
    return os.path.join(SRT_DIR, f"season_{season}", f"episode_{episode}.srt")


def extract_subtitles(srt_entries, abs_start, abs_end):
    """Extract subtitle lines overlapping with [abs_start, abs_end]."""
    lines = []
    for s, e, text in srt_entries:
        if s < abs_end and e > abs_start:
            lines.append(text)
    return lines


def main():
    with open(DATA_PATH) as f:
        data = json.load(f)
    print(f"Loaded {len(data)} QA entries")

    # Group entries by vid_name to compute clip order per episode
    # vid_name format: s01e02_seg02_clip_12
    ep_clips = defaultdict(set)  # (season, episode) -> set of vid_names
    for item in data:
        vn = item["vid_name"]
        m = re.match(r"s(\d+)e(\d+)_", vn)
        if m:
            ep_clips[(int(m.group(1)), int(m.group(2)))].add(vn)

    # Compute absolute start time for each clip
    # Clips are sequential: seg01_clip_00, seg01_clip_01, ..., seg02_clip_00, ...
    clip_abs_start = {}
    for (season, episode), vid_names in sorted(ep_clips.items()):
        sorted_clips = sorted(vid_names)
        cumulative = 0.0
        for vn in sorted_clips:
            frames_dir = os.path.join(FRAMES_BASE, vn)
            if os.path.isdir(frames_dir):
                nframes = len(os.listdir(frames_dir))
            else:
                nframes = 0
            clip_abs_start[vn] = cumulative
            cumulative += nframes / 3.0
        print(f"  s{season:02d}e{episode:02d}: {len(sorted_clips)} clips, {cumulative:.1f}s total")

    # Parse SRT files (cache per episode)
    srt_cache = {}

    # Build eval JSON
    eval_data = []
    sub_found = 0
    for item in data:
        vn = item["vid_name"]
        m = re.match(r"s(\d+)e(\d+)_", vn)
        if not m:
            continue
        season, episode = int(m.group(1)), int(m.group(2))

        # Absolute time range for this QA
        abs_start = clip_abs_start.get(vn, 0.0) + item["ts"][0]
        abs_end = clip_abs_start.get(vn, 0.0) + item["ts"][1]

        # Get subtitles
        srt_key = (season, episode)
        if srt_key not in srt_cache:
            srt_path = get_srt_path(season, episode)
            if os.path.exists(srt_path):
                srt_cache[srt_key] = parse_srt(srt_path)
            else:
                srt_cache[srt_key] = []
                print(f"  WARNING: SRT not found: {srt_path}")

        sub_lines = extract_subtitles(srt_cache[srt_key], abs_start, abs_end)
        if sub_lines:
            sub_found += 1

        entry = {
            "qid": item["qid"],
            "vid_name": vn,
            "ts": item["ts"],
            "q": item["q"],
            "answer_idx": item["answer_idx"],
        }
        for i in range(5):
            entry[f"a{i}"] = item[f"a{i}"]
        entry["subtitle"] = " ".join(sub_lines) if sub_lines else ""

        eval_data.append(entry)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(eval_data, f, indent=2, ensure_ascii=False)

    print(f"\nSaved {len(eval_data)} entries to {OUT_PATH}")
    print(f"Entries with subtitles: {sub_found}/{len(eval_data)}")


if __name__ == "__main__":
    main()
