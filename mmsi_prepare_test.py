"""
MMSI Test Set Preparation: Extract per-utterance video clips for gaze pipeline.

Step 1 (this script):
  For each test game utterance, extract ~5s clip at 1fps from the per-game video.
  Saves clips to output dir and a manifest JSON for downstream processing.

Step 2 (after running gaze pipeline on the clips):
  Run mmsi_match_players.py to match pipeline person_ids to MMSI Player IDs.

Usage:
    python mmsi_prepare_test.py [--dataset youtube|ego4d|all]
"""

import argparse
import json
import os
import re
import subprocess

import numpy as np
from tqdm import tqdm

# ── Paths ──
MMSI_BASE = "/projects/illinois/eng/cs/jrehg/users/ajojic2/MMSI"

DATASETS = {
    "youtube": {
        "split_file": f"{MMSI_BASE}/datasets/youtube/data_split.json",
        "meta_dir": f"{MMSI_BASE}/datasets/youtube/meta_data",
        "keypoint_dir": f"{MMSI_BASE}/keypoints/keypoints_youtube",
        "transcript_dir": f"{MMSI_BASE}/datasets/youtube/transcripts/anonymized",
        "transcript_labeled_dir": f"{MMSI_BASE}/datasets/youtube/transcripts/anonymized_labeled",
        "video_dir": "/projects/illinois/eng/cs/jrehg/users/arkimjh/gaze_social/amongus/Youtube",
        "name_sep": "#",  # game_id uses # → filename uses space
    },
    "ego4d": {
        "split_file": f"{MMSI_BASE}/datasets/ego4d/data_split.json",
        "meta_dir": f"{MMSI_BASE}/datasets/ego4d/meta_data",
        "keypoint_dir": f"{MMSI_BASE}/keypoints/keypoints_ego4d",
        "transcript_dir": f"{MMSI_BASE}/datasets/ego4d/transcripts/anonymized",
        "transcript_labeled_dir": f"{MMSI_BASE}/datasets/ego4d/transcripts/anonymized_labeled",
        "video_dir": "/projects/illinois/eng/cs/jrehg/users/arkimjh/gaze_social/amongus/Ego4d",
        "name_sep": None,  # ego4d names don't need conversion
    },
}

OUTPUT_BASE = "/projects/illinois/eng/cs/jrehg/users/arkimjh/gaze_social/amongus"

# Clip parameters
CLIP_BEFORE_SEC = 2  # seconds before utterance
CLIP_AFTER_SEC = 3   # seconds after utterance (total ~5s like MMSI's 3.2s but with buffer)
SAMPLE_FPS = 1


SECOND_PRONOUNS = ["you", "your"]
THIRD_PRONOUNS = ["he", "his", "him", "she", "her"]


def parse_utterance_time(line):
    """Parse (MM:SS) timestamp from utterance line. Returns seconds."""
    match = re.search(r"\((\d+):(\d+)\)", line)
    if match:
        return int(match.group(1)) * 60 + int(match.group(2))
    return None


def has_task_annotation(utt, utt_labeled, player_num):
    """
    Check if an utterance is a test data point for any task (STI/PCR/MPP).
    Uses the exact same logic as MMSI dataloader.process_word().
    Returns list of tasks this utterance is relevant to.
    """
    words = utt.split()
    lab_words = utt_labeled.split()
    is_player = words[0].startswith("[Player")
    if not is_player:
        return []

    tasks = []
    utterance_involved = False
    for word_i, word in enumerate(words):
        # STI: second person pronoun + labeled target at end
        if (word.lower() in SECOND_PRONOUNS
                and utt_labeled.rstrip().endswith("]")
                and word_i != 0 and not utterance_involved):
            brackets = re.findall(r"\[(.*?)\]", utt_labeled)
            names_in_bracket = [
                name in [f"Player{i}" for i in range(player_num)]
                for name in brackets[-1].split()
            ]
            if names_in_bracket.count(True) == 1:
                tasks.append("STI")
                utterance_involved = True

        # PCR: third person pronoun with Player annotation
        if (any(p in word.lower() for p in THIRD_PRONOUNS)
                and word_i != 0 and word_i < len(lab_words)
                and "Player" in lab_words[word_i]):
            tasks.append("PCR")

        # MPP: [PlayerX] mention (not speaker position)
        if word.startswith("[Player") and word_i != 0:
            tasks.append("MPP")

    return tasks


def load_test_split(dataset):
    with open(DATASETS[dataset]["split_file"]) as f:
        return json.load(f)["test"]


def load_meta(dataset, game_id):
    cfg = DATASETS[dataset]
    if dataset == "youtube":
        video_name = game_id.rsplit("_", 1)[0]
        game_key = game_id.rsplit("_", 1)[1]
    else:
        # ego4d: UUID_GameX_CameraY → meta file is UUID.json, key is GameX
        parts = game_id.split("_")
        video_name = parts[0]
        game_key = parts[1]

    meta_path = os.path.join(cfg["meta_dir"], f"{video_name}.json")
    with open(meta_path) as f:
        meta = json.load(f)
    return meta[game_key]


def load_transcripts(dataset, game_id):
    """Load anonymized and labeled transcript lines."""
    cfg = DATASETS[dataset]
    with open(os.path.join(cfg["transcript_dir"], f"{game_id}.txt")) as f:
        anon = [l for l in f.read().strip().split("\n") if l]
    with open(os.path.join(cfg["transcript_labeled_dir"], f"{game_id}.txt")) as f:
        labeled = [l for l in f.read().strip().split("\n") if l]
    return anon, labeled


def get_video_path(dataset, game_id):
    cfg = DATASETS[dataset]
    if cfg["name_sep"]:
        filename = game_id.replace(cfg["name_sep"], " ") + ".mp4"
    else:
        filename = game_id + ".mp4"
    return os.path.join(cfg["video_dir"], filename)


def get_video_duration(video_path):
    r = subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
         "-of", "csv=p=0", video_path],
        capture_output=True, text=True,
    )
    return float(r.stdout.strip())


def extract_clip(video_path, start_sec, duration_sec, output_path):
    """Extract a clip at 1fps for gaze pipeline processing."""
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(max(0, start_sec)),
        "-i", video_path,
        "-t", str(duration_sec),
        "-vf", "fps=1",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "18",
        "-an",
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0


def get_keypoint_positions_at_time(dataset, game_id, time_sec, player_num):
    """
    Get MMSI keypoint bbox centers at a specific timestamp.
    Keypoints are at 5fps, so frame_idx = 5 * time_sec.
    Returns dict: {player_idx: (norm_cx, norm_cy)}
    """
    cfg = DATASETS[dataset]
    kp_path = os.path.join(cfg["keypoint_dir"], f"{game_id}.npy")
    if not os.path.exists(kp_path):
        return {}

    kp_data = np.load(kp_path, allow_pickle=True)
    frame_idx = min(int(5 * time_sec), len(kp_data) - 1)

    # Try a window around the target frame to find detections
    positions = {}
    for offset in range(0, 50):
        for fi in [frame_idx + offset, frame_idx - offset]:
            if 0 <= fi < len(kp_data) and len(kp_data[fi]) > 0:
                # Get image dimensions from max coords
                max_x = max(d["box"][0] + d["box"][2] for d in kp_data[fi])
                max_y = max(d["box"][1] + d["box"][3] for d in kp_data[fi])
                if max_x == 0 or max_y == 0:
                    continue
                for det in kp_data[fi]:
                    idx = det["idx"]
                    if idx < player_num and idx not in positions:
                        box = det["box"]
                        cx = (box[0] + box[2] / 2) / max_x
                        cy = (box[1] + box[3] / 2) / max_y
                        positions[idx] = (float(cx), float(cy))
                if len(positions) >= player_num:
                    return positions
    return positions


def process_dataset(dataset, skip_existing=True):
    test_games = load_test_split(dataset)
    print(f"\n{'='*60}")
    print(f"Dataset: {dataset}, test games: {len(test_games)}")
    print(f"{'='*60}")

    clip_dir = os.path.join(OUTPUT_BASE, f"mmsi_test_clips_{dataset}")
    os.makedirs(clip_dir, exist_ok=True)

    manifest = []
    total_clips = 0
    skipped_clips = 0

    for game_id in tqdm(sorted(test_games), desc=f"[{dataset}]"):
        meta = load_meta(dataset, game_id)
        player_names = meta["playerNames"]
        player_num = len(player_names)
        anon_lines, labeled_lines = load_transcripts(dataset, game_id)
        video_path = get_video_path(dataset, game_id)

        if not os.path.exists(video_path):
            tqdm.write(f"  [SKIP] {game_id}: video not found")
            continue

        video_dur = get_video_duration(video_path)

        # Process only utterances that are actual test data points
        for utt_idx, (line, line_labeled) in enumerate(zip(anon_lines, labeled_lines)):
            tasks = has_task_annotation(line, line_labeled, player_num)
            if not tasks:
                continue

            time_sec = parse_utterance_time(line)
            if time_sec is None:
                continue

            clip_name = f"{game_id}_utt{utt_idx:04d}"
            clip_path = os.path.join(clip_dir, f"{clip_name}.mp4")

            # Extract clip
            start = max(0, time_sec - CLIP_BEFORE_SEC)
            end = min(video_dur, time_sec + CLIP_AFTER_SEC)
            duration = end - start

            if skip_existing and os.path.exists(clip_path):
                skipped_clips += 1
            else:
                ok = extract_clip(video_path, start, duration, clip_path)
                if not ok:
                    tqdm.write(f"  [FAIL] {clip_name}")
                    continue
                total_clips += 1

            # Get MMSI keypoint positions at this utterance time
            kp_pos = get_keypoint_positions_at_time(dataset, game_id, time_sec, player_num)

            # Parse speaker
            speaker_match = re.search(r"\[Player(\d+)\]", line.split(":")[0])
            speaker_id = int(speaker_match.group(1)) if speaker_match else None

            manifest.append({
                "clip_name": clip_name,
                "clip_path": clip_path,
                "dataset": dataset,
                "game_id": game_id,
                "utt_idx": utt_idx,
                "utterance": line,
                "utterance_labeled": line_labeled,
                "tasks": tasks,
                "utt_time_sec": time_sec,
                "clip_start_sec": start,
                "clip_duration_sec": duration,
                "speaker_player_id": speaker_id,
                "player_names": player_names,
                "player_num": player_num,
                "kp_positions": {str(k): list(v) for k, v in kp_pos.items()},
            })

    print(f"  Clips: {total_clips} created, {skipped_clips} skipped (exist)")
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="all", choices=["youtube", "ego4d", "all"])
    parser.add_argument("--no_skip", action="store_true")
    args = parser.parse_args()

    datasets = ["youtube", "ego4d"] if args.dataset == "all" else [args.dataset]
    all_manifest = []

    for ds in datasets:
        manifest = process_dataset(ds, skip_existing=not args.no_skip)
        all_manifest.extend(manifest)

    # Save manifest
    manifest_path = os.path.join(OUTPUT_BASE, "mmsi_test_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(all_manifest, f, indent=2, ensure_ascii=False)

    print(f"\nTotal utterance clips: {len(all_manifest)}")
    print(f"Manifest saved to: {manifest_path}")
    print(f"\nNext step: run gaze pipeline on clips in:")
    for ds in datasets:
        print(f"  {OUTPUT_BASE}/mmsi_test_clips_{ds}/")


if __name__ == "__main__":
    main()
