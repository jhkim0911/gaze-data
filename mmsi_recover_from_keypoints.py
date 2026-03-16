"""
Recover 19 missing MMSI clips using original MMSI keypoint bboxes.

These clips had 0 person detections from SAM but have MMSI keypoint data.
Uses keypoint box + idx directly (no Hungarian matching needed).

Output:
  benchmark/MMSI/videos/{clip}.mp4         - offline (all frames)
  benchmark/Online-MMSI/videos/{clip}.mp4  - online (trimmed to utterance time)
  amongus/mmsi_test_matched_kp_recovery.json
"""

import json
import os
import subprocess

import cv2
import numpy as np
from tqdm import tqdm

BASE = "/projects/illinois/eng/cs/jrehg/users/arkimjh/gaze_social/amongus"
BENCHMARK = "/projects/illinois/eng/cs/jrehg/users/arkimjh/gaze_social/benchmark"
MMSI_BASE = "/projects/illinois/eng/cs/jrehg/users/ajojic2/MMSI"
MANIFEST = f"{BASE}/mmsi_test_manifest.json"
MATCHED_JSON = f"{BASE}/mmsi_test_matched.json"
MATCHED_MISSING_JSON = f"{BASE}/mmsi_test_matched_missing.json"
VIZ_DIR = f"{BENCHMARK}/MMSI/videos"
ONLINE_VIZ_DIR = f"{BENCHMARK}/Online-MMSI/videos"
KEYPOINT_DIRS = {
    "youtube": f"{MMSI_BASE}/keypoints/keypoints_youtube",
    "ego4d": f"{MMSI_BASE}/keypoints/keypoints_ego4d",
}

COLORS = [
    (0, 255, 0),    # Green
    (255, 128, 0),  # Blue-ish
    (0, 128, 255),  # Orange
    (255, 255, 0),  # Cyan
    (255, 0, 255),  # Magenta
    (0, 255, 255),  # Yellow
    (128, 0, 255),  # Purple
    (255, 0, 128),  # Pink
    (128, 255, 0),  # Lime green
    (0, 64, 255),   # Dark orange
]


def draw_person_bbox(frame, bbox_px, label, color, line_thickness=1):
    x1, y1, x2, y2 = map(int, bbox_px)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, line_thickness, cv2.LINE_AA)

    font_scale = 0.6
    font_thickness = 2
    (text_w, text_h), _ = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
    bbox_center_x = (x1 + x2) // 2
    label_x1 = bbox_center_x - (text_w + 6) // 2
    label_x2 = label_x1 + text_w + 6
    label_height = text_h + 8

    if y1 >= label_height:
        label_y1 = y1 - label_height
        label_y2 = y1
        text_y = y1 - 4
    else:
        label_y1 = y1
        label_y2 = y1 + label_height
        text_y = y1 + text_h + 4

    cv2.rectangle(frame, (label_x1, label_y1), (label_x2, label_y2), color, -1)
    cv2.putText(frame, label, (label_x1 + 3, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0),
                font_thickness, cv2.LINE_AA)


def reencode_h264(video_path):
    temp_path = video_path.replace(".mp4", "_temp.mp4")
    try:
        subprocess.run([
            "ffmpeg", "-y", "-i", video_path,
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
            "-pix_fmt", "yuv420p", "-an", temp_path,
        ], capture_output=True, check=True)
        os.replace(temp_path, video_path)
    except subprocess.CalledProcessError:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def get_kp_detections(kp_data, frame_idx):
    """Get keypoint detections at or near the given frame index."""
    for offset in range(0, 25):
        for fi in [frame_idx + offset, frame_idx - offset]:
            if 0 <= fi < len(kp_data) and len(kp_data[fi]) > 0:
                return kp_data[fi]
    return []


def create_viz(clip_path, kp_data, clip_start_sec, player_num,
               output_path, max_frames=None, alpha=0.6):
    cap = cv2.VideoCapture(clip_path)
    if not cap.isOpened():
        return False

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 1.0, (width, height))

    written = 0
    for clip_frame_i in range(total_frames):
        if max_frames is not None and written >= max_frames:
            break

        ret, frame = cap.read()
        if not ret:
            break

        # Clip at 1fps, keypoints at 5fps
        video_time_sec = clip_start_sec + clip_frame_i
        kp_frame_idx = int(5 * video_time_sec)
        detections = get_kp_detections(kp_data, kp_frame_idx)

        overlay = frame.copy()
        for det in detections:
            idx = det["idx"]
            if idx >= player_num:
                continue
            box = det["box"]  # [x, y, w, h]
            bbox_px = [box[0], box[1], box[0] + box[2], box[1] + box[3]]
            label = f"P{idx}"
            color = COLORS[idx % len(COLORS)]
            draw_person_bbox(overlay, bbox_px, label, color)

        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        out.write(frame)
        written += 1

    cap.release()
    out.release()
    reencode_h264(output_path)
    return written > 0


def main():
    manifest = json.load(open(MANIFEST))
    matched = json.load(open(MATCHED_JSON))
    matched_missing = json.load(open(MATCHED_MISSING_JSON))
    matched_clips = {e["clip_name"] for e in matched}

    # Recover: 49 from matched_missing (SAM bbox was bad) + remaining unmatched
    matched_missing_clips = {e["clip_name"] for e in matched_missing}
    to_recover = [e for e in manifest
                  if (e["clip_name"] in matched_missing_clips
                      or e["clip_name"] not in matched_clips)
                  and e.get("kp_positions")]
    print(f"Recovering {len(to_recover)} clips using MMSI keypoints")

    kp_cache = {}
    results = []

    for entry in tqdm(to_recover, desc="Recovering"):
        clip_name = entry["clip_name"]
        dataset = entry["dataset"]
        game_id = entry["game_id"]
        player_num = entry["player_num"]

        # Load keypoint data
        cache_key = f"{dataset}__{game_id}"
        if cache_key not in kp_cache:
            kp_path = os.path.join(KEYPOINT_DIRS[dataset], f"{game_id}.npy")
            kp_cache[cache_key] = np.load(kp_path, allow_pickle=True)
        kp_data = kp_cache[cache_key]

        clip_path = entry["clip_path"]
        viz_path = os.path.join(VIZ_DIR, f"{clip_name}.mp4")
        online_viz_path = os.path.join(ONLINE_VIZ_DIR, f"{clip_name}.mp4")

        online_frames = entry["utt_time_sec"] - entry["clip_start_sec"] + 1

        # Offline
        ok = create_viz(clip_path, kp_data, entry["clip_start_sec"],
                        player_num, viz_path)
        if not ok:
            tqdm.write(f"  [FAIL] {clip_name}")
            continue

        # Online
        create_viz(clip_path, kp_data, entry["clip_start_sec"],
                   player_num, online_viz_path, max_frames=online_frames)

        # Identity matching: idx IS player id
        kp_positions = entry["kp_positions"]
        matching = {str(k): int(k) for k in kp_positions.keys()}

        results.append({
            "clip_name": clip_name,
            "dataset": dataset,
            "game_id": game_id,
            "utt_idx": entry["utt_idx"],
            "utterance": entry["utterance"],
            "tasks": entry["tasks"],
            "player_num": player_num,
            "player_names": entry["player_names"],
            "speaker_player_id": entry["speaker_player_id"],
            "matching": matching,
            "n_pipeline_persons": len(kp_positions),
        })

    out_path = f"{BASE}/mmsi_test_matched_kp_recovery.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nRecovered: {len(results)} / {len(to_recover)}")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
