"""
Match pipeline person_ids to MMSI PlayerX using Hungarian matching.
Creates visualization videos with P0/P1/... labels (same style as visualize_person_bbox.py).

Input:
  mmsi_test_manifest.json
  mmsi_test_gaze_{youtube,ego4d}/*_sam3rf_gaze.json

Output:
  mmsi_test_matched.json                  - matching results per clip
  benchmark/MMSI/videos/{clip}.mp4        - full annotated videos (offline)
  benchmark/Online-MMSI/videos/{clip}.mp4 - trimmed to utterance time (online)
"""

import json
import os
import subprocess

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

BASE = "/projects/illinois/eng/cs/jrehg/users/arkimjh/gaze_social/amongus"
BENCHMARK = "/projects/illinois/eng/cs/jrehg/users/arkimjh/gaze_social/benchmark"
MANIFEST = f"{BASE}/mmsi_test_manifest.json"
GAZE_DIRS = {
    "youtube": f"{BASE}/mmsi_test_gaze_youtube",
    "ego4d": f"{BASE}/mmsi_test_gaze_ego4d",
}
VIZ_DIR = f"{BENCHMARK}/MMSI/missing_vid"
ONLINE_VIZ_DIR = f"{BENCHMARK}/Online-MMSI/missing_vid"
MATCHED_JSON = f"{BASE}/mmsi_test_matched_missing.json"
MISSING_JSON_DIR = f"{BENCHMARK}/MMSI/json"

# Same color palette as visualize_person_bbox.py
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


def compute_person_centers(gaze_data):
    """Average body bbox center for each person across all frames."""
    sums = {}
    counts = {}
    for frame in gaze_data["frames"]:
        for p in frame.get("persons", []):
            pid = p["person_id"]
            bbox = p.get("body_bbox", [])
            if len(bbox) == 4:
                cx = (bbox[0] + bbox[2]) / 2
                cy = (bbox[1] + bbox[3]) / 2
                if pid not in sums:
                    sums[pid] = [0.0, 0.0]
                    counts[pid] = 0
                sums[pid][0] += cx
                sums[pid][1] += cy
                counts[pid] += 1
    return {pid: [sums[pid][0] / counts[pid], sums[pid][1] / counts[pid]]
            for pid in sums}


def hungarian_match(pipeline_centers, kp_positions):
    """Match pipeline person_ids to MMSI player indices via min-cost assignment."""
    pids = list(pipeline_centers.keys())
    player_idxs = list(kp_positions.keys())
    if not pids or not player_idxs:
        return {}

    cost = np.zeros((len(pids), len(player_idxs)))
    for i, pid in enumerate(pids):
        for j, pidx in enumerate(player_idxs):
            dx = pipeline_centers[pid][0] - kp_positions[pidx][0]
            dy = pipeline_centers[pid][1] - kp_positions[pidx][1]
            cost[i, j] = np.sqrt(dx**2 + dy**2)

    row_ind, col_ind = linear_sum_assignment(cost)
    return {pids[r]: int(player_idxs[c]) for r, c in zip(row_ind, col_ind)}


# ── Drawing (identical to visualize_person_bbox.py) ──

def draw_person_bbox(frame, bbox_px, label, color, line_thickness=1):
    """Draw body bbox + centered label above it (same as visualize_person_bbox.py)."""
    x1, y1, x2, y2 = map(int, bbox_px)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, line_thickness, cv2.LINE_AA)

    font_scale = 0.6
    font_thickness = 2
    (text_w, text_h), _ = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
    )
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
    """Re-encode to H.264 for compatibility (same as visualize_person_bbox.py)."""
    temp_path = video_path.replace(".mp4", "_temp.mp4")
    try:
        cmd = [
            "ffmpeg", "-y", "-i", video_path,
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
            "-pix_fmt", "yuv420p", "-an", temp_path,
        ]
        subprocess.run(cmd, capture_output=True, check=True)
        os.replace(temp_path, video_path)
    except subprocess.CalledProcessError:
        if os.path.exists(temp_path):
            os.remove(temp_path)


# ── Visualization ──

def create_viz(clip_path, gaze_data, matching, output_path,
               max_frames=None, alpha=0.6):
    """Create bbox visualization video (same style as visualize_person_bbox.py).

    Args:
        max_frames: If set, only write the first N frames (for online trimming).
    """
    cap = cv2.VideoCapture(clip_path)
    if not cap.isOpened():
        return False

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_fps = gaze_data.get("video_fps", 1.0)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, output_fps, (width, height))

    written = 0
    for frame_info in gaze_data["frames"]:
        if max_frames is not None and written >= max_frames:
            break

        frame_idx = frame_info["frame_idx"]
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue

        overlay = frame.copy()

        for person in frame_info.get("persons", []):
            pid = person["person_id"]
            body_bbox_px = person.get("body_bbox_px")
            if not body_bbox_px:
                continue

            player_idx = matching.get(pid)
            if player_idx is not None:
                label = f"P{player_idx}"
                color = COLORS[player_idx % len(COLORS)]
            else:
                label = f"?{pid}"
                color = (128, 128, 128)

            draw_person_bbox(overlay, body_bbox_px, label, color)

        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        out.write(frame)
        written += 1

    cap.release()
    out.release()
    reencode_h264(output_path)
    return True


# ── Main processing ──

def process_entry(entry, skip_existing=True):
    """Load gaze data, match players, create offline + online visualizations."""
    clip_name = entry["clip_name"]
    dataset = entry["dataset"]
    gaze_path = os.path.join(GAZE_DIRS[dataset], f"{clip_name}_sam3rf_gaze.json")
    viz_path = os.path.join(VIZ_DIR, f"{clip_name}.mp4")
    online_viz_path = os.path.join(ONLINE_VIZ_DIR, f"{clip_name}.mp4")

    result = {
        "clip_name": clip_name,
        "dataset": dataset,
        "game_id": entry["game_id"],
        "utt_idx": entry["utt_idx"],
        "utterance": entry["utterance"],
        "tasks": entry["tasks"],
        "player_num": entry["player_num"],
        "player_names": entry["player_names"],
        "speaker_player_id": entry["speaker_player_id"],
        "matching": {},
        "n_pipeline_persons": 0,
    }

    # Load gaze JSON
    if not os.path.exists(gaze_path):
        return None  # drop
    gaze_data = json.load(open(gaze_path))
    persons_summary = gaze_data.get("persons_summary", {})
    if not persons_summary:
        return None  # drop: 0 persons

    kp_positions = entry.get("kp_positions", {})
    if not kp_positions:
        return None  # drop: no MMSI keypoints

    result["n_pipeline_persons"] = len(persons_summary)

    # Hungarian matching
    pipeline_centers = compute_person_centers(gaze_data)
    matching = hungarian_match(pipeline_centers, kp_positions)
    result["matching"] = {str(pid): pidx for pid, pidx in matching.items()}

    # Online: frames up to utterance time only
    # utterance is at (utt_time_sec - clip_start_sec) into the clip
    online_frames = entry["utt_time_sec"] - entry["clip_start_sec"] + 1

    # Create offline visualization (all frames)
    if not (skip_existing and os.path.exists(viz_path)):
        create_viz(entry["clip_path"], gaze_data, matching, viz_path)

    # Create online visualization (trimmed to utterance time)
    if not (skip_existing and os.path.exists(online_viz_path)):
        create_viz(entry["clip_path"], gaze_data, matching, online_viz_path,
                   max_frames=online_frames)

    return result


def main():
    manifest = json.load(open(MANIFEST))

    # Only process clips that have gaze JSONs in MISSING_JSON_DIR
    missing_clips = set()
    for f in os.listdir(MISSING_JSON_DIR):
        if f.endswith("_sam3rf_gaze.json"):
            clip_name = f.replace("_sam3rf_gaze.json", "")
            missing_clips.add(clip_name)

    manifest = [e for e in manifest if e["clip_name"] in missing_clips]
    print(f"Processing {len(manifest)} missing clips")

    os.makedirs(VIZ_DIR, exist_ok=True)
    os.makedirs(ONLINE_VIZ_DIR, exist_ok=True)

    results = []
    dropped = 0

    for entry in tqdm(manifest, desc="Matching missing"):
        result = process_entry(entry, skip_existing=False)
        if result is None:
            dropped += 1
        else:
            results.append(result)

    with open(MATCHED_JSON, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nDone: {len(results)} matched, {dropped} dropped")
    print(f"Saved: {MATCHED_JSON}")


if __name__ == "__main__":
    main()
