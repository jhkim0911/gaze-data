"""Create TVQA+ videos with bbox annotations drawn on frames.

For each QA sample: extract ts-range frames at 3fps, draw bbox, encode to video.
Output: {qid}.mp4 at 3fps.
"""

import json
import os
import subprocess
import tempfile

import cv2
import numpy as np
from tqdm import tqdm

DATA_PATH = "/projects/illinois/eng/cs/jrehg/users/arkimjh/gaze_social/tvqa/penxiang/tvqa_plus_valid_preprocessed.json"
FRAMES_BASE = "/projects/illinois/eng/cs/jrehg/users/arkimjh/gaze_social/tvqa/penxiang/frames_hq/bbt_frames"
OUT_DIR = "/projects/illinois/eng/cs/jrehg/users/arkimjh/gaze_social/benchmark/original_tvqa+/video"

FRAME_INTERVAL = 6  # original TVQA+ samples every 6th frame (0.5fps from 3fps)
INPUT_FPS = 0.5  # each bbox frame represents 2 seconds
OUT_FPS = 3  # output video playback fps (ffmpeg will duplicate frames)

# Color palette (BGR) — matches gaze_annotation/visualize_person_bbox.py
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
LABEL_COLORS = {}


def get_color(label):
    if label not in LABEL_COLORS:
        LABEL_COLORS[label] = COLORS[len(LABEL_COLORS) % len(COLORS)]
    return LABEL_COLORS[label]


def draw_bbox(frame, bboxes):
    """Draw bboxes with labels, resolving overlapping labels by shifting up."""
    font_scale = 0.6
    font_thickness = 2

    # First pass: compute label rects for all bboxes
    entries = []
    for b in bboxes:
        x1, y1 = b["left"], b["top"]
        x2, y2 = x1 + b["width"], y1 + b["height"]
        label = b["label"]
        color = get_color(label)
        (tw, th), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
        )
        cx = (x1 + x2) // 2
        lx1 = cx - (tw + 6) // 2
        lx2 = lx1 + tw + 6
        lh = th + 8
        if y1 >= lh:
            ly1 = y1 - lh
        else:
            ly1 = y1
        entries.append({
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "label": label, "color": color,
            "lx1": lx1, "lx2": lx2, "ly1": ly1, "lh": lh, "th": th,
        })

    # Second pass: resolve overlaps by shifting labels up
    placed = []  # list of (lx1, ly1, lx2, ly2) already drawn
    for e in entries:
        ly1 = e["ly1"]
        while True:
            ly2 = ly1 + e["lh"]
            overlap = False
            for px1, py1, px2, py2 in placed:
                if e["lx1"] < px2 and e["lx2"] > px1 and ly1 < py2 and ly2 > py1:
                    overlap = True
                    ly1 = py1 - e["lh"]  # shift above the conflicting label
                    break
            if not overlap:
                break
        e["ly1"] = max(ly1, 0)  # clamp to frame top
        placed.append((e["lx1"], e["ly1"], e["lx2"], e["ly1"] + e["lh"]))

    # Draw
    for e in entries:
        cv2.rectangle(frame, (e["x1"], e["y1"]), (e["x2"], e["y2"]),
                       e["color"], 1, cv2.LINE_AA)
        ly1 = e["ly1"]
        ly2 = ly1 + e["lh"]
        ty = ly1 + e["th"] + 4
        cv2.rectangle(frame, (e["lx1"], ly1), (e["lx2"], ly2), e["color"], -1)
        cv2.putText(frame, e["label"], (e["lx1"] + 3, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0),
                    font_thickness, cv2.LINE_AA)
    return frame


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    data = json.load(open(DATA_PATH))
    print(f"Total samples: {len(data)}")

    # Build bbox lookup: img_id (str) -> list of bbox dicts
    for item in tqdm(data, desc="Creating videos"):
        qid = item["qid"]
        vid_name = item["vid_name"]
        ts = item["ts"]
        bbox = item["bbox"]

        frames_dir = os.path.join(FRAMES_BASE, vid_name)
        if not os.path.isdir(frames_dir):
            print(f"Skip: {frames_dir} not found")
            continue

        # Sample bbox-annotated frames (every FRAME_INTERVAL, matching original 0.5fps)
        bbox_fids = sorted(int(k) for k in bbox.keys())
        if not bbox_fids:
            print(f"Skip qid={qid}: no bbox frames")
            continue

        frames = []
        for fid in bbox_fids:
            fpath = os.path.join(frames_dir, f"{fid:05d}.jpg")
            if not os.path.exists(fpath):
                continue
            img = cv2.imread(fpath)
            if bbox[str(fid)]:
                img = draw_bbox(img, bbox[str(fid)])
            frames.append(img)

        if not frames:
            print(f"Skip qid={qid}: no frames")
            continue

        # Write frames to temp dir, encode with ffmpeg (h264)
        h, w = frames[0].shape[:2]
        out_path = os.path.join(OUT_DIR, f"{qid}.mp4")
        with tempfile.TemporaryDirectory() as tmpdir:
            for i, f in enumerate(frames):
                cv2.imwrite(os.path.join(tmpdir, f"{i:05d}.jpg"), f)
            subprocess.run([
                "ffmpeg", "-y",
                "-framerate", str(INPUT_FPS),
                "-i", os.path.join(tmpdir, "%05d.jpg"),
                "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                "-pix_fmt", "yuv420p",
                "-r", str(OUT_FPS),
                "-loglevel", "error",
                out_path,
            ], check=True)

    print("Done")


if __name__ == "__main__":
    main()
