"""
Create per-question MP4 videos from TVQA+ validation with bbox annotations.

For each QA entry, uses its bbox annotated frames (0.5 FPS) with bounding boxes
drawn in the same style as visualize_person_bbox.py.
Character names are anonymized to P0, P1, ... in both video labels and text.

Output:
  val_video/{qid}.mp4           - annotated video with anonymized labels
  tvqa_plus_valid_anonymized.json - anonymized Q/A text with name mapping
"""

import json
import os
import re
import subprocess

import cv2
from tqdm import tqdm

VALID_JSON = "/projects/illinois/eng/cs/jrehg/users/arkimjh/gaze_social/tvqa/penxiang/tvqa_plus_valid_preprocessed.json"
FRAMES_DIR = "/projects/illinois/eng/cs/jrehg/users/arkimjh/gaze_social/tvqa/penxiang/frames_hq/bbt_frames"
OUTPUT_DIR = "/projects/illinois/eng/cs/jrehg/users/arkimjh/gaze_social/benchmark/TVQA+/videos"
ANON_JSON = "/projects/illinois/eng/cs/jrehg/users/arkimjh/gaze_social/tvqa/penxiang/tvqa_plus_valid_preprocessed_formatted.json"

# Color palette (BGR) - same as visualize_person_bbox.py
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

# BBT character name variants (lowercase → canonical name)
BBT_CHARS = {
    "leonard": "Leonard", "lenard": "Leonard", "leanord": "Leonard", "leonrd": "Leonard",
    "sheldon": "Sheldon", "shedon": "Sheldon", "sheldin": "Sheldon",
    "penny": "Penny", "penney": "Penny", "peeny": "Penny",
    "howard": "Howard", "howards": "Howard",
    "raj": "Raj", "raf": "Raj", "rajesh": "Raj",
    "bernadette": "Bernadette", "benadette": "Bernadette", "berndette": "Bernadette",
    "amy": "Amy",
    "stuart": "Stuart", "stewart": "Stuart", "staurt": "Stuart",
    "priya": "Priya", "preya": "Priya",
    "wil": "Wil",  # Wil Wheaton — exclude "will" (too common as verb)
    "wheaton": "Wil",
    "leslie": "Leslie", "lesley": "Leslie",
    "barry": "Barry", "kripke": "Barry",  # Barry Kripke
    "emily": "Emily",
    "lucy": "Lucy",
    "zack": "Zack",
    "stephanie": "Stephanie",
    "beverly": "Beverly",
    "mary": "Mary",
    "missy": "Missy",
    "ramona": "Ramona",
}


def get_canonical(name):
    """Return canonical BBT character name, or None if not a character."""
    return BBT_CHARS.get(name.lower().strip())


def build_name_map(entry):
    """Build canonical_name → Px mapping from bbox labels and text fields."""
    names = set()
    # From bbox labels
    for boxes in entry.get("bbox", {}).values():
        for b in boxes:
            c = get_canonical(b["label"])
            if c:
                names.add(c)
    # From question + answer text
    text = " ".join([entry.get("q", "")] + [entry.get(f"a{i}", "") for i in range(5)])
    for word in text.split():
        c = get_canonical(word.strip(".,!?;:'\"()[]"))
        if c:
            names.add(c)
    return {n: f"P{i}" for i, n in enumerate(sorted(names))}


def anon_label(label, nmap):
    """Anonymize a bbox label. Returns Px if person, original if object."""
    c = get_canonical(label)
    return nmap[c] if c and c in nmap else label


def anon_text(text, nmap):
    """Replace character names in text with Px identifiers (case-sensitive)."""
    replacements = []
    for variant, canon in BBT_CHARS.items():
        if canon in nmap:
            # Use Title Case for matching in text (e.g. "leonard" → "Leonard")
            replacements.append((variant.capitalize(), nmap[canon]))
    # Longest first to avoid partial matches (e.g. "Bernadette" before "Amy")
    replacements.sort(key=lambda x: len(x[0]), reverse=True)
    seen = set()
    for text_form, px in replacements:
        if text_form in seen:
            continue
        seen.add(text_form)
        text = re.sub(r'\b' + re.escape(text_form) + r'\b', px, text)
    return text


# ── Label overlap resolution ──

def compute_label_info(x1, y1, x2, y2, label, color, font_scale=0.6, font_thickness=2):
    """Compute label rectangle position (default: centered above bbox)."""
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    (text_w, text_h), _ = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
    )
    bbox_cx = (x1 + x2) // 2
    lx1 = bbox_cx - (text_w + 6) // 2
    lx2 = lx1 + text_w + 6
    lh = text_h + 8

    if y1 >= lh:
        ly1, ly2, ty = y1 - lh, y1, y1 - 4
    else:
        ly1, ly2, ty = y1, y1 + lh, y1 + text_h + 4

    return {
        'bbox': (x1, y1, x2, y2), 'label': label, 'color': color,
        'lr': [lx1, ly1, lx2, ly2], 'tp': [lx1 + 3, ty],
        'text_h': text_h, 'lh': lh,
    }


def rects_overlap(a, b):
    return a[0] < b[2] and a[2] > b[0] and a[1] < b[3] and a[3] > b[1]


def resolve_overlaps(infos):
    """Greedily move overlapping labels below their bbox."""
    placed = []
    for info in infos:
        if any(rects_overlap(info['lr'], p) for p in placed):
            # Move label below bbox
            _, _, _, by2 = info['bbox']
            lx1, _, lx2, _ = info['lr']
            ny1 = by2 + 2
            info['lr'] = [lx1, ny1, lx2, ny1 + info['lh']]
            info['tp'] = [lx1 + 3, ny1 + info['text_h'] + 4]
            # Still overlapping? Keep offsetting down
            attempts = 0
            while any(rects_overlap(info['lr'], p) for p in placed) and attempts < 5:
                info['lr'][1] += info['lh'] + 2
                info['lr'][3] += info['lh'] + 2
                info['tp'][1] += info['lh'] + 2
                attempts += 1
        placed.append(info['lr'][:])
    return infos


def draw_annotations(frame, infos, line_thickness=1, font_scale=0.6, font_thickness=2):
    """Draw all bboxes and their (overlap-resolved) labels."""
    for info in infos:
        x1, y1, x2, y2 = info['bbox']
        cv2.rectangle(frame, (x1, y1), (x2, y2), info['color'], line_thickness, cv2.LINE_AA)
        lx1, ly1, lx2, ly2 = info['lr']
        cv2.rectangle(frame, (lx1, ly1), (lx2, ly2), info['color'], -1)
        cv2.putText(frame, info['label'], tuple(info['tp']),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), font_thickness, cv2.LINE_AA)


# ── Main processing ──

def process_question(entry, skip_existing=True):
    """Process one QA entry: anonymize names, draw bboxes, encode to MP4."""
    qid = entry["qid"]
    vid_name = entry["vid_name"]
    out_path = os.path.join(OUTPUT_DIR, f"{qid}.mp4")
    bbox_dict = entry.get("bbox", {})

    # Build name mapping (always needed for anonymized JSON)
    name_map = build_name_map(entry)

    anon_entry = {
        "qid": qid,
        "vid_name": vid_name,
        "ts": entry.get("ts"),
        "q": anon_text(entry["q"], name_map),
        "answer_idx": entry["answer_idx"],
        "name_mapping": {v: k for k, v in name_map.items()},
    }
    for i in range(5):
        anon_entry[f"a{i}"] = anon_text(entry[f"a{i}"], name_map)

    if skip_existing and os.path.exists(out_path):
        return "skip", anon_entry

    frames_path = os.path.join(FRAMES_DIR, vid_name)
    if not os.path.isdir(frames_path):
        return "no_frames", anon_entry

    if not bbox_dict:
        return "no_bbox", anon_entry

    # Parse and anonymize bboxes per frame
    frame_bboxes = {}
    for frame_key, boxes in bbox_dict.items():
        frame_no = int(frame_key)
        frame_bboxes[frame_no] = [
            (anon_label(b["label"], name_map), b["left"], b["top"],
             b["left"] + b["width"], b["top"] + b["height"])
            for b in boxes
        ]

    # Px label → color mapping (consistent across frames)
    all_labels = sorted(set(
        label for boxes in frame_bboxes.values() for label, *_ in boxes
    ))
    label_color = {label: COLORS[i % len(COLORS)] for i, label in enumerate(all_labels)}

    annotated_frame_nos = sorted(frame_bboxes.keys())

    tmp_dir = os.path.join(OUTPUT_DIR, f"_tmp_{qid}")
    os.makedirs(tmp_dir, exist_ok=True)

    try:
        for seq_idx, frame_no in enumerate(annotated_frame_nos, start=1):
            fname = f"{frame_no:05d}.jpg"
            frame = cv2.imread(os.path.join(frames_path, fname))
            if frame is None:
                continue

            # Compute label positions, resolve overlaps, then draw with alpha blend
            overlay = frame.copy()
            infos = [
                compute_label_info(x1, y1, x2, y2, label, label_color[label])
                for label, x1, y1, x2, y2 in frame_bboxes[frame_no]
            ]
            infos = resolve_overlaps(infos)
            draw_annotations(overlay, infos)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

            cv2.imwrite(os.path.join(tmp_dir, f"{seq_idx:05d}.jpg"), frame)

        cmd = [
            "ffmpeg", "-y",
            "-framerate", "0.5",
            "-i", os.path.join(tmp_dir, "%05d.jpg"),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-crf", "18",
            out_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            return f"ffmpeg_error: {result.stderr[:100]}", anon_entry

        return "ok", anon_entry

    finally:
        for f in os.listdir(tmp_dir):
            os.remove(os.path.join(tmp_dir, f))
        os.rmdir(tmp_dir)


def main():
    with open(VALID_JSON) as f:
        data = json.load(f)

    print(f"Total QA entries: {len(data)}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    stats = {"ok": 0, "skip": 0, "fail": 0}
    anon_data = []

    for entry in tqdm(data, desc="Making per-question videos"):
        result, anon_entry = process_question(entry)
        anon_data.append(anon_entry)
        if result == "ok":
            stats["ok"] += 1
        elif result == "skip":
            stats["skip"] += 1
        else:
            stats["fail"] += 1
            tqdm.write(f"FAIL [qid={entry['qid']}]: {result}")

    # Save anonymized JSON
    with open(ANON_JSON, "w") as f:
        json.dump(anon_data, f, indent=2, ensure_ascii=False)
    print(f"Anonymized JSON saved to: {ANON_JSON}")

    print(f"\nDone: {stats['ok']} created, {stats['skip']} skipped, {stats['fail']} failed")


if __name__ == "__main__":
    main()
