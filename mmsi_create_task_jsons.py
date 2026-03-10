"""
Generate per-task JSON files for LVLM evaluation on MMSI benchmark.

Replicates MMSI dataloader task annotation logic (STI/PCR/MPP),
formats each data point with video path, anonymized utterance (PX format),
conversation context, question, choices, and ground truth answer.

Generates both offline (symmetric context ±5) and online (past-only, d=10) versions.

Output (under /projects/.../amongus/):
  mmsi_test_sti.json          / online_mmsi_test_sti.json
  mmsi_test_pcr.json          / online_mmsi_test_pcr.json
  mmsi_test_mpp.json          / online_mmsi_test_mpp.json
"""

import copy
import json
import os
import re

BASE = "/projects/illinois/eng/cs/jrehg/users/arkimjh/gaze_social/amongus"
MANIFEST = f"{BASE}/mmsi_test_manifest.json"
TRANSCRIPT_DIRS = {
    "youtube": "/projects/illinois/eng/cs/jrehg/users/ajojic2/MMSI/datasets/youtube/transcripts",
    "ego4d": "/projects/illinois/eng/cs/jrehg/users/ajojic2/MMSI/datasets/ego4d/transcripts",
}
OFFLINE_VIDEO_REL = "MMSI/videos"
ONLINE_VIDEO_REL = "Online-MMSI/videos"
CONTEXT_LENGTH = 5
ONLINE_CONTEXT_LENGTH = 10

SECOND_PRONOUNS = ["you", "your"]
THIRD_PRONOUNS = ["he", "his", "him", "she", "her"]


def load_transcript(dataset, game_id):
    """Load anonymized and anonymized_labeled transcripts."""
    base = TRANSCRIPT_DIRS[dataset]
    with open(f"{base}/anonymized/{game_id}.txt") as f:
        utts = [l for l in f.read().split("\n") if l]
    with open(f"{base}/anonymized_labeled/{game_id}.txt") as f:
        utts_lab = [l for l in f.read().split("\n") if l]
    return utts, utts_lab


def anonymize_text(text):
    """Convert [PlayerX] → PX, remove timestamps."""
    text = re.sub(r"\[Player(\d+)\]", r"P\1", text)
    text = re.sub(r" \(\d{2}:\d{2}\)", "", text)
    return text


def get_context(utterances, utt_idx):
    """Get conversation context window (same as MMSI dataloader)."""
    n = len(utterances)
    start = max(0, utt_idx - CONTEXT_LENGTH
                - max(0, CONTEXT_LENGTH - (n - utt_idx) + 1))
    end = min(start + 2 * CONTEXT_LENGTH + 1, n)
    return [anonymize_text(utterances[i]) for i in range(start, end)]


def get_online_context(utterances, utt_idx):
    """Get past-only context window for online setting (d=10 historical turns)."""
    start = max(0, utt_idx - ONLINE_CONTEXT_LENGTH + 1)
    end = utt_idx + 1  # inclusive of current utterance
    return [anonymize_text(utterances[i]) for i in range(start, end)]


def extract_task_points(entry, utterances, utterances_labeled):
    """Extract per-word task data points from a single utterance.

    Replicates MMSI dataloader.process_word() logic exactly.
    """
    utt_idx = entry["utt_idx"]
    utt = utterances[utt_idx]
    utt_lab = utterances_labeled[utt_idx]
    player_num = entry["player_num"]

    words = utt.split()
    lab_words = utt_lab.split()
    is_player = words[0].startswith("[Player")
    if not is_player:
        return []

    speaker_id = int(re.search(r"\[Player(\d+)\]", words[0]).group(1))
    clip_name = entry["clip_name"]
    dataset = entry["dataset"]
    viz_rel = f"{OFFLINE_VIDEO_REL}/{clip_name}.mp4"
    online_viz_rel = f"{ONLINE_VIDEO_REL}/{clip_name}.mp4"

    utt_anon = anonymize_text(utt)
    context = get_context(utterances, utt_idx)
    online_context = get_online_context(utterances, utt_idx)
    choices = [f"P{i}" for i in range(player_num)]

    base_info = {
        "clip_name": clip_name,
        "video_path": viz_rel,
        "online_video_path": online_viz_rel,
        "dataset": dataset,
        "game_id": entry["game_id"],
        "speaker": f"P{speaker_id}",
        "player_num": player_num,
        "utterance": utt_anon,
        "context": context,
        "online_context": online_context,
        "choices": choices,
    }

    points = []
    utterance_involved = False  # STI: at most one per utterance

    for word_i, word in enumerate(words):
        if word_i == 0:
            continue

        # ── STI ──
        if (word.lower() in SECOND_PRONOUNS
                and utt_lab.rstrip().endswith("]")
                and not utterance_involved):
            brackets = re.findall(r"\[(.*?)\]", utt_lab)
            valid = [name in [f"Player{i}" for i in range(player_num)]
                     for name in brackets[-1].split()]
            if valid.count(True) == 1:
                label = int(re.search(r"Player(\d+)", brackets[-1]).group(1))
                utterance_involved = True
                dp = {
                    **base_info,
                    "id": f"{dataset}__{clip_name}__STI__w{word_i}",
                    "task": "STI",
                    "question": f"Based on the video, who is P{speaker_id} talking to?",
                    "answer": f"P{label}",
                    "answer_idx": label,
                }
                points.append(dp)

        # ── PCR ──
        if (any(p in word.lower() for p in THIRD_PRONOUNS)
                and word_i < len(lab_words)
                and "Player" in lab_words[word_i]):
            label = int(re.search(r"Player(\d+)", lab_words[word_i]).group(1))
            pronoun = [p for p in THIRD_PRONOUNS if p in word.lower()][-1]
            dp = {
                **base_info,
                "id": f"{dataset}__{clip_name}__PCR__w{word_i}",
                "task": "PCR",
                "question": (f"In the utterance by P{speaker_id}, "
                             f"the pronoun '{pronoun}' refers to which player?"),
                "answer": f"P{label}",
                "answer_idx": label,
            }
            points.append(dp)

        # ── MPP ──
        if word.startswith("[Player"):
            label = int(re.search(r"\[Player(\d+)\]", word).group(1))
            words_masked = copy.deepcopy(words)
            words_masked[word_i] = re.sub(r"\[Player\d+\]", "[MASK]", words_masked[word_i])
            utt_masked = anonymize_text(" ".join(words_masked))
            dp = {
                **base_info,
                "id": f"{dataset}__{clip_name}__MPP__w{word_i}",
                "task": "MPP",
                "utterance_masked": utt_masked,
                "question": (f"In the utterance by P{speaker_id}: "
                             f"'{utt_masked}', which player does [MASK] refer to?"),
                "answer": f"P{label}",
                "answer_idx": label,
            }
            points.append(dp)

    return points


def main():
    manifest = json.load(open(MANIFEST))
    print(f"Total manifest entries: {len(manifest)}")

    # Cache transcripts by game_id
    transcript_cache = {}

    sti_data, pcr_data, mpp_data = [], [], []

    for entry in manifest:
        game_id = entry["game_id"]
        dataset = entry["dataset"]
        cache_key = f"{dataset}__{game_id}"
        if cache_key not in transcript_cache:
            transcript_cache[cache_key] = load_transcript(dataset, game_id)
        utts, utts_lab = transcript_cache[cache_key]

        for dp in extract_task_points(entry, utts, utts_lab):
            if dp["task"] == "STI":
                sti_data.append(dp)
            elif dp["task"] == "PCR":
                pcr_data.append(dp)
            elif dp["task"] == "MPP":
                mpp_data.append(dp)

    # Save per-task JSONs
    drop_keys = {"online_context", "online_video_path"}
    for name, data in [("sti", sti_data), ("pcr", pcr_data), ("mpp", mpp_data)]:
        # Offline: symmetric context, MMSI/videos/ path
        offline_data = [{k: v for k, v in dp.items() if k not in drop_keys}
                        for dp in data]
        path = f"{BASE}/mmsi_test_{name}.json"
        with open(path, "w") as f:
            json.dump(offline_data, f, indent=2, ensure_ascii=False)
        print(f"  {name.upper()}: {len(data)} data points → {path}")

        # Online: past-only context, Online-MMSI/videos/ path
        online_data = []
        for dp in data:
            dp_online = {k: v for k, v in dp.items() if k not in drop_keys}
            dp_online["context"] = dp["online_context"]
            dp_online["video_path"] = dp["online_video_path"]
            online_data.append(dp_online)
        path = f"{BASE}/online_mmsi_test_{name}.json"
        with open(path, "w") as f:
            json.dump(online_data, f, indent=2, ensure_ascii=False)
        print(f"  {name.upper()} (online): {len(data)} data points → {path}")

    total = len(sti_data) + len(pcr_data) + len(mpp_data)
    print(f"\nTotal: {total} (×2 for offline + online)")


if __name__ == "__main__":
    main()
