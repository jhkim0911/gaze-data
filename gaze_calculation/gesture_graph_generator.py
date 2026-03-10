#!/usr/bin/env python3
"""
Gesture Graph Generator - Per-Gesture Descriptions from Raw Gesture Data

Intermediate step: gesture_info (raw JSON) → Gemini (per-gesture descriptions) → gesture_graph (enriched JSON)
Sends filtered gestures to Gemini to generate a natural language description for each gesture,
interpreting raw detection data into meaningful social context for downstream QA generation.

Usage:
    # Quality check (5 random files)
    python -m gaze_calculation.gesture_graph_generator \
        --dataset social_gesture --max_files 5

    # Half the dataset
    python -m gaze_calculation.gesture_graph_generator \
        --dataset friendsmmc --sample_ratio 0.5 --skip_existing

    # All datasets
    python -m gaze_calculation.gesture_graph_generator \
        --all --sample_ratio 0.5 --skip_existing

    # Single file
    python -m gaze_calculation.gesture_graph_generator \
        --gestures_json /path/to/gestures_v2.json
"""

import argparse
import json
import math
import os
import subprocess
import sys
import time
import random
import re
import functools
import glob
from typing import Dict, List, Optional, Tuple

sys.stdout.reconfigure(line_buffering=True)

# Load .env
ENV_FILE_PATH = "/u/arkimjh/code/ECCV-jh/.env"
try:
    from dotenv import load_dotenv
    if os.path.exists(ENV_FILE_PATH):
        load_dotenv(ENV_FILE_PATH)
    else:
        load_dotenv()
except ImportError:
    pass

try:
    from google import genai
    from google.genai import types
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("Warning: google-genai not installed. Install: pip install google-genai")

DATASETS = [
    'avsbench', 'embody3d', 'friendsmmc', 'social_gesture',
    'social-iq', 'tvqa', 'werewolf',
]
DEFAULT_BASE_PATH = "/projects/illinois/eng/cs/jrehg/users/arkimjh/gaze_social"

MAX_GESTURES_FOR_PROMPT = 50
ADJACENCY_THRESHOLD_SEC = 3.0


def retry_with_backoff(retries=5, initial_delay=5, backoff_factor=2):
    """Retry decorator with exponential backoff for API rate limits."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay
            for i in range(retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    msg = str(e)
                    if "429" in msg or "RESOURCE_EXHAUSTED" in msg or "500" in msg:
                        if i == retries - 1:
                            raise
                        sleep = delay + random.uniform(0, 1)
                        print(f"  API Error ({msg}). Retrying in {sleep:.1f}s...")
                        time.sleep(sleep)
                        delay *= backoff_factor
                    else:
                        raise
            return func(*args, **kwargs)
        return wrapper
    return decorator


def _format_person(person_id: int) -> str:
    return f"Person {person_id}"


def _get_video_duration(video_path: str) -> Optional[float]:
    """Get video duration in seconds via ffprobe."""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
             "-of", "csv=p=0", video_path],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            return float(result.stdout.strip())
    except (subprocess.TimeoutExpired, ValueError):
        pass
    return None


def _resolve_video_path(gesture_file: str, base_dir: str, dataset_name: str) -> Optional[str]:
    """Resolve gesture filename → nbbox_videos/{name}_bbox_viz.mp4."""
    video_name = os.path.basename(gesture_file)
    for suffix in ["_gestures_v2.json", "_gestures.json"]:
        if video_name.endswith(suffix):
            video_name = video_name[:-len(suffix)]
            break
    video_path = os.path.join(base_dir, dataset_name, "nbbox_videos", f"{video_name}_bbox_viz.mp4")
    if os.path.exists(video_path):
        return video_path
    return None


class GestureGraphGenerator:
    """Generate enriched gesture graphs with Gemini-generated per-gesture descriptions."""

    def __init__(
        self,
        api_key: str,
        model_name: str = "models/gemini-3-pro-preview",
        confidence_threshold: float = 0.85,
    ):
        self.api_key = api_key
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold

        if GEMINI_AVAILABLE:
            self.client = genai.Client(api_key=api_key)
        else:
            self.client = None

    # ------------------------------------------------------------------
    # Step 1: Load, Sanitize & Filter
    # ------------------------------------------------------------------

    @staticmethod
    def _snap_to_half(t: float, ceil: bool = False) -> float:
        """Snap timestamp to nearest 0.5 boundary (floor or ceil)."""
        if ceil:
            return math.ceil(t * 2) / 2
        return math.floor(t * 2) / 2

    def load_and_filter(self, gestures_json_path: str, max_duration: float = 120.0) -> Tuple[dict, List[dict]]:
        """Load gestures JSON, sanitize timestamps, filter by confidence & duration, sort by time."""
        with open(gestures_json_path, 'r') as f:
            raw_data = json.load(f)

        gestures = raw_data.get("gestures", [])
        # Add gesture_index for tracking (gestures don't have event_id)
        for i, g in enumerate(gestures):
            g["gesture_index"] = i

        # Snap timestamps to 0.5 boundaries (classifier hallucination cleanup)
        for g in gestures:
            g["start_time"] = self._snap_to_half(g.get("start_time", 0), ceil=False)
            g["end_time"] = self._snap_to_half(g.get("end_time", 0), ceil=True)

        before_count = len([g for g in gestures if g.get("confidence", 0) >= self.confidence_threshold])
        filtered = [
            g for g in gestures
            if g.get("confidence", 0) >= self.confidence_threshold
            and g.get("end_time", 0) <= max_duration
        ]
        dropped = before_count - len(filtered)
        if dropped > 0:
            print(f"  Dropped {dropped} gestures exceeding video duration ({max_duration}s)")

        filtered.sort(key=lambda g: (g.get("start_time", 0), g.get("end_time", 0)))
        return raw_data, filtered

    # ------------------------------------------------------------------
    # Step 2: Compute Temporal Relations (adjacent gestures only)
    # ------------------------------------------------------------------

    def compute_temporal_relations(self, gestures: List[dict]) -> list:
        """Find overlaps and adjacencies between gestures."""
        relations = []

        for i, g_a in enumerate(gestures):
            a_idx = g_a.get("gesture_index")
            a_end = g_a.get("end_time", 0)

            for j in range(i + 1, len(gestures)):
                g_b = gestures[j]
                b_idx = g_b.get("gesture_index")
                b_start = g_b.get("start_time", 0)
                b_end = g_b.get("end_time", 0)

                if b_start - a_end > ADJACENCY_THRESHOLD_SEC:
                    break

                if a_end > b_start:
                    overlap = min(a_end, b_end) - b_start
                    relations.append({
                        "type": "overlaps",
                        "gesture_a": a_idx,
                        "gesture_b": b_idx,
                        "overlap_seconds": round(overlap, 2),
                    })
                else:
                    gap = b_start - a_end
                    relations.append({
                        "type": "immediately_follows",
                        "from_gesture": a_idx,
                        "to_gesture": b_idx,
                        "gap_seconds": round(gap, 2),
                    })

        return relations

    # ------------------------------------------------------------------
    # Step 3: Gemini Call — per-gesture descriptions
    # ------------------------------------------------------------------

    def build_prompt(self, sorted_gestures: List[dict]) -> str:
        """Construct Gemini prompt asking for per-gesture descriptions."""
        gestures_for_prompt = []
        for g in sorted_gestures:
            entry = {
                "gesture_index": g.get("gesture_index"),
                "gesture_type": g.get("gesture_type"),
                "initiator_id": g.get("initiator_id"),
                "start_time": g.get("start_time"),
                "end_time": g.get("end_time"),
                "target_type": g.get("target_type"),
                "target_person_id": g.get("target_person_id"),
                "target_description": g.get("target_description"),
            }
            gestures_for_prompt.append(entry)

        gestures_json = json.dumps(gestures_for_prompt, indent=2)

        prompt = f"""You are writing factual descriptions for deictic gesture events detected by a computer vision pipeline.

## Gesture Type Definitions

- **pointing**: A person extends their hand/arm to indicate a direction, person, object, or location.
- **showing**: A person holds up an object or orients something for visual inspection by another person.
- **giving**: A person extends their hand to transfer or hand over an object to another person.
- **reaching**: A person extends their arm to grasp or interact with an object.

## Target Types

- **person**: Gesture directed toward another individual (target_person_id identifies who).
- **object**: Gesture toward an inanimate object (target_description describes it).
- **location**: Gesture toward a spatial location (target_description describes where).

## Gestures (sorted by time, confidence >= {self.confidence_threshold})

{gestures_json}

## Task

Write a **one-sentence factual description** for each gesture.

Rules:
1. State WHAT happened: who did what, toward whom/what, when (start/end time)
2. Include key details naturally (e.g. target person, object, duration)
3. **ONLY describe what the data shows.** Do NOT speculate or add interpretation.
4. **NEVER** use words like: "suggesting", "likely", "indicating", "possibly", "implying", "appears to", "seems to", "may have", "probably"
5. **NEVER** infer causes or motivations (e.g. "to get attention", "in response to", "showing interest")
6. Do NOT just restate raw field names/numbers — write natural language
7. Use "Person X" format for person references

Good examples:
→ "From 4.0s to 5.0s, Person 3 pointed at Person 4."
→ "From 36.0s to 37.5s, Person 0 held up an object (VR headset) to show the group."
→ "From 22.5s to 23.5s, Person 0 reached for an object on the center of the table."
→ "From 69.0s to 70.5s, Person 4 showed cards to Person 3."

Bad examples (speculative — DO NOT write like this):
→ "...suggesting Person 0 wanted to draw attention to the VR headset."
→ "...likely in response to Person 3's earlier pointing."
→ "...indicating a transfer of an important item."

## Output Format (JSON)

Return a JSON object mapping gesture_index (as string) to its description:

{{
  "0": "From 4.0s to 5.0s, Person 3 pointed at ...",
  "3": "From 22.5s to 23.5s, Person 0 reached for ...",
  ...
}}

Return ONLY valid JSON. No markdown fences, no commentary outside JSON.
"""
        return prompt

    @retry_with_backoff()
    def _call_gemini(self, prompt: str) -> dict:
        """Send prompt to Gemini and parse JSON response."""
        if not self.client:
            return {}

        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.3,
        )

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=[prompt],
            config=config,
        )

        raw_text = response.text.strip() if response.text else ""

        try:
            parsed = json.loads(raw_text)
        except json.JSONDecodeError:
            json_match = re.search(r'\{.*\}', raw_text, re.DOTALL)
            if json_match:
                try:
                    parsed = json.loads(json_match.group())
                except json.JSONDecodeError:
                    parsed = {}
            else:
                parsed = {}

        return parsed

    # ------------------------------------------------------------------
    # Full Pipeline
    # ------------------------------------------------------------------

    def generate_graph(self, gestures_json_path: str, output_path: str, max_duration: float = 120.0) -> dict:
        """Full pipeline: load → filter → Gemini descriptions → temporal relations → save."""
        print(f"  Loading: {os.path.basename(gestures_json_path)}")
        raw_data, filtered_gestures = self.load_and_filter(gestures_json_path, max_duration=max_duration)
        total_gestures = len(raw_data.get("gestures", []))
        num_filtered = len(filtered_gestures)
        print(f"  Gestures: {total_gestures} total, {num_filtered} filtered (>= {self.confidence_threshold})")

        # Compute temporal relations
        temporal_relations = self.compute_temporal_relations(filtered_gestures)

        # Timeline range
        if filtered_gestures:
            min_time = filtered_gestures[0].get("start_time", 0)
            max_time = max(g.get("end_time", 0) for g in filtered_gestures)
        else:
            min_time, max_time = 0.0, 0.0

        # Gemini: generate per-gesture descriptions
        if num_filtered > 0:
            gestures_for_prompt = filtered_gestures
            if num_filtered > MAX_GESTURES_FOR_PROMPT:
                print(f"  Truncating {num_filtered} gestures to top {MAX_GESTURES_FOR_PROMPT} by confidence")
                gestures_for_prompt = sorted(
                    filtered_gestures, key=lambda g: g.get("confidence", 0), reverse=True
                )[:MAX_GESTURES_FOR_PROMPT]
                gestures_for_prompt.sort(key=lambda g: (g.get("start_time", 0), g.get("end_time", 0)))

            prompt = self.build_prompt(gestures_for_prompt)
            print(f"  Calling Gemini ({self.model_name}) for {len(gestures_for_prompt)} gesture descriptions...")

            try:
                descriptions = self._call_gemini(prompt)
                matched = 0
                for g in filtered_gestures:
                    gidx = str(g.get("gesture_index"))
                    if gidx in descriptions:
                        g["description"] = descriptions[gidx]
                        matched += 1
                    else:
                        g["description"] = None
                print(f"  Gemini OK: {matched}/{num_filtered} gestures got descriptions")
            except Exception as e:
                print(f"  Gemini call failed: {e}")
                for g in filtered_gestures:
                    g["description"] = None

        output = {
            "video_path": raw_data.get("video_path", ""),
            "source_gesture_file": gestures_json_path,
            "classifier_version": raw_data.get("classifier_version", ""),
            "num_gestures_total": total_gestures,
            "num_gestures_filtered": num_filtered,
            "confidence_threshold": self.confidence_threshold,
            "gesture_counts": raw_data.get("gesture_counts", {}),
            "timeline": {
                "duration_covered": [round(min_time, 2), round(max_time, 2)],
                "gestures_sorted": filtered_gestures,
            },
            "temporal_relations": temporal_relations,
        }

        self._save_output(output, output_path)
        return output

    def _save_output(self, output: dict, output_path: str) -> None:
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"  Saved: {output_path}")


def process_dataset(
    base_dir: str,
    dataset_name: str,
    generator: GestureGraphGenerator,
    skip_existing: bool = False,
    max_files: int = 0,
    sample_ratio: float = 1.0,
) -> Tuple[int, int, int]:
    """Process all gesture files in a dataset."""
    gesture_dir = os.path.join(base_dir, dataset_name, "gesture_info")
    output_dir = os.path.join(base_dir, dataset_name, "gesture_graph")

    if not os.path.isdir(gesture_dir):
        print(f"Gesture info directory not found: {gesture_dir}")
        return 0, 0, 0

    gesture_files = sorted(glob.glob(os.path.join(gesture_dir, "*_gestures_v2.json")))
    total = len(gesture_files)

    if total == 0:
        print(f"No gesture files found in: {gesture_dir}")
        return 0, 0, 0

    sample_n = total
    if max_files > 0 and max_files < total:
        sample_n = max_files
    elif 0 < sample_ratio < 1.0:
        sample_n = max(1, int(total * sample_ratio))

    if sample_n < total:
        random.seed(42)
        gesture_files = random.sample(gesture_files, sample_n)
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset_name} (sampled {sample_n}/{total} files)")
    else:
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset_name} ({total} files)")

    print(f"Input:   {gesture_dir}")
    print(f"Output:  {output_dir}")
    print(f"{'='*60}")

    to_process = len(gesture_files)
    processed, skipped, failed = 0, 0, 0

    for i, gestures_json in enumerate(gesture_files):
        base = os.path.basename(gestures_json)
        video_name = base
        for suffix in ["_gestures_v2.json", "_gestures.json"]:
            if video_name.endswith(suffix):
                video_name = video_name[:-len(suffix)]
                break
        output_path = os.path.join(output_dir, f"{video_name}_gesture_graph.json")

        if skip_existing and os.path.exists(output_path):
            skipped += 1
            continue

        print(f"\n[{i+1}/{to_process}] {video_name}")

        # Get actual video duration via ffprobe
        video_path = _resolve_video_path(gestures_json, base_dir, dataset_name)
        max_dur = 120.0
        if video_path:
            dur = _get_video_duration(video_path)
            if dur:
                max_dur = dur
        else:
            print(f"  Warning: video not found, using max_duration={max_dur}s")

        try:
            generator.generate_graph(gestures_json, output_path, max_duration=max_dur)
            processed += 1
        except Exception as e:
            print(f"  FAILED: {e}")
            failed += 1

        time.sleep(0.5)

    print(f"\n{dataset_name} done: {processed} processed, {skipped} skipped, {failed} failed")
    return processed, skipped, failed


def main():
    parser = argparse.ArgumentParser(
        description="Generate gesture graphs with per-gesture descriptions from raw gesture data"
    )

    parser.add_argument("--base_dir", type=str, default=DEFAULT_BASE_PATH)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--all", action="store_true")

    parser.add_argument("--gestures_json", type=str, default=None)
    parser.add_argument("--output_json", type=str, default=None)

    parser.add_argument("--confidence", type=float, default=0.85)
    parser.add_argument("--video_path", type=str, default=None,
                        help="Video file for duration check (single file mode)")
    parser.add_argument("--model", type=str, default="models/gemini-3-pro-preview")
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--max_files", type=int, default=0)
    parser.add_argument("--sample_ratio", type=float, default=1.0)

    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("ERROR: No API key. Set GOOGLE_API_KEY or use --api_key")
        sys.exit(1)

    generator = GestureGraphGenerator(
        api_key=api_key,
        model_name=args.model,
        confidence_threshold=args.confidence,
    )

    if args.gestures_json:
        if not os.path.exists(args.gestures_json):
            print(f"ERROR: File not found: {args.gestures_json}")
            sys.exit(1)

        output_path = args.output_json
        if not output_path:
            base = os.path.splitext(args.gestures_json)[0]
            for suffix in ["_gestures_v2", "_gestures"]:
                if base.endswith(suffix):
                    base = base[:-len(suffix)]
                    break
            output_path = f"{base}_gesture_graph.json"

        max_dur = 120.0
        if args.video_path:
            dur = _get_video_duration(args.video_path)
            if dur:
                max_dur = dur

        print(f"Single file mode")
        print(f"Input:  {args.gestures_json}")
        print(f"Output: {output_path}")
        generator.generate_graph(args.gestures_json, output_path, max_duration=max_dur)
        return

    if args.all:
        datasets = DATASETS
    elif args.dataset:
        datasets = [args.dataset]
    else:
        parser.print_help()
        print("\nProvide --dataset, --all, or --gestures_json")
        sys.exit(1)

    total_processed, total_skipped, total_failed = 0, 0, 0

    for ds in datasets:
        p, s, f = process_dataset(
            base_dir=args.base_dir,
            dataset_name=ds,
            generator=generator,
            skip_existing=args.skip_existing,
            max_files=args.max_files,
            sample_ratio=args.sample_ratio,
        )
        total_processed += p
        total_skipped += s
        total_failed += f

    print(f"\n{'='*60}")
    print(f"All done: {total_processed} processed, {total_skipped} skipped, {total_failed} failed")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
