#!/usr/bin/env python3
"""
Cross Graph Generator - Unified graph from merged gaze+gesture cross_data.

Reads cross_data (merged gaze events + gestures per video), applies
cross-modal-aware pre-filtering, calls Gemini Flash for per-event
descriptions, and outputs an enriched graph JSON.

Input:  {dataset}/cross_data/{video}_cross_data.json
Output: {dataset}/graph_data/{video}_graph.json

Usage:
    # Single dataset
    python -m gaze_calculation.cross_graph_generator --dataset embody3d --skip_existing

    # All datasets
    python -m gaze_calculation.cross_graph_generator --all --skip_existing

    # Sample 10 files
    python -m gaze_calculation.cross_graph_generator --dataset embody3d --max_files 10

    # Single file
    python -m gaze_calculation.cross_graph_generator \
        --cross_data_json /path/to/cross_data.json
"""

import argparse
import concurrent.futures
import functools
import glob
import json
import os
import random
import re
import sys
import time
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

# Filtering thresholds
CROSS_MODAL_PROXIMITY_SEC = 3.0
MAX_EVENTS_PER_VIDEO = 25


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


# ------------------------------------------------------------------
# Pre-filtering: cross-modal priority + dedup + cap
# ------------------------------------------------------------------

def _events_overlap(a: dict, b: dict) -> bool:
    """Check if two events overlap temporally."""
    return a["start_time"] < b["end_time"] and b["start_time"] < a["end_time"]


def _events_close(a: dict, b: dict, proximity: float) -> bool:
    """Check if two events are within proximity seconds of each other."""
    gap = max(a["start_time"], b["start_time"]) - min(a["end_time"], b["end_time"])
    return gap <= proximity


def _same_participants(a: dict, b: dict) -> bool:
    """Check if two events involve the same participants."""
    a_persons = set()
    b_persons = set()

    if a["source"] == "gaze":
        a_persons = set(a.get("persons_involved", []))
    else:
        if a.get("initiator_id") is not None:
            a_persons.add(a["initiator_id"])
        if a.get("target_person_id") is not None:
            a_persons.add(a["target_person_id"])

    if b["source"] == "gaze":
        b_persons = set(b.get("persons_involved", []))
    else:
        if b.get("initiator_id") is not None:
            b_persons.add(b["initiator_id"])
        if b.get("target_person_id") is not None:
            b_persons.add(b["target_person_id"])

    return a_persons == b_persons


def _event_type(ev: dict) -> str:
    """Get the type key for an event (event_type for gaze, gesture_type for gesture)."""
    if ev["source"] == "gaze":
        return ev.get("event_type", "")
    return ev.get("gesture_type", "")


def detect_cross_modal_pairs(
    timeline: List[dict], proximity: float = CROSS_MODAL_PROXIMITY_SEC,
) -> List[dict]:
    """Find gaze-gesture pairs within temporal proximity.

    Returns list of {gaze_idx, gesture_idx, relation, overlap_or_gap_sec}.
    Indices refer to positions in the input timeline list.
    """
    gaze_indices = [i for i, ev in enumerate(timeline) if ev["source"] == "gaze"]
    gesture_indices = [i for i, ev in enumerate(timeline) if ev["source"] == "gesture"]

    pairs = []
    for gi in gesture_indices:
        g = timeline[gi]
        for ei in gaze_indices:
            e = timeline[ei]
            if _events_overlap(e, g):
                overlap = min(e["end_time"], g["end_time"]) - max(e["start_time"], g["start_time"])
                pairs.append({
                    "gaze_idx": ei, "gesture_idx": gi,
                    "relation": "overlap", "seconds": round(overlap, 2),
                })
            elif _events_close(e, g, proximity):
                gap = max(e["start_time"], g["start_time"]) - min(e["end_time"], g["end_time"])
                pairs.append({
                    "gaze_idx": ei, "gesture_idx": gi,
                    "relation": "adjacent", "seconds": round(gap, 2),
                })

    return pairs


def filter_events(
    timeline: List[dict],
    max_events: int = MAX_EVENTS_PER_VIDEO,
    proximity: float = CROSS_MODAL_PROXIMITY_SEC,
) -> Tuple[List[dict], List[dict]]:
    """Pre-filter events: cross-modal priority, dedup, cap.

    Returns (filtered_timeline, cross_modal_pairs).
    """
    if not timeline:
        return [], []

    # Step 1: Detect cross-modal pairs and mark protected indices
    cross_pairs = detect_cross_modal_pairs(timeline, proximity)
    protected = set()
    for p in cross_pairs:
        protected.add(p["gaze_idx"])
        protected.add(p["gesture_idx"])

    # Step 2: Dedup — same source + same type + same participants + overlapping time
    # Keep highest confidence among duplicates
    keep = set(range(len(timeline)))
    for i in range(len(timeline)):
        if i not in keep:
            continue
        for j in range(i + 1, len(timeline)):
            if j not in keep:
                continue
            a, b = timeline[i], timeline[j]
            if (a["source"] == b["source"]
                    and _event_type(a) == _event_type(b)
                    and _same_participants(a, b)
                    and _events_overlap(a, b)):
                # Drop the one with lower confidence (but never drop protected)
                if i in protected and j in protected:
                    continue
                if a.get("confidence", 0) >= b.get("confidence", 0):
                    if j not in protected:
                        keep.discard(j)
                else:
                    if i not in protected:
                        keep.discard(i)
                        break  # i is removed, stop comparing

    deduped = [timeline[i] for i in sorted(keep)]
    deduped_indices = sorted(keep)
    dropped_dedup = len(timeline) - len(deduped)

    # Step 3: Cap — if over max_events, prioritize protected + high confidence
    if len(deduped) > max_events:
        # Map original indices to deduped positions
        protected_deduped = set()
        for orig_idx in protected:
            if orig_idx in keep:
                pos = deduped_indices.index(orig_idx)
                protected_deduped.add(pos)

        # Ensure type diversity: keep at least 1 of each event type
        type_seen = {}
        diverse_set = set()
        for pos, ev in enumerate(deduped):
            t = (ev["source"], _event_type(ev))
            if t not in type_seen:
                type_seen[t] = pos
                diverse_set.add(pos)

        must_keep = protected_deduped | diverse_set
        remaining = [i for i in range(len(deduped)) if i not in must_keep]
        # Sort remaining by confidence descending
        remaining.sort(key=lambda i: deduped[i].get("confidence", 0), reverse=True)

        final_indices = sorted(must_keep)
        slots_left = max_events - len(final_indices)
        if slots_left > 0:
            final_indices.extend(sorted(remaining[:slots_left]))
        final_indices = sorted(set(final_indices))

        capped = [deduped[i] for i in final_indices]
    else:
        capped = deduped

    # Recompute cross-modal pairs on the final filtered timeline
    final_pairs = detect_cross_modal_pairs(capped, proximity)

    return capped, final_pairs


# ------------------------------------------------------------------
# CrossGraphGenerator
# ------------------------------------------------------------------

class CrossGraphGenerator:
    """Generate unified graph from cross_data with Gemini Flash descriptions."""

    def __init__(
        self,
        api_key: str,
        model_name: str = "models/gemini-3-flash-preview",
        max_events: int = MAX_EVENTS_PER_VIDEO,
    ):
        self.api_key = api_key
        self.model_name = model_name
        self.max_events = max_events

        if GEMINI_AVAILABLE:
            self.client = genai.Client(api_key=api_key)
        else:
            self.client = None

    def _build_gaze_entry(self, i: int, ev: dict) -> dict:
        """Build prompt entry for a gaze event with social details."""
        entry = {
            "index": i,
            "source": "gaze",
            "event_type": ev.get("event_type"),
            "start_time": ev.get("start_time"),
            "end_time": ev.get("end_time"),
            "persons_involved": ev.get("persons_involved"),
        }
        etype = ev.get("event_type", "")
        if etype == "gaze_following":
            entry["leader_id"] = ev.get("leader_id")
            entry["follower_id"] = ev.get("follower_id")
            if ev.get("lag_seconds"):
                entry["lag_seconds"] = ev["lag_seconds"]
        elif etype == "mutual_gaze":
            if ev.get("gaze_duration"):
                entry["duration"] = ev["gaze_duration"]
        elif etype in ("joint_attention", "attention_capture"):
            if ev.get("num_persons"):
                entry["num_persons"] = ev["num_persons"]
        return entry

    def _build_gesture_entry(self, i: int, ev: dict) -> dict:
        """Build prompt entry for a gesture event with target details."""
        entry = {
            "index": i,
            "source": "gesture",
            "gesture_type": ev.get("gesture_type"),
            "start_time": ev.get("start_time"),
            "end_time": ev.get("end_time"),
            "initiator_id": ev.get("initiator_id"),
            "target_type": ev.get("target_type"),
            "target_person_id": ev.get("target_person_id"),
        }
        if ev.get("target_description"):
            entry["target_description"] = ev["target_description"]
        return entry

    def build_prompt(self, events: List[dict]) -> str:
        """Build Gemini prompt for per-event descriptions."""
        events_for_prompt = []
        for i, ev in enumerate(events):
            if ev["source"] == "gaze":
                events_for_prompt.append(self._build_gaze_entry(i, ev))
            else:
                events_for_prompt.append(self._build_gesture_entry(i, ev))

        events_json = json.dumps(events_for_prompt, indent=2)

        prompt = f"""You are writing factual descriptions for social events detected by a computer vision pipeline.
Each event is either a gaze event or a deictic gesture event.

## Gaze Event Types

- **sudden_gaze_shift**: A person rapidly changed where they're looking.
- **joint_attention**: Multiple people looking at the same target simultaneously.
- **gaze_following**: One person (follower) looks where another person (leader) was looking, after a time delay.
- **attention_capture**: Multiple people suddenly shifted gaze to the same location simultaneously.
- **mutual_gaze**: Two people looking directly at each other (eye contact).

## Gesture Types

- **pointing**: A person extends their hand/arm to indicate a direction, person, object, or location.
- **showing**: A person holds up an object or orients something for visual inspection by another person.
- **giving**: A person extends their hand to transfer or hand over an object to another person.
- **reaching**: A person extends their arm to grasp or interact with an object.

## Gesture Target Types

- **person**: Gesture directed toward another individual (target_person_id identifies who).
- **object**: Gesture toward an inanimate object (target_description describes it).
- **location**: Gesture toward a spatial location (target_description describes where).

## Events (sorted by time)

{events_json}

## Task

Write a **one-sentence factual description** for each event.

Rules:
1. State WHAT happened: who did what to whom, when (start/end time)
2. For gaze events: mention persons involved, time range, and the social relationship (e.g. who followed whom, how many people looked together)
3. For gesture events: mention who performed it, the target (person or object from target_description), and the time
4. **Do NOT include raw numerical values** such as velocity, convergence score, gaze distance, or coordinates
5. **ONLY describe what the data shows.** Do NOT speculate or add interpretation.
6. **NEVER** use words like: "suggesting", "likely", "indicating", "possibly", "implying", "appears to", "seems to", "may have", "probably"
7. **NEVER** infer causes or motivations (e.g. "to get attention", "in response to", "showing interest")
8. Use "Person X" format for person references. Do NOT just restate raw field names.

Good examples:
→ "From 5.0s to 6.5s, Person 3 followed Person 0's gaze direction after a 1.5-second delay."
→ "From 42.0s to 43.5s, Person 0 and Person 1 maintained mutual eye contact for 1.5 seconds."
→ "At 13.0s, Person 0, Person 2, and Person 3 simultaneously shifted their gaze to the same location."
→ "From 4.0s to 5.0s, Person 3 pointed at Person 4."
→ "From 36.0s to 37.5s, Person 0 held up an object (VR headset) to show the group."
→ "From 69.0s to 70.5s, Person 4 showed cards to Person 3."

Bad examples (DO NOT write like this):
→ "...with a mean velocity of approximately 0.49."
→ "...suggesting Person 0 wanted to draw attention to something."
→ "...likely in response to Person 3's earlier pointing."

## Output Format (JSON)

Return a JSON object mapping the event index (as string) to its description:

{{
  "0": "From 8.0s to 8.5s, Person 0 and Person 2 ...",
  "1": "At 21.5s, Person 0 pointed at ...",
  ...
}}

Return ONLY valid JSON. No markdown fences, no commentary."""
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

    def generate_graph(self, cross_data_path: str, output_path: str) -> dict:
        """Full pipeline: load cross_data → filter → Gemini descriptions → save."""
        with open(cross_data_path, 'r') as f:
            cross_data = json.load(f)

        video_name = cross_data.get("video_name", "")
        timeline = cross_data.get("timeline", [])
        num_raw = len(timeline)

        if num_raw == 0:
            output = {
                "video_name": video_name,
                "num_events_raw": 0,
                "num_events_filtered": 0,
                "events": [],
                "cross_modal_pairs": [],
            }
            self._save_output(output, output_path)
            return output

        # Pre-filter
        filtered, cross_pairs = filter_events(timeline, max_events=self.max_events)
        num_filtered = len(filtered)

        if num_filtered < num_raw:
            print(f"  Filtered: {num_raw} → {num_filtered} events ({num_raw - num_filtered} dropped)")

        # Gemini: per-event descriptions
        if num_filtered > 0:
            prompt = self.build_prompt(filtered)
            print(f"  Calling Gemini ({self.model_name}) for {num_filtered} descriptions...")

            try:
                descriptions = self._call_gemini(prompt)
                matched = 0
                for i, ev in enumerate(filtered):
                    desc = descriptions.get(str(i))
                    if desc:
                        ev["description"] = desc
                        matched += 1
                    else:
                        ev["description"] = None
                print(f"  Gemini OK: {matched}/{num_filtered} descriptions")
            except Exception as e:
                print(f"  Gemini call failed: {e}")
                for ev in filtered:
                    ev["description"] = None

        output = {
            "video_name": video_name,
            "num_events_raw": num_raw,
            "num_events_filtered": num_filtered,
            "num_gaze": sum(1 for e in filtered if e["source"] == "gaze"),
            "num_gesture": sum(1 for e in filtered if e["source"] == "gesture"),
            "num_cross_modal_pairs": len(cross_pairs),
            "events": filtered,
            "cross_modal_pairs": cross_pairs,
        }

        self._save_output(output, output_path)
        return output

    def _save_output(self, output: dict, output_path: str) -> None:
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"  Saved: {output_path}")


# ------------------------------------------------------------------
# Dataset processing
# ------------------------------------------------------------------

def _process_single_file(
    generator: CrossGraphGenerator,
    cross_json: str,
    output_dir: str,
    skip_existing: bool,
    file_idx: int,
    total_files: int,
) -> str:
    """Process a single cross_data file. Returns 'processed', 'skipped', or 'failed'."""
    video_name = os.path.basename(cross_json).replace("_cross_data.json", "")
    output_path = os.path.join(output_dir, f"{video_name}_graph.json")

    if skip_existing and os.path.exists(output_path):
        return "skipped"

    print(f"\n[{file_idx+1}/{total_files}] {video_name}")

    try:
        generator.generate_graph(cross_json, output_path)
        return "processed"
    except Exception as e:
        print(f"  FAILED: {e}")
        return "failed"


def process_dataset(
    base_dir: str,
    dataset_name: str,
    generator: CrossGraphGenerator,
    skip_existing: bool = False,
    max_files: int = 0,
    sample_ratio: float = 1.0,
    num_workers: int = 10,
) -> Tuple[int, int, int]:
    """Process all cross_data files in a dataset."""
    cross_dir = os.path.join(base_dir, dataset_name, "cross_data")
    output_dir = os.path.join(base_dir, dataset_name, "graph_data")

    if not os.path.isdir(cross_dir):
        print(f"Cross data directory not found: {cross_dir}")
        return 0, 0, 0

    cross_files = sorted(glob.glob(os.path.join(cross_dir, "*_cross_data.json")))
    total = len(cross_files)

    if total == 0:
        print(f"No cross_data files found in: {cross_dir}")
        return 0, 0, 0

    sample_n = total
    if max_files > 0 and max_files < total:
        sample_n = max_files
    elif 0 < sample_ratio < 1.0:
        sample_n = max(1, int(total * sample_ratio))

    if sample_n < total:
        random.seed(42)
        cross_files = random.sample(cross_files, sample_n)
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset_name} (sampled {sample_n}/{total} files)")
    else:
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset_name} ({total} files)")

    print(f"Input:   {cross_dir}")
    print(f"Output:  {output_dir}")
    print(f"Workers: {num_workers}")
    print(f"{'='*60}")

    processed, skipped, failed = 0, 0, 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(
                _process_single_file,
                generator, cross_json, output_dir, skip_existing, i, len(cross_files),
            ): cross_json
            for i, cross_json in enumerate(cross_files)
        }

        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result == "processed":
                processed += 1
            elif result == "skipped":
                skipped += 1
            else:
                failed += 1

    print(f"\n{dataset_name} done: {processed} processed, {skipped} skipped, {failed} failed")
    return processed, skipped, failed


def main():
    parser = argparse.ArgumentParser(
        description="Generate unified graph from cross_data (merged gaze+gesture)"
    )

    parser.add_argument("--base_dir", type=str, default=DEFAULT_BASE_PATH)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--all", action="store_true")

    parser.add_argument("--cross_data_json", type=str, default=None)
    parser.add_argument("--output_json", type=str, default=None)

    parser.add_argument("--model", type=str, default="models/gemini-3-flash-preview")
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--max_events", type=int, default=MAX_EVENTS_PER_VIDEO)
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--max_files", type=int, default=0)
    parser.add_argument("--sample_ratio", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=10,
                        help="Number of parallel threads for Gemini API calls")

    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("ERROR: No API key. Set GOOGLE_API_KEY or use --api_key")
        sys.exit(1)

    generator = CrossGraphGenerator(
        api_key=api_key,
        model_name=args.model,
        max_events=args.max_events,
    )

    # Single file mode
    if args.cross_data_json:
        if not os.path.exists(args.cross_data_json):
            print(f"ERROR: File not found: {args.cross_data_json}")
            sys.exit(1)

        output_path = args.output_json
        if not output_path:
            base = args.cross_data_json.replace("_cross_data.json", "")
            output_path = f"{base}_graph.json"

        print(f"Single file mode")
        print(f"Input:  {args.cross_data_json}")
        print(f"Output: {output_path}")
        generator.generate_graph(args.cross_data_json, output_path)
        return

    # Dataset mode
    if args.all:
        datasets = DATASETS
    elif args.dataset:
        datasets = [args.dataset]
    else:
        parser.print_help()
        print("\nProvide --dataset, --all, or --cross_data_json")
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
            num_workers=args.num_workers,
        )
        total_processed += p
        total_skipped += s
        total_failed += f

    print(f"\n{'='*60}")
    print(f"All done: {total_processed} processed, {total_skipped} skipped, {total_failed} failed")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
