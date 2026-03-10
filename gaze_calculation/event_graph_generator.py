#!/usr/bin/env python3
"""
Event Graph Generator - Per-Event Descriptions from Raw Gaze Event Data

Intermediate step: event_data (raw JSON) → Gemini (per-event descriptions) → event_info (enriched JSON)
Sends filtered events to Gemini to generate a natural language description for each event,
interpreting raw detection data into meaningful social context for downstream QA generation.

Usage:
    # Quality check (5 random files)
    python -m gaze_calculation.event_graph_generator \
        --dataset social_gesture --max_files 5

    # Half the dataset
    python -m gaze_calculation.event_graph_generator \
        --dataset werewolf --sample_ratio 0.5 --skip_existing

    # All datasets
    python -m gaze_calculation.event_graph_generator \
        --all --sample_ratio 0.5 --skip_existing

    # Single file
    python -m gaze_calculation.event_graph_generator \
        --events_json /path/to/events.json
"""

import argparse
import json
import os
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

MAX_EVENTS_FOR_PROMPT = 50
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


def _format_persons(persons: list) -> str:
    if not persons:
        return "unknown persons"
    if len(persons) == 1:
        return f"Person {persons[0]}"
    if len(persons) == 2:
        return f"Person {persons[0]} and Person {persons[1]}"
    parts = [f"Person {p}" for p in persons[:-1]]
    return f"{', '.join(parts)}, and Person {persons[-1]}"


class EventGraphGenerator:
    """Generate enriched event graphs with Gemini-generated per-event descriptions."""

    def __init__(
        self,
        api_key: str,
        model_name: str = "models/gemini-3-pro-preview",
        confidence_threshold: float = 0.9,
    ):
        self.api_key = api_key
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold

        if GEMINI_AVAILABLE:
            self.client = genai.Client(api_key=api_key)
        else:
            self.client = None

    # ------------------------------------------------------------------
    # Step 1: Load & Filter
    # ------------------------------------------------------------------

    def load_and_filter(self, events_json_path: str) -> Tuple[dict, List[dict]]:
        """Load events JSON, filter by confidence, sort by time."""
        with open(events_json_path, 'r') as f:
            raw_data = json.load(f)

        events = raw_data.get("events", [])
        filtered = [
            e for e in events
            if e.get("confidence", 0) >= self.confidence_threshold
        ]
        filtered.sort(key=lambda e: (e.get("start_time", 0), e.get("end_time", 0)))
        return raw_data, filtered

    # ------------------------------------------------------------------
    # Step 2: Compute Temporal Relations (adjacent events only)
    # ------------------------------------------------------------------

    def compute_temporal_relations(self, events: List[dict]) -> list:
        """Find overlaps and adjacencies between events."""
        relations = []

        for i, ev_a in enumerate(events):
            a_id = ev_a.get("event_id")
            a_end = ev_a.get("end_time", 0)

            for j in range(i + 1, len(events)):
                ev_b = events[j]
                b_id = ev_b.get("event_id")
                b_start = ev_b.get("start_time", 0)
                b_end = ev_b.get("end_time", 0)

                if b_start - a_end > ADJACENCY_THRESHOLD_SEC:
                    break

                if a_end > b_start:
                    overlap = min(a_end, b_end) - b_start
                    relations.append({
                        "type": "overlaps",
                        "event_a": a_id,
                        "event_b": b_id,
                        "overlap_seconds": round(overlap, 2),
                    })
                else:
                    gap = b_start - a_end
                    relations.append({
                        "type": "immediately_follows",
                        "from_event_id": a_id,
                        "to_event_id": b_id,
                        "gap_seconds": round(gap, 2),
                    })

        return relations

    # ------------------------------------------------------------------
    # Step 3: Gemini Call — per-event descriptions
    # ------------------------------------------------------------------

    def _extract_social_details(self, ev: dict) -> dict:
        """Extract only person-relationship details, drop raw numerical values."""
        details = ev.get("details", {})
        event_type = ev.get("event_type", "")
        social = {}
        if event_type == "gaze_following":
            social["leader_id"] = details.get("leader_id")
            social["follower_id"] = details.get("follower_id")
            if details.get("lag_seconds"):
                social["lag_seconds"] = details["lag_seconds"]
        elif event_type == "joint_attention":
            social["num_persons"] = details.get("num_persons")
        elif event_type == "attention_capture":
            social["num_persons"] = details.get("num_persons")
        elif event_type == "mutual_gaze":
            social["duration"] = details.get("duration")
        return social

    def build_prompt(self, sorted_events: List[dict]) -> str:
        """Construct Gemini prompt asking for per-event descriptions."""
        events_for_prompt = []
        for ev in sorted_events:
            entry = {
                "event_id": ev.get("event_id"),
                "event_type": ev.get("event_type"),
                "start_time": ev.get("start_time"),
                "end_time": ev.get("end_time"),
                "persons_involved": ev.get("persons_involved"),
            }
            social_details = self._extract_social_details(ev)
            if social_details:
                entry["details"] = social_details
            events_for_prompt.append(entry)

        events_json = json.dumps(events_for_prompt, indent=2)

        prompt = f"""You are writing factual descriptions for social gaze events detected by a computer vision pipeline.

## Event Type Definitions

- **sudden_gaze_shift**: A person rapidly changed where they're looking.
- **joint_attention**: Multiple people looking at the same target simultaneously.
- **gaze_following**: One person (follower) looks where another person (leader) was looking, after a time delay.
- **attention_capture**: Multiple people suddenly shifted gaze to the same location simultaneously.
- **mutual_gaze**: Two people looking directly at each other (eye contact).

## Events (sorted by time, confidence >= {self.confidence_threshold})

{events_json}

## Task

Write a **one-sentence factual description** for each event.

Rules:
1. State WHAT happened: who did what to whom, when (start/end time)
2. Only mention: persons involved, time range, and the social relationship (e.g. who followed whom, how many people looked together)
3. **Do NOT include any raw numerical values** such as velocity, convergence score, gaze distance, or coordinates
4. **ONLY describe what the data shows.** Do NOT speculate or add interpretation.
5. **NEVER** use words like: "suggesting", "likely", "indicating", "possibly", "implying", "appears to", "seems to", "may have", "probably"
6. **NEVER** infer causes or motivations (e.g. "reacting to an external stimulus", "caught their interest", "a shared distraction")

Good examples:
→ "From 5.0s to 6.5s, Person 3 followed Person 0's gaze direction after a 1.5-second delay."
→ "At 13.0s, Person 0, Person 2, and Person 3 simultaneously shifted their gaze to the same location."
→ "From 42.0s to 43.5s, Person 0 and Person 1 maintained mutual eye contact for 1.5 seconds."

Bad examples (DO NOT write like this):
→ "...with a mean velocity of approximately 0.49."
→ "...with a convergence score of 0.85."
→ "...suggesting Person 0 directed attention to something."

## Output Format (JSON)

Return a JSON object mapping event_id (as string) to its description:

{{
  "4000": "From 0.0s to 1.5s, Person 1 and Person 3 ...",
  "2015": "From 2.0s to 3.5s, Person 2 ...",
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

    def generate_graph(self, events_json_path: str, output_path: str) -> dict:
        """Full pipeline: load → filter → Gemini descriptions → temporal relations → save."""
        print(f"  Loading: {os.path.basename(events_json_path)}")
        raw_data, filtered_events = self.load_and_filter(events_json_path)
        total_events = len(raw_data.get("events", []))
        num_filtered = len(filtered_events)
        print(f"  Events: {total_events} total, {num_filtered} filtered (>= {self.confidence_threshold})")

        # Compute temporal relations
        temporal_relations = self.compute_temporal_relations(filtered_events)

        # Timeline range
        if filtered_events:
            min_time = filtered_events[0].get("start_time", 0)
            max_time = max(e.get("end_time", 0) for e in filtered_events)
        else:
            min_time, max_time = 0.0, 0.0

        # Gemini: generate per-event descriptions
        if num_filtered > 0:
            events_for_prompt = filtered_events
            if num_filtered > MAX_EVENTS_FOR_PROMPT:
                print(f"  Truncating {num_filtered} events to top {MAX_EVENTS_FOR_PROMPT} by confidence")
                events_for_prompt = sorted(
                    filtered_events, key=lambda e: e.get("confidence", 0), reverse=True
                )[:MAX_EVENTS_FOR_PROMPT]
                events_for_prompt.sort(key=lambda e: (e.get("start_time", 0), e.get("end_time", 0)))

            prompt = self.build_prompt(events_for_prompt)
            print(f"  Calling Gemini ({self.model_name}) for {len(events_for_prompt)} event descriptions...")

            try:
                descriptions = self._call_gemini(prompt)
                matched = 0
                for ev in filtered_events:
                    eid = str(ev.get("event_id"))
                    if eid in descriptions:
                        ev["description"] = descriptions[eid]
                        matched += 1
                    else:
                        ev["description"] = None
                print(f"  Gemini OK: {matched}/{num_filtered} events got descriptions")
            except Exception as e:
                print(f"  Gemini call failed: {e}")
                for ev in filtered_events:
                    ev["description"] = None

        output = {
            "video_path": raw_data.get("video_path", ""),
            "source_event_file": events_json_path,
            "num_events_total": total_events,
            "num_events_filtered": num_filtered,
            "confidence_threshold": self.confidence_threshold,
            "timeline": {
                "duration_covered": [round(min_time, 2), round(max_time, 2)],
                "events_sorted": filtered_events,
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
    generator: EventGraphGenerator,
    skip_existing: bool = False,
    max_files: int = 0,
    sample_ratio: float = 1.0,
) -> Tuple[int, int, int]:
    """Process all event files in a dataset."""
    event_dir = os.path.join(base_dir, dataset_name, "event_data")
    output_dir = os.path.join(base_dir, dataset_name, "gaze_graph")

    if not os.path.isdir(event_dir):
        print(f"Event data directory not found: {event_dir}")
        return 0, 0, 0

    event_files = sorted(glob.glob(os.path.join(event_dir, "*_events.json")))
    total = len(event_files)

    if total == 0:
        print(f"No event files found in: {event_dir}")
        return 0, 0, 0

    sample_n = total
    if max_files > 0 and max_files < total:
        sample_n = max_files
    elif 0 < sample_ratio < 1.0:
        sample_n = max(1, int(total * sample_ratio))

    if sample_n < total:
        random.seed(42)
        event_files = random.sample(event_files, sample_n)
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset_name} (sampled {sample_n}/{total} files)")
    else:
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset_name} ({total} files)")

    print(f"Input:   {event_dir}")
    print(f"Output:  {output_dir}")
    print(f"{'='*60}")

    to_process = len(event_files)
    processed, skipped, failed = 0, 0, 0

    for i, events_json in enumerate(event_files):
        base = os.path.basename(events_json)
        video_name = base
        for suffix in ["_sam3rf_gaze_events.json", "_gaze_events.json", "_events.json"]:
            if video_name.endswith(suffix):
                video_name = video_name[:-len(suffix)]
                break
        output_path = os.path.join(output_dir, f"{video_name}_event_graph.json")

        if skip_existing and os.path.exists(output_path):
            skipped += 1
            continue

        print(f"\n[{i+1}/{to_process}] {video_name}")

        try:
            generator.generate_graph(events_json, output_path)
            processed += 1
        except Exception as e:
            print(f"  FAILED: {e}")
            failed += 1

        time.sleep(0.5)

    print(f"\n{dataset_name} done: {processed} processed, {skipped} skipped, {failed} failed")
    return processed, skipped, failed


def main():
    parser = argparse.ArgumentParser(
        description="Generate event graphs with per-event descriptions from raw gaze event data"
    )

    parser.add_argument("--base_dir", type=str, default=DEFAULT_BASE_PATH)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--all", action="store_true")

    parser.add_argument("--events_json", type=str, default=None)
    parser.add_argument("--output_json", type=str, default=None)

    parser.add_argument("--confidence", type=float, default=0.9)
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

    generator = EventGraphGenerator(
        api_key=api_key,
        model_name=args.model,
        confidence_threshold=args.confidence,
    )

    if args.events_json:
        if not os.path.exists(args.events_json):
            print(f"ERROR: File not found: {args.events_json}")
            sys.exit(1)

        output_path = args.output_json
        if not output_path:
            base = os.path.splitext(args.events_json)[0]
            for suffix in ["_sam3rf_gaze_events", "_gaze_events", "_events"]:
                if base.endswith(suffix):
                    base = base[:-len(suffix)]
                    break
            output_path = f"{base}_event_graph.json"

        print(f"Single file mode")
        print(f"Input:  {args.events_json}")
        print(f"Output: {output_path}")
        generator.generate_graph(args.events_json, output_path)
        return

    if args.all:
        datasets = DATASETS
    elif args.dataset:
        datasets = [args.dataset]
    else:
        parser.print_help()
        print("\nProvide --dataset, --all, or --events_json")
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
