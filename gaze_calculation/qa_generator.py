#!/usr/bin/env python3
"""
QA Generator (Stage 9) - Generate QA pairs from graph_data using Gemini.

Reads graph_data JSON (events + cross_modal_pairs), selects appropriate
prompts (A-E) based on event density, calls Gemini Flash to generate
QA pairs, validates output, and saves per-video QA JSON.

Input:  {dataset}/graph_data/{video}_graph.json
Output: {dataset}/qa_data/{video}_qa.json

Usage:
    # Single dataset
    python -m gaze_calculation.qa_generator --dataset embody3d --skip_existing

    # All datasets
    python -m gaze_calculation.qa_generator --all --skip_existing

    # Sample 10 files
    python -m gaze_calculation.qa_generator --dataset embody3d --max_files 10
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
from collections import Counter
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

CATEGORY_DIFFICULTY = {
    "T1": "easy", "T2": "easy", "T3": "medium", "T4": "medium",
    "T5": "hard", "T6": "hard",
    "G1": "easy", "G2": "easy", "G3": "medium", "G4": "medium",
    "G5": "hard", "G6": "hard",
    "C1": "medium", "C2": "medium", "C3": "hard", "C4": "hard",
}

# Categories that use open-ended format (multi-event synthesis / narrative)
# All other categories use MCQ only
OPEN_ENDED_CATEGORIES = {"T6", "G5", "C3", "C4"}

BANNED_WORDS = [
    "suggesting", "indicating", "likely", "because",
    "implies", "emotion", "feeling", "probably", "seems",
]


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
                        print(f"  API Error ({msg[:80]}). Retrying in {sleep:.1f}s...")
                        time.sleep(sleep)
                        delay *= backoff_factor
                    else:
                        raise
            return func(*args, **kwargs)
        return wrapper
    return decorator


# ------------------------------------------------------------------
# Prompt Templates
# ------------------------------------------------------------------

MASTER_SYSTEM_PROMPT = """You are a QA generator for a non-verbal social reasoning dataset used to train Video-LLMs via reinforcement learning (GRPO).

You will receive structured event annotations detected in a video clip. Your job: generate question-answer pairs that the model must answer by WATCHING THE VIDEO.

=== STRICT OUTPUT RULES ===

1. ANSWER FORMAT — two types only:
   - MCQ: answer field = single letter (A/B/C/D). answer_text = full text of correct option.
   - Open-ended: answer field = answer_text = 2-4 sentence descriptive answer. Must reference specific Person IDs and timestamps from the event data.

2. ANSWER CONTENT:
   - Factual, no reasoning. NEVER include these words in answers:
     "suggesting", "indicating", "likely", "because", "implies", "seems", "probably", "emotion", "feeling"

3. QUESTION QUALITY:
   - Must require WATCHING THE VIDEO to answer. Not answerable from question text alone.
   - Use natural language: say "eye contact" not "mutual gaze event", "points at" not "performs a pointing gesture", "looking at the same thing" not "joint attention with high convergence".
   - Include specific timestamps from the event data.
   - All person references: "Person {id}" format.

4. MCQ RULES:
   - Exactly 4 options (A/B/C/D).
   - Distractors must be plausible: other person IDs in the video, other event types that didn't happen, swapped roles, nearby but incorrect timestamps.
   - Option text must be concise (<=15 words per option).

5. OPEN-ENDED RULES:
   - No options field.
   - Answer must be 2-4 sentences describing observable facts grounded in event timestamps and person IDs.
   - State facts only — no interpretation, no hedging, no speculation.
   - Open-ended is ONLY for categories: T6, G5, C3, C4. All other categories MUST be MCQ.

6. REASONING:
   - Every QA must include a "reasoning" field with step-by-step process.
   - Format: "<think> step-by-step reasoning here </think>"
   - Explain: which event(s) you used, how you derived the answer, why distractors are wrong (MCQ) or what facts the answer covers (open-ended).
   - This is for training data quality — the reasoning is NOT shown to the Video-LLM.

7. OUTPUT FORMAT:
   Return ONLY a JSON array. No explanation, no markdown, no commentary.
   Each element:
   {
     "category": "T1",
     "difficulty": "easy",
     "format": "mcq" | "open_ended",
     "question": "...",
     "options": ["A) ...", "B) ...", "C) ...", "D) ..."],
     "answer": "C",
     "answer_text": "Person 3",
     "reasoning": "<think> Used mutual_gaze event 4000 (P1,P3, 13.0-15.5s). At 14.0s P1 is looking at P3. Distractors: P0, P2 are other persons in video, 'no one' is implausible given mutual_gaze. </think>",
     "source_event_ids": [4000],
     "time_range": [13.0, 15.5]
   }
   - "options" field: ONLY for mcq. Omit entirely for open_ended.
   - "source_event_ids": gaze event_id(s) or gesture index(es) used as GT.
   - "time_range": [start, end] of relevant events."""


def _build_prompt_a(metadata: dict, gaze_events: list) -> str:
    """Prompt A: Few-event gaze (1-2 events -> T1, T2) -> 2-3 QA."""
    return f"""{MASTER_SYSTEM_PROMPT}

TASK: Generate 2-3 QA pairs from the gaze event(s) below.
This video has few gaze events, so focus on EASY categories only.

=== CATEGORIES ===

T1 (Gaze Target Identification) [Easy]:
  "Who is Person X looking at at time T?"
  - Tests: unidirectional gaze direction perception.
  - GT: persons_involved, leader_id/follower_id from event data.
  - Format: MCQ. Distractors = other person IDs in video.

T2 (Gaze Event Classification) [Easy]:
  "What best describes the gaze interaction at time T?"
  - Tests: recognizing gaze event type from visual observation.
  - GT: event_type field.
  - Format: MCQ. Distractors = other gaze event types described in natural language.
  - IMPORTANT: describe event types naturally, never use technical terms.
    mutual_gaze -> "looking at each other" / "eye contact"
    joint_attention -> "looking at the same thing" / "shared attention"
    gaze_following -> "following someone's gaze"
    attention_capture -> "everyone suddenly shifts their gaze"
    sudden_gaze_shift -> "quickly shifts gaze direction"

=== FORMAT ===
ALL questions must be MCQ. No short_answer or open_ended.

=== NOW GENERATE ===

VIDEO METADATA:
- video_id: {metadata['video_id']}
- dataset: {metadata['dataset']}
- persons_in_video: {metadata['person_ids']}
- video_duration: {metadata['duration']}s

GAZE EVENTS:
{json.dumps(gaze_events, indent=2)}

Generate the QA pairs now. Return ONLY the JSON array."""


def _build_prompt_b(metadata: dict, gaze_events: list) -> str:
    """Prompt B: Full gaze (3+ events -> T1-T6) -> 5-8 QA."""
    event_types = {e["event_type"] for e in gaze_events}
    has_t5 = "gaze_following" in event_types
    has_t6 = bool(event_types & {"joint_attention", "attention_capture"})

    t5_note = "" if has_t5 else "\n  NOTE: No gaze_following events — SKIP T5."
    t6_note = "" if has_t6 else "\n  NOTE: No joint_attention/attention_capture — SKIP T6."

    return f"""{MASTER_SYSTEM_PROMPT}

TASK: Generate 5-8 QA pairs from the gaze events below.
Cover ALL difficulty levels with the category mix specified.

=== CATEGORIES (generate at least 1 from each difficulty tier) ===

[EASY — single event, direct lookup]

T1 (Gaze Target Identification):
  "Who is Person X looking at at time T?"
  GT: persons_involved. Distractors: other person IDs.

T2 (Gaze Event Classification):
  "What best describes the gaze interaction at time T?"
  GT: event_type. Distractors: other event types in natural language.
  Natural language mapping:
    mutual_gaze -> "eye contact" / "looking at each other"
    joint_attention -> "looking at the same thing"
    gaze_following -> "following someone's gaze"
    attention_capture -> "everyone suddenly shifts gaze"
    sudden_gaze_shift -> "quickly changes gaze direction"

[MEDIUM — cross-event comparison or bidirectional reasoning]

T3 (Temporal Gaze Reasoning):
  Duration: "How long do Person X and Person Y maintain eye contact?"
  Ordering: "Between Person X and Person Y, which gaze event happens first: eye contact or gaze following?"
  IMPORTANT: Always scope to a specific person pair. Never ask "which happens first" across all events.
  GT: start_time, end_time, duration. Distractors: swapped order, wrong durations.

T4 (Mutual Gaze Recognition):
  "Which pair of people makes eye contact?"
  GT: mutual_gaze persons_involved (exactly 2). Distractors: other possible pairs.

[HARD — multi-step reasoning, multi-event aggregation]

T5 (Gaze Following & Social Influence):
  "Who looks first at the target?" / "Is the gaze-following one-directional?"
  GT: gaze_following leader_id, follower_id. Distractors: swap roles.
  ONLY generate if gaze_following events exist.{t5_note}

T6 (Group Attention Dynamics):
  "Who is NOT part of the shared attention?" / "What happens after the group attention ends?"
  GT: joint_attention/attention_capture persons_involved, event timeline.
  ONLY generate if joint_attention or attention_capture events exist.{t6_note}

=== FORMAT ===
ALL questions must be MCQ. No open_ended in this prompt.

=== NOW GENERATE ===

VIDEO METADATA:
- video_id: {metadata['video_id']}
- dataset: {metadata['dataset']}
- persons_in_video: {metadata['person_ids']}
- video_duration: {metadata['duration']}s

GAZE EVENTS:
{json.dumps(gaze_events, indent=2)}

Generate the QA pairs now. Return ONLY the JSON array."""


def _build_prompt_c(metadata: dict, gesture_events: list) -> str:
    """Prompt C: Few-gesture (1-3 gestures -> G1, G2) -> 2-3 QA."""
    return f"""{MASTER_SYSTEM_PROMPT}

TASK: Generate 2-3 QA pairs from the gesture annotation(s) below.
This video has few gestures, so focus on EASY categories only.

=== CATEGORIES ===

G1 (Gesture Recognition) [Easy]:
  "What type of hand gesture does Person X perform at T?" / "Who is Person X pointing at?"
  - Tests: type + initiator + target of a single gesture.
  - GT: gesture_type, initiator_id, target_person_id, target_type.
  - Format: MCQ preferred.
  - Natural language: "points at" / "hands something to" / "holds up an object toward" / "reaches for"

G2 (Gesture Type Classification) [Easy]:
  "What type of gesture does Person X perform at T seconds?"
  - Tests: classifying the gesture type at a specific moment.
  - GT: gesture_type. Options MUST be the 4 gesture types: A) Pointing B) Giving C) Showing D) Reaching.
  - Format: MCQ only (always 4-choice with the 4 gesture types).

IMPORTANT: "target_description" is free text for your context. NEVER use it as the answer.

=== FORMAT ===
ALL questions must be MCQ. No short_answer or open_ended.

=== NOW GENERATE ===

VIDEO METADATA:
- video_id: {metadata['video_id']}
- dataset: {metadata['dataset']}
- persons_in_video: {metadata['person_ids']}
- video_duration: {metadata['duration']}s

GESTURE ANNOTATIONS:
{json.dumps(gesture_events, indent=2)}

Generate the QA pairs now. Return ONLY the JSON array."""


def _build_prompt_d(metadata: dict, gesture_events: list) -> str:
    """Prompt D: Full gesture (4+ gestures -> G1-G6) -> 4-7 QA."""
    # Check if G5/G6 are feasible
    p2p = [g for g in gesture_events
           if g.get("target_type") == "person" and g.get("target_person_id") is not None]
    has_g5_g6 = len(p2p) >= 2

    g5_note = "" if has_g5_g6 else "\n  NOTE: Not enough person-to-person gestures — SKIP G5."
    g6_note = "" if has_g5_g6 else "\n  NOTE: Not enough person-to-person gestures — SKIP G6."

    return f"""{MASTER_SYSTEM_PROMPT}

TASK: Generate 4-7 QA pairs from the gesture annotations below.
Cover ALL difficulty levels with the category mix specified.

=== CATEGORIES (generate at least 1 from each difficulty tier) ===

[EASY — single gesture, direct lookup]

G1 (Gesture Recognition):
  "What gesture does Person X perform at T?" / "Who is Person X pointing at?"
  GT: gesture_type, initiator_id, target_person_id, target_type.

G2 (Gesture Type Classification):
  "What type of gesture does Person X perform at T seconds?"
  GT: gesture_type. Options MUST be: A) Pointing B) Giving C) Showing D) Reaching. MCQ only.

[MEDIUM — multi-gesture comparison or aggregation]

G3 (Gesture Temporal Reasoning):
  "Which happens first: X or Y?" / "How long does the gesture last?"
  GT: start_time, end_time comparison.

G4 (Gesture Frequency & Distribution):
  "Who performs the most gestures?" / "What is the most common gesture type?"
  GT: count by initiator_id or gesture_type -> argmax.

[HARD — cross-gesture chaining or bidirectional reasoning]

G5 (Gesture Sequence Chains):
  "P1 gives to P4, then P4 shows to someone. Who?" / "What gesture follows X?"
  GT: temporal sequence + participant tracking across gestures.
  ONLY generate if 2+ person-to-person gestures share a participant.{g5_note}

G6 (Reciprocal Gesture Patterns):
  "P3 points at P4. Does P4 gesture back toward P3?"
  GT: cross-reference initiator<->target across events.
  ONLY generate if 2+ person-to-person gestures involve overlapping participant pairs.{g6_note}

IMPORTANT: "target_description" is context only. NEVER use as answer.

=== FORMAT ===
ALL questions must be MCQ. No open_ended in this prompt.

=== NOW GENERATE ===

VIDEO METADATA:
- video_id: {metadata['video_id']}
- dataset: {metadata['dataset']}
- persons_in_video: {metadata['person_ids']}
- video_duration: {metadata['duration']}s

GESTURE ANNOTATIONS:
{json.dumps(gesture_events, indent=2)}

Generate the QA pairs now. Return ONLY the JSON array."""


def _build_prompt_e(metadata: dict, gaze_events: list, gesture_events: list) -> str:
    """Prompt E: Cross-modal (C1-C4) -> 3-5 QA."""
    return f"""{MASTER_SYSTEM_PROMPT}

TASK: Generate 3-5 cross-modal QA pairs from the gaze AND gesture data below.
Every question MUST require information from BOTH gaze and gesture data to answer.

=== CROSS-MODAL REQUIREMENT ===
- BAD (single modality): "What gesture does P0 do at 21s?" -> answerable from gesture alone
- GOOD (cross-modal): "When P1 gives something to P4 at 4s, are they making eye contact?" -> needs BOTH

=== MATCHING RULES ===
- "Co-occur": time ranges overlap by any amount.
- "Adjacent": one ends within 3 seconds of the other starting.
- "Person match": gesture initiator/target in gaze persons_involved.

=== CATEGORIES ===

[MEDIUM — timestamp overlap or temporal order across modalities]

C1 (Gaze-Gesture Temporal Alignment):
  "While Person X is handing something to Person Y, is there a gaze interaction?"
  "Which happens first: the gaze shift or the reaching gesture?"
  GT: timestamp comparison across gaze_events and gesture_events.

C2 (Eye Contact During Interaction):
  "When Person X gives something to Person Y, are they making eye contact?"
  GT: mutual_gaze overlapping with giving/showing + same participant pair.

[HARD — causal/response chain or person-level integration]

C3 (Gaze Response to Gesture):
  "After Person X points at Person Y, do others shift their gaze?"
  "What gesture happens right after the group attention event ends?"
  GT: gesture end_time -> nearby gaze event (or vice versa), participant match.

C4 (Cross-Modal Person Dynamics):
  "Person X points at Person Y. Is Person Y involved in any gaze event at that time?"
  "Who appears in the most events across both gaze and gesture data?"
  GT: person presence check across both modalities.

=== FORMAT ===
ALL questions must be MCQ. No open_ended in this prompt.

=== NOW GENERATE ===

VIDEO METADATA:
- video_id: {metadata['video_id']}
- dataset: {metadata['dataset']}
- persons_in_video: {metadata['person_ids']}
- video_duration: {metadata['duration']}s

GAZE EVENTS:
{json.dumps(gaze_events, indent=2)}

GESTURE EVENTS:
{json.dumps(gesture_events, indent=2)}

Generate the QA pairs now. Return ONLY the JSON array."""


def _build_prompt_b_oe(metadata: dict, gaze_events: list) -> str:
    """Prompt B-OE: Open-ended T6 only from gaze events -> 1-2 QA."""
    return f"""{MASTER_SYSTEM_PROMPT}

TASK: Generate 1-2 OPEN-ENDED QA pairs about group attention dynamics (T6).

=== CATEGORY ===

T6 (Group Attention Dynamics) [OPEN-ENDED]:
  Ask the model to DESCRIBE the group's attention structure — who shares attention, who is excluded, and what changes over time.
  Example questions:
    "Describe the group's attention pattern from Ts to Te."
    "What happens to the group's gaze after the shared focus ends at Ts?"
  GT: joint_attention/attention_capture persons_involved, event timeline.
  Answer: 2-4 sentences describing observable facts with Person IDs and timestamps.

=== FORMAT ===
ALL questions must be open_ended. No MCQ in this prompt.

=== NOW GENERATE ===

VIDEO METADATA:
- video_id: {metadata['video_id']}
- dataset: {metadata['dataset']}
- persons_in_video: {metadata['person_ids']}
- video_duration: {metadata['duration']}s

GAZE EVENTS:
{json.dumps(gaze_events, indent=2)}

Generate the QA pairs now. Return ONLY the JSON array."""


def _build_prompt_d_oe(metadata: dict, gesture_events: list) -> str:
    """Prompt D-OE: Open-ended G5 only from gesture events -> 1-2 QA."""
    return f"""{MASTER_SYSTEM_PROMPT}

TASK: Generate 1-2 OPEN-ENDED QA pairs about gesture sequence chains (G5).

=== CATEGORY ===

G5 (Gesture Sequence Chains) [OPEN-ENDED]:
  Ask the model to DESCRIBE a sequence of gestures involving shared participants — object transfer chains, action sequences.
  Example questions:
    "Describe the sequence of gestures involving Person X from Ts to Te."
    "Trace the object transfer chain starting from Person X's gesture at Ts."
  GT: temporal sequence + participant tracking across gestures.
  Answer: 2-4 sentences describing the gesture chain with Person IDs and timestamps.

IMPORTANT: "target_description" is context only. NEVER use as answer.

=== FORMAT ===
ALL questions must be open_ended. No MCQ in this prompt.

=== NOW GENERATE ===

VIDEO METADATA:
- video_id: {metadata['video_id']}
- dataset: {metadata['dataset']}
- persons_in_video: {metadata['person_ids']}
- video_duration: {metadata['duration']}s

GESTURE ANNOTATIONS:
{json.dumps(gesture_events, indent=2)}

Generate the QA pairs now. Return ONLY the JSON array."""


def _build_prompt_e_oe(metadata: dict, gaze_events: list, gesture_events: list) -> str:
    """Prompt E-OE: Open-ended C3, C4 only from cross-modal data -> 2-3 QA."""
    return f"""{MASTER_SYSTEM_PROMPT}

TASK: Generate 2-3 OPEN-ENDED cross-modal QA pairs (C3, C4).
Every question MUST require information from BOTH gaze and gesture data.

=== CATEGORIES ===

C3 (Gaze Response to Gesture) [OPEN-ENDED]:
  Ask the model to DESCRIBE the causal chain between gesture and gaze response.
  Example questions:
    "Describe how the group's gaze changes after Person X points at Person Y at Ts."
    "What happens across gaze and gesture around the Ts-Te window?"
  GT: gesture end_time -> nearby gaze event, participant match.
  Answer: 2-4 sentences describing the cross-modal causal chain.

C4 (Cross-Modal Person Dynamics) [OPEN-ENDED]:
  Ask the model to DESCRIBE a person's role across both modalities.
  Example questions:
    "Describe Person X's involvement in both gaze and gesture interactions."
    "How do gaze and gesture interactions relate around Ts?"
  GT: person presence check across both modalities.
  Answer: 2-4 sentences integrating gaze and gesture behavior.

=== FORMAT ===
ALL questions must be open_ended. No MCQ in this prompt.

=== NOW GENERATE ===

VIDEO METADATA:
- video_id: {metadata['video_id']}
- dataset: {metadata['dataset']}
- persons_in_video: {metadata['person_ids']}
- video_duration: {metadata['duration']}s

GAZE EVENTS:
{json.dumps(gaze_events, indent=2)}

GESTURE EVENTS:
{json.dumps(gesture_events, indent=2)}

Generate the QA pairs now. Return ONLY the JSON array."""


# ------------------------------------------------------------------
# Prompt Selection
# ------------------------------------------------------------------

def _extract_person_ids(events: list) -> list:
    """Extract all unique person IDs from events."""
    pids = set()
    for ev in events:
        if ev.get("source") == "gaze":
            for pid in ev.get("persons_involved", []):
                pids.add(pid)
        else:
            if ev.get("initiator_id") is not None:
                pids.add(ev["initiator_id"])
            if ev.get("target_person_id") is not None:
                pids.add(ev["target_person_id"])
    return sorted(pids)


def _estimate_duration(events: list) -> float:
    """Estimate video duration from event timestamps."""
    if not events:
        return 0.0
    max_t = max(ev.get("end_time", ev.get("start_time", 0)) for ev in events)
    return round(max_t + 5.0, 1)  # add buffer


def _prepare_gaze_event(ev: dict) -> dict:
    """Prepare a gaze event for the prompt (strip description, keep GT fields)."""
    out = {
        "event_id": ev.get("event_id"),
        "event_type": ev.get("event_type"),
        "start_time": ev.get("start_time"),
        "end_time": ev.get("end_time"),
        "persons_involved": ev.get("persons_involved"),
    }
    etype = ev.get("event_type", "")
    if etype == "gaze_following":
        out["leader_id"] = ev.get("leader_id")
        out["follower_id"] = ev.get("follower_id")
        out["lag_seconds"] = ev.get("lag_seconds")
    elif etype == "mutual_gaze":
        out["gaze_duration"] = ev.get("gaze_duration")
    elif etype in ("joint_attention", "attention_capture"):
        out["num_persons"] = ev.get("num_persons", len(ev.get("persons_involved", [])))
    return out


def _prepare_gesture_event(ev: dict) -> dict:
    """Prepare a gesture event for the prompt."""
    out = {
        "gesture_index": ev.get("gesture_index"),
        "gesture_type": ev.get("gesture_type"),
        "start_time": ev.get("start_time"),
        "end_time": ev.get("end_time"),
        "initiator_id": ev.get("initiator_id"),
        "target_type": ev.get("target_type"),
        "target_person_id": ev.get("target_person_id"),
        "target_description": ev.get("target_description"),
        "confidence": ev.get("confidence"),
    }
    return out


def select_prompts(
    graph_data: dict, video_id: str, dataset: str,
) -> List[Tuple[str, str]]:
    """Select which prompts to call based on event density.

    Returns list of (prompt_name, prompt_text) tuples.
    """
    events = graph_data.get("events", [])
    cross_pairs = graph_data.get("cross_modal_pairs", [])

    gaze_events = [ev for ev in events if ev.get("source") == "gaze"]
    gesture_events = [ev for ev in events if ev.get("source") == "gesture"]

    person_ids = _extract_person_ids(events)
    duration = _estimate_duration(events)

    metadata = {
        "video_id": video_id,
        "dataset": dataset,
        "person_ids": person_ids,
        "duration": duration,
    }

    # Prepare clean events for prompts
    gaze_clean = [_prepare_gaze_event(ev) for ev in gaze_events]
    gesture_clean = [_prepare_gesture_event(ev) for ev in gesture_events]

    prompts = []

    # Gaze prompts
    if len(gaze_clean) > 0:
        if len(gaze_clean) <= 2:
            prompts.append(("prompt_A", _build_prompt_a(metadata, gaze_clean)))
        else:
            prompts.append(("prompt_B", _build_prompt_b(metadata, gaze_clean)))
            # T6 open-ended (separate call) — only if joint_attention or attention_capture exist
            event_types = {e["event_type"] for e in gaze_clean}
            if event_types & {"joint_attention", "attention_capture"}:
                prompts.append(("prompt_B_OE", _build_prompt_b_oe(metadata, gaze_clean)))

    # Gesture prompts
    if len(gesture_clean) > 0:
        if len(gesture_clean) <= 3:
            prompts.append(("prompt_C", _build_prompt_c(metadata, gesture_clean)))
        else:
            prompts.append(("prompt_D", _build_prompt_d(metadata, gesture_clean)))
            # G5 open-ended (separate call) — only if 2+ person-to-person gestures share a participant
            p2p = [g for g in gesture_clean
                   if g.get("target_type") == "person" and g.get("target_person_id") is not None]
            if len(p2p) >= 2:
                prompts.append(("prompt_D_OE", _build_prompt_d_oe(metadata, gesture_clean)))

    # Cross-modal prompt
    if len(gaze_clean) > 0 and len(gesture_clean) > 0 and len(cross_pairs) >= 2:
        prompts.append(("prompt_E", _build_prompt_e(metadata, gaze_clean, gesture_clean)))
        # C3/C4 open-ended (separate call)
        prompts.append(("prompt_E_OE", _build_prompt_e_oe(metadata, gaze_clean, gesture_clean)))

    return prompts


# ------------------------------------------------------------------
# Validation
# ------------------------------------------------------------------

def validate_qa(
    qa_pairs: list, person_ids: list, mode: str,
    expected_format: str = "mcq",
) -> list:
    """Validate generated QA pairs.

    mode = 'gaze' | 'gesture' | 'cross'
    expected_format = 'mcq' | 'open_ended'
    """
    valid = []
    seen_questions = []

    for qa in qa_pairs:
        if not isinstance(qa, dict):
            continue

        cat = qa.get("category", "")

        # Enforce the expected format from the prompt call
        fmt = expected_format
        qa["format"] = fmt
        if fmt == "open_ended":
            qa.pop("options", None)

        # Format-specific checks
        if fmt == "mcq":
            if len(qa.get("answer", "")) != 1 or qa.get("answer", "") not in "ABCD":
                continue
            if len(qa.get("options", [])) != 4:
                continue
        elif fmt == "open_ended":
            answer_text = qa.get("answer_text", "")
            if len(answer_text.split()) < 5:
                continue  # too short for open-ended
            qa["answer"] = answer_text  # sync answer = answer_text
        else:
            continue

        # Banned words
        answer_text = qa.get("answer_text", "").lower()
        if any(w in answer_text for w in BANNED_WORDS):
            continue

        # Person ID validation
        text = qa.get("question", "") + " " + qa.get("answer_text", "")
        for opt in qa.get("options", []):
            text += " " + opt
        mentioned = re.findall(r'Person (\d+)', text)
        if person_ids and any(int(pid) not in person_ids for pid in mentioned):
            continue

        # Category validation (cat already set above)
        if mode == "gaze" and not cat.startswith("T"):
            continue
        if mode == "gesture" and not cat.startswith("G"):
            continue
        if mode == "cross" and not cat.startswith("C"):
            continue

        # time_range required
        tr = qa.get("time_range")
        if not tr or not isinstance(tr, list) or len(tr) < 2:
            continue

        # Fix difficulty
        expected = CATEGORY_DIFFICULTY.get(cat)
        if expected:
            qa["difficulty"] = expected

        # Cross-modal check
        if mode == "cross":
            q = qa.get("question", "").lower()
            gaze_words = ["look", "gaze", "eye contact", "attention", "watching"]
            gesture_words = ["point", "give", "hand", "show", "reach", "gesture"]
            if not (any(w in q for w in gaze_words) and any(w in q for w in gesture_words)):
                continue

        # Dedup
        q_words = set(qa.get("question", "").lower().split())
        is_dup = False
        for prev in seen_questions:
            overlap = len(q_words & prev) / max(len(q_words | prev), 1)
            if overlap > 0.6:
                is_dup = True
                break
        if is_dup:
            continue
        seen_questions.append(q_words)

        valid.append(qa)

    return valid


# ------------------------------------------------------------------
# QA Generator
# ------------------------------------------------------------------

class QAGenerator:
    """Generate QA pairs from graph_data using Gemini."""

    def __init__(
        self,
        api_key: str,
        model_name: str = "models/gemini-3.1-pro-preview",
    ):
        self.api_key = api_key
        self.model_name = model_name

        if GEMINI_AVAILABLE:
            self.client = genai.Client(api_key=api_key)
        else:
            self.client = None

    @retry_with_backoff()
    def _call_gemini(self, prompt: str) -> list:
        """Send prompt to Gemini and parse JSON array response."""
        if not self.client:
            return []

        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.7,
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
            # Try to extract JSON array
            match = re.search(r'\[.*\]', raw_text, re.DOTALL)
            if match:
                try:
                    parsed = json.loads(match.group())
                except json.JSONDecodeError:
                    parsed = []
            else:
                parsed = []

        if isinstance(parsed, dict):
            # Sometimes model wraps in {"qa_pairs": [...]}
            for key in parsed:
                if isinstance(parsed[key], list):
                    parsed = parsed[key]
                    break
            else:
                parsed = []

        if not isinstance(parsed, list):
            parsed = []

        return parsed

    def generate_qa(self, graph_path: str, output_path: str, dataset: str) -> dict:
        """Full pipeline: load graph_data -> select prompts -> call Gemini -> validate -> save."""
        with open(graph_path, 'r') as f:
            graph_data = json.load(f)

        video_name = graph_data.get("video_name", os.path.basename(graph_path).replace("_graph.json", ""))
        events = graph_data.get("events", [])

        if not events:
            output = {
                "video_id": video_name,
                "dataset": dataset,
                "num_qa": 0,
                "qa_pairs": [],
                "prompt_calls": [],
            }
            self._save_output(output, output_path)
            return output

        prompts = select_prompts(graph_data, video_name, dataset)

        if not prompts:
            output = {
                "video_id": video_name,
                "dataset": dataset,
                "num_qa": 0,
                "qa_pairs": [],
                "prompt_calls": [],
            }
            self._save_output(output, output_path)
            return output

        person_ids = _extract_person_ids(events)
        all_qa = []
        prompt_calls = []

        PROMPT_CONFIG = {
            "prompt_A":    ("gaze",    "mcq"),
            "prompt_B":    ("gaze",    "mcq"),
            "prompt_B_OE": ("gaze",    "open_ended"),
            "prompt_C":    ("gesture", "mcq"),
            "prompt_D":    ("gesture", "mcq"),
            "prompt_D_OE": ("gesture", "open_ended"),
            "prompt_E":    ("cross",   "mcq"),
            "prompt_E_OE": ("cross",   "open_ended"),
        }

        for prompt_name, prompt_text in prompts:
            mode, expected_format = PROMPT_CONFIG[prompt_name]

            try:
                raw_qa = self._call_gemini(prompt_text)
                validated = validate_qa(raw_qa, person_ids, mode, expected_format)

                # Tag each QA with video info
                for qa in validated:
                    qa["video_id"] = video_name
                    qa["dataset"] = dataset

                all_qa.extend(validated)
                prompt_calls.append({
                    "prompt": prompt_name,
                    "raw_count": len(raw_qa),
                    "valid_count": len(validated),
                })
                print(f"    {prompt_name}: {len(raw_qa)} raw -> {len(validated)} valid")
            except Exception as e:
                print(f"    {prompt_name}: FAILED ({e})")
                prompt_calls.append({
                    "prompt": prompt_name,
                    "raw_count": 0,
                    "valid_count": 0,
                    "error": str(e)[:200],
                })

        output = {
            "video_id": video_name,
            "dataset": dataset,
            "num_qa": len(all_qa),
            "qa_pairs": all_qa,
            "prompt_calls": prompt_calls,
        }

        self._save_output(output, output_path)
        return output

    def _save_output(self, output: dict, output_path: str) -> None:
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)


# ------------------------------------------------------------------
# Dataset processing
# ------------------------------------------------------------------

def _process_single_file(
    generator: QAGenerator,
    graph_json: str,
    output_dir: str,
    dataset: str,
    skip_existing: bool,
    file_idx: int,
    total_files: int,
) -> Tuple[str, int]:
    """Process a single graph_data file. Returns (status, num_qa)."""
    video_name = os.path.basename(graph_json).replace("_graph.json", "")
    output_path = os.path.join(output_dir, f"{video_name}_qa.json")

    if skip_existing and os.path.exists(output_path):
        return "skipped", 0

    print(f"\n  [{file_idx+1}/{total_files}] {video_name}")

    try:
        result = generator.generate_qa(graph_json, output_path, dataset)
        return "processed", result["num_qa"]
    except Exception as e:
        print(f"    FAILED: {e}")
        return "failed", 0


def process_dataset(
    base_dir: str,
    dataset_name: str,
    generator: QAGenerator,
    skip_existing: bool = False,
    max_files: int = 0,
    num_workers: int = 5,
) -> Tuple[int, int, int, int]:
    """Process all graph_data files in a dataset.

    Returns (processed, skipped, failed, total_qa).
    """
    graph_dir = os.path.join(base_dir, dataset_name, "graph_data")
    output_dir = os.path.join(base_dir, dataset_name, "qa_data")

    if not os.path.isdir(graph_dir):
        print(f"Graph data directory not found: {graph_dir}")
        return 0, 0, 0, 0

    graph_files = sorted(glob.glob(os.path.join(graph_dir, "*_graph.json")))
    total = len(graph_files)

    if total == 0:
        print(f"No graph_data files found in: {graph_dir}")
        return 0, 0, 0, 0

    if max_files > 0 and max_files < total:
        random.seed(42)
        graph_files = random.sample(graph_files, max_files)

    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name} ({len(graph_files)}/{total} files)")
    print(f"Input:   {graph_dir}")
    print(f"Output:  {output_dir}")
    print(f"Workers: {num_workers}")
    print(f"{'='*60}")

    processed, skipped, failed, total_qa = 0, 0, 0, 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(
                _process_single_file,
                generator, gf, output_dir, dataset_name, skip_existing, i, len(graph_files),
            ): gf
            for i, gf in enumerate(graph_files)
        }

        for future in concurrent.futures.as_completed(futures):
            status, nqa = future.result()
            if status == "processed":
                processed += 1
                total_qa += nqa
            elif status == "skipped":
                skipped += 1
            else:
                failed += 1

    print(f"\n{dataset_name} done: {processed} processed, {skipped} skipped, {failed} failed, {total_qa} QA pairs")
    return processed, skipped, failed, total_qa


def main():
    parser = argparse.ArgumentParser(
        description="Generate QA pairs from graph_data (Stage 9)"
    )

    parser.add_argument("--base_dir", type=str, default=DEFAULT_BASE_PATH)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--all", action="store_true")

    parser.add_argument("--graph_json", type=str, default=None,
                        help="Single file mode: path to graph_data JSON")
    parser.add_argument("--output_json", type=str, default=None)

    parser.add_argument("--model", type=str, default="models/gemini-3.1-pro-preview")
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--max_files", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=5,
                        help="Number of parallel threads (default 5, lower than graph gen due to longer prompts)")

    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("ERROR: No API key. Set GOOGLE_API_KEY or use --api_key")
        sys.exit(1)

    generator = QAGenerator(
        api_key=api_key,
        model_name=args.model,
    )

    # Single file mode
    if args.graph_json:
        if not os.path.exists(args.graph_json):
            print(f"ERROR: File not found: {args.graph_json}")
            sys.exit(1)

        output_path = args.output_json
        if not output_path:
            output_path = args.graph_json.replace("_graph.json", "_qa.json")

        print(f"Single file mode")
        print(f"Input:  {args.graph_json}")
        print(f"Output: {output_path}")

        dataset = "unknown"
        for ds in DATASETS:
            if ds in args.graph_json:
                dataset = ds
                break

        result = generator.generate_qa(args.graph_json, output_path, dataset)
        print(f"\nGenerated {result['num_qa']} QA pairs")
        return

    # Dataset mode
    if args.all:
        datasets = DATASETS
    elif args.dataset:
        datasets = [args.dataset]
    else:
        parser.print_help()
        print("\nProvide --dataset, --all, or --graph_json")
        sys.exit(1)

    total_processed, total_skipped, total_failed, grand_total_qa = 0, 0, 0, 0

    for ds in datasets:
        p, s, f, q = process_dataset(
            base_dir=args.base_dir,
            dataset_name=ds,
            generator=generator,
            skip_existing=args.skip_existing,
            max_files=args.max_files,
            num_workers=args.num_workers,
        )
        total_processed += p
        total_skipped += s
        total_failed += f
        grand_total_qa += q

    print(f"\n{'='*60}")
    print(f"All done: {total_processed} processed, {total_skipped} skipped, {total_failed} failed")
    print(f"Total QA pairs generated: {grand_total_qa}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
