"""Analyze this entire video ({video_duration:.1f} seconds) and detect ALL deictic gestures.

## About This Video
- The video contains only Person IDs (e.g., P0, P1, P2) displayed above each person’s head. No bounding boxes or gaze direction information should be assumed or used.
- Scan the entire video from start to end and annotate ALL deictic gestures defined below with their exact timestamps.
- Deictic gestures are identified solely from visible hand, finger, and arm movements, based on communicative intent, and must NOT rely on gaze direction, head orientation, or any inferred attention cues.

## Timestamp - Visual Reference (CRITICAL)
- Look at the text overlay at the top-left of the video: "Frame: X | t=Y.YYs".
- DO NOT calculate the time yourself.
- ALWAYS use the 't' value (seconds) shown on the screen for your timestamps.

## Deictic Gesture Types

1. **pointing** - Hand/arm gesture intentionally indicating a specific target 
  - YES: Clear hand or arm orientation that unambiguously indicates a specific person or object in the scene, with clear communicative intent to direct attention 
  - NOT the case: Talking with hands (beat/emphasis gestures), casual or rhythmic hand movements, arm resting, or reaching/manipulating gestures where the primary intent is interaction rather than indicating 
  - Start: When the hand/arm begins a purposeful movement to indicate a specific target 
  - End: When the indicating posture relaxes, retracts, or clearly stops referring to that target

2. **showing** - Presenting an object for others to visually inspect
  - YES: The subject intentionally presents an object by orienting, tilting, sliding, or positioning it so that its visual properties are clearly exposed to others
  - NOT the case: Simply holding or carrying an object without presenting it, object resting on table/lap without intentional presentation, or manipulating an object for use rather than display
  - Start: When the object begins to be intentionally oriented or positioned for visual inspection
  - End: When the object returns to a neutral/use position or is no longer presented

3. **giving** - Intentionally offering an object to transfer possession
  - YES: The subject intentionally offers an object to another person in a way that clearly invites transfer of possession, regardless of whether the recipient actually takes it
  - NOT the case: Simply holding or displaying an object, presenting an object for viewing (showing), or moving an object without intent to hand it over
  - Start: When the object begins to be purposefully moved or offered toward another person for transfer
  - End: When the offering action clearly ends (e.g., the object is released, pulled back, or the offer is withdrawn)

4. **reaching** - Extending the hand toward an object with intent to obtain it
  - YES: The subject clearly expresses intent to acquire the object through a directed hand/arm extension toward an object not in their possession, often accompanied by pre-grasp configuration (open/curved fingers) or body movement (e.g., leaning forward)
  - NOT the case: Pointing or indicating gestures, casual arm extension without acquisition intent, expressive or beat gestures, or any motion where the hand is not plausibly attempting to obtain the object
  - Start: When the hand/arm begins a purposeful reach toward an object with acquisition intent
  - End: When the reach terminates (contact attempt, clear failure to obtain, or withdrawal of the hand)

## Rules
1. Be **strict and conservative**: Annotate a gesture ONLY when the gesture type and intent are **clearly and unambiguously identifiable** from hand/arm motion alone.
2. If a gesture’s **intent or type cannot be clearly determined** from the hand/arm motion alone, do NOT annotate it.
3. Each gesture must have a clear start_time and end_time (in seconds).
4. Person IDs must exactly match the visible Person IDs (P0, P1, P2, ...).
5. If no **high-confidence** deictic gestures are visible, return an EMPTY array.


## IMPORTANT NOTE
- Focus ONLY on hand, finger, and arm movements that clearly express communicative intent.
- Do NOT infer intent from gaze, head orientation, speech, or context outside the visible gesture.
- If communicative intent cannot be confidently inferred from the gesture itself, do NOT annotate.
- There will be multiple gesture events in the video even simultaneously.
- If a previous prediction or critique report is provided, use it to reduce false positives rather than increase coverage.

## Response Format (JSON only)

```json
{{
    "gestures": [
        {{
            "gesture_type": "pointing" | "showing" | "giving" | "reaching",
            "initiator_id": <number>,
            "start_time": <t value from screen>,
            "end_time": <t value from screen>,
            "target_type": "person" | "object" | "location",
            "target_person_id": <ID or null>,
            "target_description": "<brief description>",
            "confidence": <0.0 to 1.0>
        }},
        ...
    ]
}}
```

Return EMPTY array if no gestures detected. For example,
```json
{{
    "gestures": []
}}
```
"""