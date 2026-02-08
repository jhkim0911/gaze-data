# Deictic Gesture QA Taxonomy for Video-LLM Post-Training (GRPO)

> Gesture JSON → GPT/Gemini → QA generation → Video + QA → GRPO training
>
> **핵심**: 모델은 비디오를 보고 답한다. Answer는 짧고 factual. CoT는 GRPO가 알아서 학습.

---

## 0. Data Inventory

| Field | Type | Verifiable GT | 용도 |
|---|---|---|---|
| `gesture_type` | `"pointing"` \| `"showing"` \| `"giving"` \| `"reaching"` | ✓ | 분류, 탐지 |
| `initiator_id` | int | ✓ | 누가 제스처를 했나 |
| `start_time` / `end_time` | float (seconds) | ✓ | 시간 관련 QA |
| `target_type` | `"person"` \| `"object"` | ✓ | 대상 분류 |
| `target_person_id` | int \| null | ✓ | 사람 대상일 때 누구인지 |
| `target_description` | string (free text) | ✗ (Gemini 생성) | 질문 생성 참고용만. 정답으로 안씀 |
| `confidence` | float | — | 필터링용 (≥0.85) |

> **핵심 원칙**: `target_description`은 자유 텍스트이므로 reward 계산용 GT로 쓰지 않는다. 정답은 반드시 structured fields (`gesture_type`, `initiator_id`, `target_person_id`, `target_type`, timestamps)에서만 추출.

---

## 1. Design Principles

- **Answer**: 짧고 factual. MCQ는 A/B/C/D. Short-answer는 1-5 words. No reasoning in answer.
- **Question**: 비디오를 봐야 답할 수 있고, JSON에서 GT를 뽑을 수 있어야 함.
- **Format**: MCQ 50-60%, Short-answer 30-40%, Open-ended ≤10%
- **Yes/no**: 최소화 (≤5%)
- **target_description 활용**: 질문을 자연스럽게 만들 때만 참고. 정답에는 사용 안함.

---

## 2. Taxonomy

```
DEICTIC GESTURE QA
│
├─ G1. Gesture Identification               ← "어떤 제스처를, 누가, 누구한테"
│   ├─ G1a. Gesture type classification
│   ├─ G1b. Initiator identification
│   └─ G1c. Target identification
│
├─ G2. Gesture Detection                    ← "제스처가 있었나, 몇번이나"
│   ├─ G2a. Binary detection
│   └─ G2b. Type-specific detection
│
├─ G3. Temporal Reasoning                   ← "언제, 얼마나, 무엇이 먼저"
│   ├─ G3a. Duration / comparison
│   ├─ G3b. Ordering
│   └─ G3c. Before / After
│
├─ G4. Inter-Person Gesture Dynamics        ← "주고받기, 체인, 상호작용"
│   ├─ G4a. Object transfer chain
│   ├─ G4b. Reciprocal gestures
│   └─ G4c. Pointing patterns (mutual / one-way)
│
└─ G5. Gesture Frequency & Activity         ← "누가 제일 많이, 어떤 타입이 주로"
    ├─ G5a. Most active gesturer
    ├─ G5b. Most targeted person
    └─ G5c. Dominant gesture type
```

---

## 3. Category Details & Examples

### G1. Gesture Identification

| Sub | Question Pattern | Format | Answer Source | Difficulty |
|---|---|---|---|---|
| G1a | "What type of hand gesture does Person X perform at T?" | MCQ | `gesture_type` | Easy |
| G1b | "Who [points at / gives something to] Person Y at T?" | MCQ | `initiator_id` | Easy |
| G1c | "At T, Person X points at whom?" | MCQ | `target_person_id` | Easy |
| G1c | "Is Person X reaching for a person or an object at T?" | MCQ | `target_type` | Easy |

**G1 예시 (from 152_1_002):**

```
Q: "At 21.0s, what type of hand gesture does Person 0 perform?"
Options: A) Pointing  B) Showing an object  C) Giving something  D) Reaching for something
Answer: A
→ source: gesture[0].gesture_type = "pointing"

Q: "At 97.0s, who is Person 0 pointing at?"
Options: A) Person 1  B) Person 3  C) Person 4  D) Himself
Answer: B
→ source: gesture[9].target_person_id = 3
```

### G2. Gesture Detection

| Sub | Question Pattern | Format | Answer Source | Difficulty |
|---|---|---|---|---|
| G2a | "Does anyone perform a hand gesture between T1 and T2?" | MCQ | time range check | Easy |
| G2b | "Does Person X point at anyone in this clip?" | MCQ | filter by initiator + type | Medium |
| G2b | "Does anyone give an object to Person Y?" | MCQ | filter by target + type | Medium |

**G2 예시 (from 152_1_001):**

```
Q: "Does anyone give an object to Person 3 in this clip?"
Options: A) Yes, one person does  B) Yes, multiple people do  C) No  D) Person 3 gives to others instead
Answer: B
→ source: gesture[6] (P4→P3) + gesture[7] (P2→P3), both "giving"

Q: "Does Person 1 point at anyone in this clip?"
Options: A) Yes  B) No
Answer: B
→ source: no pointing gesture with initiator_id=1 in 152_1_001
```

### G3. Temporal Reasoning

| Sub | Question Pattern | Format | Answer Source | Difficulty |
|---|---|---|---|---|
| G3a | "How long does Person X's pointing gesture at T last?" | MCQ | end_time - start_time | Medium |
| G3a | "Which lasts longer: P0's pointing at 21s or P0's pointing at 88s?" | MCQ | duration comparison | Medium |
| G3b | "Which happens first: P1 giving to P4, or P4 showing to P0?" | MCQ | start_time comparison | Medium |
| G3c | "What gesture occurs right after P3 points at P4 at 65s?" | MCQ | next event by start_time | Hard |

**G3 예시 (from 152_1_001):**

```
Q: "Which happens first: Person 4 showing something to Person 0, or Person 2 giving something to Person 0?"
Options: A) Person 4 showing  B) Person 2 giving  C) They happen at the same time  D) Neither happens
Answer: A
→ source: showing start_time=16.5 < giving start_time=20.0

Q: "What gesture occurs right after Person 3 points at Person 4?"
Options: A) Person 4 gives something to Person 3  B) Person 0 reaches for an object  C) Person 1 shows something  D) Nothing happens
Answer: A
→ source: pointing ends 68.0, next gesture is giving at 69.0 (P4→P3)
```

### G4. Inter-Person Gesture Dynamics

| Sub | Question Pattern | Format | Answer Source | Difficulty |
|---|---|---|---|---|
| G4a | "P1 gives something to P4 at 4s. Does P4 later gesture toward P1?" | MCQ | cross-event lookup | Hard |
| G4a | "An object goes from P1→P4, then P4 shows it. Who does P4 show it to?" | MCQ | chain tracking | Hard |
| G4b | "After P2 gives to P0, does P0 ever gesture back to P2?" | MCQ | reciprocal check | Hard |
| G4c | "P0 and P3 both point in this clip. Do they point at each other?" | MCQ | cross-reference targets | Hard |

**G4 예시 (from 152_1_001):**

```
Q: "Person 1 gives something to Person 4 at 4.0s. Later, Person 4 shows an object to someone. Who does Person 4 show it to?"
Options: A) Person 1  B) Person 0  C) Person 3  D) Person 2
Answer: B
→ source: giving P1→P4 at 4.0s, then showing P4→P0 at 16.5s (object transfer chain)

Q: "Person 3 points at Person 4 at 65s. Does Person 4 gesture back toward Person 3?"
Options: A) Yes, Person 4 gives something to Person 3  B) Yes, Person 4 points back  C) No, Person 4 ignores it  D) Person 4 gestures toward someone else
Answer: A
→ source: P3 pointing at P4 (65-68s), then P4 giving to P3 (69-70.6s)
```

**G4 예시 (from 152_1_002 — mutual pointing):**

```
Q: "Between 88s and 93s, Person 0 and Person 3 both point. Do they point at each other?"
Options: A) Yes, they point at each other  B) Only P0 points at P3  C) Only P3 points at P0  D) They point at different people
Answer: A
→ source: P0→P3 at 97-98s, P3→P0 at 90.5-92.5s. Actually: P0→P1 at 88-91, P3→P0 at 90.5-92.5. Hmm, need to be more careful.
Actually: P0 points at P1 (88-91), P3 points at P0 (90.5-92.5), then P0 points at self (91-92.5), then P0 points at P3 (97-98), P3 points at P0 (102-103.5).
So P0→P3 and P3→P0 do happen, just not simultaneously. Better question:

Q: "In this clip, Person 0 points at Person 3 at one point, and Person 3 also points at Person 0. Is their pointing mutual?"
Options: A) Yes, they each point at the other at some point  B) Only Person 0 points at Person 3  C) Only Person 3 points at Person 0  D) Neither points at the other
Answer: A
→ source: P0→P3 (97-98s) + P3→P0 (90.5-92.5s, 102-103.5s)
```

### G5. Gesture Frequency & Activity

| Sub | Question Pattern | Format | Answer Source | Difficulty |
|---|---|---|---|---|
| G5a | "Who performs the most gestures in this clip?" | MCQ / short_answer | count by initiator_id | Medium |
| G5b | "Which person is pointed at most often?" | MCQ / short_answer | count by target_person_id | Medium |
| G5c | "What is the most common gesture type in this clip?" | MCQ | count by gesture_type | Medium |

**G5 예시 (from 152_1_002):**

```
Q: "Who performs the most pointing gestures in this clip?"
Options: A) Person 0  B) Person 1  C) Person 3  D) Person 4
Answer: A
→ source: P0 has 7 pointing gestures, P3 has 2

Q: "What is the most common gesture type in this clip?"
Options: A) Pointing  B) Showing  C) Giving  D) Reaching
Answer: A
→ source: pointing=9, showing=1, reaching=2
```

---

## 4. Gesture Type → Category Mapping

| | G1 (ID) | G2 (Detect) | G3 (Temporal) | G4 (Dynamics) | G5 (Frequency) |
|---|:---:|:---:|:---:|:---:|:---:|
| pointing | ✓ type, initiator, target | ✓ detect | ✓ duration, order | ✓ mutual pointing | ✓ count |
| showing | ✓ type, initiator, target | ✓ detect | ✓ duration, order | ✓ transfer chain | ✓ count |
| giving | ✓ type, initiator, target | ✓ detect | ✓ duration, order | ✓ transfer chain, reciprocal | ✓ count |
| reaching | ✓ type, initiator | ✓ detect | ✓ duration, order | — (보통 object 대상) | ✓ count |

> **참고**: `reaching`은 대부분 `target_type="object"`이므로 G4 (inter-person dynamics)에서 제외. G4는 person-to-person 상호작용에만 적용.

---

## 5. Difficulty & Distribution

| Level | Categories | Target % |
|---|---|---|
| Easy | G1a-c, G2a | 25–30% |
| Medium | G2b, G3a, G3b, G5a-c | 35–40% |
| Hard | G3c, G4a-c | 30–35% |

---

## 6. QA JSON Format

```json
{"video_id": "str", "dataset": "str", "category": "G1a",
 "source_gesture_ids": [0], "time_range": [21.0, 22.5],
 "format": "mcq",
 "question": "At 21.0s, what type of hand gesture does Person 0 perform?",
 "options": ["A) Pointing", "B) Showing an object", "C) Giving something", "D) Reaching for something"],
 "answer": "A", "answer_text": "Pointing"}
```

```json
{"video_id": "str", "dataset": "str", "category": "G5a",
 "format": "short_answer",
 "question": "Who performs the most gestures in this clip?",
 "answer": "Person 0", "answer_text": "Person 0"}
```

---

## 7. API Prompts

### 7.1 Master Prompt (시스템 프롬프트)

```
You are a QA generator for a deictic gesture reasoning dataset.
You will receive gesture annotations detected in a video clip. Each gesture has a type (pointing, showing, giving, reaching), an initiator, timestamps, and target information.

Your job: generate question-answer pairs that a Video-LLM must answer by WATCHING the video.

STRICT RULES:
1. Answers must be SHORT and FACTUAL.
   - MCQ: output the correct option letter only (A/B/C/D)
   - Short-answer: 1-5 words only
2. NEVER put reasoning, speculation, or interpretation in the answer.
   Banned words in answers: "suggesting", "indicating", "likely", "because", "implies", "seems", "probably"
3. Questions must require WATCHING THE VIDEO to answer — not answerable from question text alone.
4. Use natural language in questions. Say "points at" not "performs a pointing gesture event". Say "hands something to" not "initiates a giving gesture toward".
5. MCQ must have exactly 4 options. Distractors must be plausible (other person IDs in the video, other gesture types, swapped roles, etc.)
6. All person references must use "Person {id}" format matching the gesture data.
7. Include specific timestamps from the gesture data in questions.
8. The field "target_description" is for your reference to understand context. Do NOT use its exact text as an answer. Answers must come from structured fields only (gesture_type, initiator_id, target_person_id, target_type, timestamps).

OUTPUT: Return a JSON array. Each element:
{
  "category": "G1a",
  "format": "mcq" | "short_answer",
  "question": "...",
  "options": ["A) ...", "B) ...", "C) ...", "D) ..."],  // only for mcq
  "answer": "A",           // letter for mcq, text for short_answer
  "answer_text": "...",    // full text of correct answer
  "source_gesture_indices": [0],
  "time_range": [21.0, 22.5]
}
```

---

### 7.2 Few-Gesture Prompt (제스처 1-3개, 간단한 비디오)

카테고리 G1, G2만 생성. 가장 기본적인 QA.

```
TASK: Generate 2-4 QA pairs from the gesture annotation(s) below.

CATEGORIES TO COVER:
- G1a (Type classification): "What type of gesture does Person X perform at T?" → MCQ
- G1b (Initiator ID): "Who [points at / hands something to] Person Y at T?" → MCQ
- G1c (Target ID): "At T, who is Person X pointing at?" → MCQ
- G2a (Detection): "Does anyone [point / give / show / reach] between T1 and T2?" → MCQ
- G2b (Type-specific): "Does Person X point at anyone in this clip?" → MCQ

FORMAT MIX: at least 1 MCQ.

VIDEO METADATA:
- video_id: {video_id}
- dataset: {dataset}
- total_persons_in_video: {person_ids}
- video_duration: {duration}s

GESTURE ANNOTATIONS:
{gestures_json}

Generate the QA pairs now. Return ONLY the JSON array, no explanation.
```

**활용 예시 (reaching only, 1 gesture):**

```
VIDEO METADATA:
- video_id: social_gesture_045
- dataset: SocialGesture
- total_persons_in_video: [0, 1, 2]
- video_duration: 30.0s

GESTURE ANNOTATIONS:
[
  {"gesture_type": "reaching", "initiator_id": 0,
   "start_time": 12.0, "end_time": 13.5,
   "target_type": "object", "target_person_id": null,
   "target_description": "cup on the table", "confidence": 0.95}
]
```

**기대 output:**
```json
[
  {"category": "G1a", "format": "mcq",
   "question": "At 12.0s, what type of hand gesture does Person 0 perform?",
   "options": ["A) Pointing at someone", "B) Showing an object to someone", "C) Handing something to someone", "D) Reaching for something"],
   "answer": "D", "answer_text": "Reaching for something",
   "source_gesture_indices": [0], "time_range": [12.0, 13.5]},

  {"category": "G1c", "format": "mcq",
   "question": "At 12.0s, is Person 0 reaching toward a person or an object?",
   "options": ["A) Toward a person", "B) Toward an object on the table", "C) Toward nothing specific", "D) Toward the camera"],
   "answer": "B", "answer_text": "Toward an object on the table",
   "source_gesture_indices": [0], "time_range": [12.0, 13.5]}
]
```

---

### 7.3 Multi-Gesture Prompt (제스처 4개 이상)

G1-G5 전체 커버. 난이도 mix.

```
TASK: Generate 6-10 QA pairs from the gesture annotations below.
Cover a MIX of difficulty levels and categories.

REQUIRED CATEGORY COVERAGE:
- At least 1 from G1 (Gesture ID): what type, who initiates, who is the target
- At least 1 from G2 (Detection): does a specific gesture type occur
- At least 1 from G3 (Temporal): duration, ordering, before/after
- At least 1 from G4 (Inter-Person Dynamics): transfer chains, reciprocal gestures, mutual pointing
  (only if multiple person-to-person gestures exist between overlapping participants)
- At least 1 from G5 (Frequency): who gestures most, most common type
  (only if 4+ gestures exist)

DIFFICULTY MIX:
- 2-3 Easy (G1, G2a): single gesture, direct identification
- 2-3 Medium (G2b, G3a, G3b, G5): counting, duration, ordering
- 2-3 Hard (G3c, G4): cross-gesture chains, reciprocal patterns

FORMAT MIX: ~60% MCQ, ~40% short_answer

VIDEO METADATA:
- video_id: {video_id}
- dataset: {dataset}
- total_persons_in_video: {person_ids}
- video_duration: {duration}s

GESTURE ANNOTATIONS:
{gestures_json}

Generate the QA pairs now. Return ONLY the JSON array, no explanation.
```

**활용 예시 (152_1_001, 11 gestures):**

```
VIDEO METADATA:
- video_id: 152_1_001
- dataset: Werewolf
- total_persons_in_video: [0, 1, 2, 3, 4]
- video_duration: 120.0s

GESTURE ANNOTATIONS:
[
  {"gesture_type": "giving", "initiator_id": 1,
   "start_time": 4.0, "end_time": 5.1,
   "target_type": "person", "target_person_id": 4,
   "target_description": "P1 extends their arm to give a small rectangular object to P4", "confidence": 0.95},

  {"gesture_type": "showing", "initiator_id": 4,
   "start_time": 16.5, "end_time": 18.6,
   "target_type": "person", "target_person_id": 0,
   "target_description": "P4 holds up an object and orients it towards P0", "confidence": 0.9},

  {"gesture_type": "giving", "initiator_id": 2,
   "start_time": 20.0, "end_time": 21.6,
   "target_type": "person", "target_person_id": 0,
   "target_description": "P2 extends their arm to give a card to P0", "confidence": 0.95},

  {"gesture_type": "reaching", "initiator_id": 0,
   "start_time": 45.0, "end_time": 46.1,
   "target_type": "object", "target_person_id": null,
   "target_description": "yellow can", "confidence": 0.95},

  {"gesture_type": "reaching", "initiator_id": 1,
   "start_time": 55.5, "end_time": 56.6,
   "target_type": "object", "target_person_id": null,
   "target_description": "object on the table", "confidence": 0.9},

  {"gesture_type": "pointing", "initiator_id": 3,
   "start_time": 65.0, "end_time": 68.0,
   "target_type": "person", "target_person_id": 4,
   "target_description": "P3 extends their open hand towards P4", "confidence": 0.85},

  {"gesture_type": "giving", "initiator_id": 4,
   "start_time": 69.0, "end_time": 70.6,
   "target_type": "person", "target_person_id": 3,
   "target_description": "P4 extends hand holding a card to give to P3", "confidence": 0.95},

  {"gesture_type": "giving", "initiator_id": 2,
   "start_time": 73.5, "end_time": 75.0,
   "target_type": "person", "target_person_id": 3,
   "target_description": "P2 extends arm to give a card to P3", "confidence": 0.95},

  {"gesture_type": "giving", "initiator_id": 2,
   "start_time": 80.0, "end_time": 81.5,
   "target_type": "person", "target_person_id": 1,
   "target_description": "P2 extends arm to give a card to P1", "confidence": 0.95},

  {"gesture_type": "showing", "initiator_id": 1,
   "start_time": 103.4, "end_time": 105.0,
   "target_type": "person", "target_person_id": 0,
   "target_description": "P1 holds out a card towards P0", "confidence": 0.9},

  {"gesture_type": "reaching", "initiator_id": 0,
   "start_time": 115.5, "end_time": 116.5,
   "target_type": "object", "target_person_id": null,
   "target_description": "object on the table", "confidence": 0.9}
]
```

**기대 output:**
```json
[
  {"category": "G1a", "format": "mcq",
   "question": "At 4.0s, what type of hand gesture does Person 1 perform toward Person 4?",
   "options": ["A) Pointing", "B) Showing something", "C) Handing something over", "D) Reaching for an object"],
   "answer": "C", "answer_text": "Handing something over",
   "source_gesture_indices": [0], "time_range": [4.0, 5.1]},

  {"category": "G1c", "format": "mcq",
   "question": "At 65.0s, who is Person 3 pointing at?",
   "options": ["A) Person 0", "B) Person 1", "C) Person 2", "D) Person 4"],
   "answer": "D", "answer_text": "Person 4",
   "source_gesture_indices": [5], "time_range": [65.0, 68.0]},

  {"category": "G2b", "format": "mcq",
   "question": "Does anyone give an object to Person 3 in this clip?",
   "options": ["A) Yes, one person does", "B) Yes, two different people do", "C) No, nobody gives anything to Person 3", "D) Person 3 gives to others instead"],
   "answer": "B", "answer_text": "Yes, two different people do",
   "source_gesture_indices": [6, 7], "time_range": [69.0, 75.0]},

  {"category": "G3b", "format": "mcq",
   "question": "Which happens first: Person 4 showing something to Person 0, or Person 2 giving something to Person 0?",
   "options": ["A) Person 4 showing", "B) Person 2 giving", "C) They happen at the same time", "D) Neither happens"],
   "answer": "A", "answer_text": "Person 4 showing",
   "source_gesture_indices": [1, 2], "time_range": [16.5, 21.6]},

  {"category": "G3c", "format": "mcq",
   "question": "What gesture happens right after Person 3 points at Person 4 at 65s?",
   "options": ["A) Person 4 hands something to Person 3", "B) Person 0 reaches for an object", "C) Person 1 shows something to Person 0", "D) Person 2 gives something to Person 1"],
   "answer": "A", "answer_text": "Person 4 hands something to Person 3",
   "source_gesture_indices": [5, 6], "time_range": [65.0, 70.6]},

  {"category": "G4a", "format": "mcq",
   "question": "Person 1 gives an object to Person 4 at 4s. Later, Person 4 holds up an object toward someone. Who does Person 4 show it to?",
   "options": ["A) Person 1", "B) Person 0", "C) Person 3", "D) Person 2"],
   "answer": "B", "answer_text": "Person 0",
   "source_gesture_indices": [0, 1], "time_range": [4.0, 18.6]},

  {"category": "G4b", "format": "mcq",
   "question": "Person 3 points at Person 4 around 65s. Does Person 4 gesture back toward Person 3 soon after?",
   "options": ["A) Yes, Person 4 gives something to Person 3", "B) Yes, Person 4 points back at Person 3", "C) No, Person 4 does not gesture toward Person 3", "D) Person 4 gestures toward Person 0 instead"],
   "answer": "A", "answer_text": "Yes, Person 4 gives something to Person 3",
   "source_gesture_indices": [5, 6], "time_range": [65.0, 70.6]},

  {"category": "G5a", "format": "short_answer",
   "question": "Who gives objects to others most frequently in this clip?",
   "answer": "Person 2", "answer_text": "Person 2",
   "source_gesture_indices": [2, 7, 8], "time_range": [20.0, 81.5]},

  {"category": "G5c", "format": "mcq",
   "question": "What is the most common type of hand gesture in this clip?",
   "options": ["A) Pointing", "B) Showing", "C) Giving/handing over", "D) Reaching"],
   "answer": "C", "answer_text": "Giving/handing over",
   "source_gesture_indices": [0, 2, 6, 7, 8], "time_range": [4.0, 81.5]}
]
```

---

### 7.4 Category-Specific Addon Prompts

multi-gesture prompt 뒤에 추가로 붙여서 특정 카테고리를 더 생성할 때 사용.

#### G4 전용 (giving/showing이 3개 이상, person-to-person)

```
ADDITIONAL TASK: Generate 2 more QA pairs focused on G4 (Inter-Person Gesture Dynamics).
The video has multiple giving/showing gestures between people.

Focus on:
- G4a: Object transfer chains — did an object pass through multiple people?
- G4b: Reciprocal gestures — after Person X gives to Person Y, does Y ever gesture back to X?

Same rules as before. Return JSON array.
```

#### G5 전용 (제스처 6개 이상)

```
ADDITIONAL TASK: Generate 2 more QA pairs focused on G5 (Gesture Frequency & Activity).
The video has many gesture events.

Focus on:
- G5a: Who is the most active gesturer overall?
- G5b: Who is pointed at or given objects most often?

Same rules as before. Return JSON array.
```

#### G3c 전용 (시간적으로 연속된 제스처 3개 이상)

```
ADDITIONAL TASK: Generate 2 more QA pairs focused on G3c (Before/After reasoning).
The video has multiple gestures in temporal sequence.

Focus on:
- What gesture immediately follows another?
- What changes after a specific gesture ends?

Use ONLY adjacent gestures in the timeline. Same rules. Return JSON array.
```

---

## 8. Generation Script 로직 (pseudo-code)

```python
def generate_qa_for_video(video_id, gestures, person_ids, duration, dataset):
    # 1. Filter by confidence
    gestures = [g for g in gestures if g["confidence"] >= 0.85]

    # 2. Pick prompt based on gesture count
    if len(gestures) <= 3:
        prompt = FEW_GESTURE_PROMPT
        n_qa = "2-4"
        categories = ["G1", "G2"]
    else:
        prompt = MULTI_GESTURE_PROMPT
        n_qa = "6-10"
        categories = ["G1", "G2", "G3", "G4", "G5"]

    # 3. Check which categories are valid
    person_to_person = [g for g in gestures
                        if g["target_type"] == "person" and g["target_person_id"] is not None]
    if len(person_to_person) < 2:
        categories.discard("G4")  # need at least 2 person-to-person gestures
    if len(gestures) < 4:
        categories.discard("G5")  # frequency questions need enough data

    # 4. Fill prompt template
    # Strip target_description from input to avoid leaking free text
    # Actually keep it — prompt says to use for context only, not as answer
    filled = SYSTEM_PROMPT + prompt.format(
        video_id=video_id,
        dataset=dataset,
        person_ids=person_ids,
        duration=duration,
        gestures_json=json.dumps(gestures, indent=2)
    )

    # 5. Call API
    response = call_llm_api(filled)
    qa_pairs = json.loads(response)

    # 6. Optional addon prompts
    giving_showing = [g for g in gestures
                      if g["gesture_type"] in ("giving", "showing")
                      and g["target_type"] == "person"]
    if len(giving_showing) >= 3:
        addon = call_llm_api(filled + G4_ADDON_PROMPT)
        qa_pairs.extend(json.loads(addon))

    if len(gestures) >= 6:
        addon = call_llm_api(filled + G5_ADDON_PROMPT)
        qa_pairs.extend(json.loads(addon))

    # 7. Validate & filter
    qa_pairs = validate_and_filter(qa_pairs, gestures, person_ids)

    # 8. Attach metadata
    for qa in qa_pairs:
        qa["video_id"] = video_id
        qa["dataset"] = dataset

    return qa_pairs


def validate_and_filter(qa_pairs, gestures, person_ids):
    valid = []
    for qa in qa_pairs:
        # Answer length check
        if qa["format"] == "mcq" and len(qa["answer"]) != 1:
            continue
        if qa["format"] == "short_answer" and len(qa["answer"].split()) > 10:
            continue

        # Hallucination check
        banned = ["suggesting", "indicating", "likely", "because",
                  "implies", "emotion", "feeling", "probably"]
        if any(w in qa["answer"].lower() for w in banned):
            continue

        # Person ID check
        mentioned_ids = re.findall(r'Person (\d+)', qa["question"] + qa["answer_text"])
        if any(int(pid) not in person_ids for pid in mentioned_ids):
            continue

        # Gesture index bounds check
        if "source_gesture_indices" in qa:
            if any(idx >= len(gestures) for idx in qa["source_gesture_indices"]):
                continue

        valid.append(qa)
    return valid
```

---

## 9. MCQ Distractor Strategy

| Question type | Correct source | Distractor strategy |
|---|---|---|
| "What gesture does PX do?" | `gesture_type` | Other 3 gesture types |
| "Who points at PY?" | `initiator_id` | Other person IDs in video |
| "Who is PX pointing at?" | `target_person_id` | Other person IDs, "no one" |
| "Person or object?" | `target_type` | Swap person↔object |
| "Which happens first?" | `start_time` comparison | Swap order, "same time", "neither" |
| "What happens after X?" | Next gesture by time | Other gestures, "nothing" |
| "Who gives most often?" | count by initiator | Other frequent initiators |
| "Most common type?" | count by gesture_type | Other types sorted by count |

---

## 10. Quality Filters

| Filter | Rule |
|---|---|
| Answer length | MCQ: 1 char. Short: ≤10 words |
| Hallucination | No emotions, dialogue, objects not in data |
| Data match | Person IDs, timestamps traceable to input JSON |
| Distractor quality | MCQ options distinct and plausible |
| Video-dependence | Q not answerable from question text alone |
| target_description leak | Answer must NOT copy target_description verbatim |
| Dedup | >60% unigram overlap with another Q from same video → drop |

---

## 11. Gaze + Gesture Cross-Modal Notes

이 taxonomy는 gesture 단독으로 사용 가능하지만, 같은 비디오에 gaze 데이터도 있다면 cross-modal QA도 가능:

```
가능한 cross-modal QA 예시:
- "Person 3 points at Person 4 at 65s. Are other people looking at Person 4 at that moment?"
  → gesture(pointing target) + gaze(gaze target) cross-reference
- "After Person 1 gives an object to Person 4, does Person 4 look at it?"
  → gesture(giving) + gaze(gaze shift) temporal chaining
```

> 이건 향후 gaze+gesture 합칠 때 별도 taxonomy로 확장 가능. 지금은 gesture 단독.
