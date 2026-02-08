# Cross-Modal (Gaze + Gesture) QA Taxonomy for Video-LLM Post-Training (GRPO)

> Gaze events + Gesture events (timestamp-aligned) → GPT/Gemini → Cross-modal QA → Video + QA → GRPO training
>
> **핵심**: 같은 비디오에서 gaze와 gesture 데이터를 동시에 활용. 단독 모달리티 QA(T1-T5, G1-G5)와 **별도로** cross-modal reasoning을 요구하는 QA만 생성.

---

## 0. Data Inventory (Combined)

### Gaze Event Fields

| Field | Values |
|---|---|
| `event_type` | `sudden_gaze_shift`, `joint_attention`, `gaze_following`, `attention_capture`, `mutual_gaze` |
| `start_time` / `end_time` | float (seconds) |
| `persons_involved` | [int] |
| Type-specific | `leader_id`/`follower_id` (gaze_following), `duration`/`num_frames` (mutual_gaze), `mean_convergence`/`num_persons` (joint_attention), `mean_velocity`/`person_velocities` (attention_capture), `max_velocity` (sudden_gaze_shift) |

### Gesture Event Fields

| Field | Values |
|---|---|
| `gesture_type` | `pointing`, `showing`, `giving`, `reaching` |
| `initiator_id` | int |
| `start_time` / `end_time` | float (seconds) |
| `target_type` | `"person"` \| `"object"` |
| `target_person_id` | int \| null |
| `target_description` | string (참고용만, GT 아님) |

### Combined Input Format (가정)

```json
{
  "video_id": "152_1_001",
  "dataset": "Werewolf",
  "persons_in_video": [0, 1, 2, 3, 4],
  "video_duration": 120.0,
  "gaze_events": [ ... ],
  "gesture_events": [ ... ]
}
```

> **원칙**: 두 모달리티의 structured fields만 GT로 사용. `target_description`은 질문 생성 참고용만.

---

## 1. Design Principles

- **Cross-modal ONLY**: 이 taxonomy는 gaze+gesture를 동시에 참조해야만 답할 수 있는 QA만 생성. 단일 모달리티로 답 가능한 건 T1-T5 / G1-G5에서 커버.
- **Answer**: 짧고 factual. MCQ는 A/B/C/D. Short-answer는 1-5 words.
- **Verifiability**: 정답은 양쪽 JSON의 structured fields에서 추출 가능해야 함.
- **Format**: MCQ 60-70%, Short-answer 30-40% (cross-modal은 reasoning 복잡도 높으므로 MCQ 비중 높임)
- **시간 기반 매칭**: 두 모달리티 간 관계는 timestamp overlap 또는 temporal proximity (≤3s gap)로 정의.

---

## 2. Taxonomy

```
CROSS-MODAL QA (Gaze × Gesture)
│
├─ C1. Temporal Co-occurrence               ← "동시에 일어나는 시선-제스처"
│   ├─ C1a. Overlap detection
│   └─ C1b. Co-occurring event type identification
│
├─ C2. Gaze Response to Gesture             ← "제스처 후 시선이 어떻게 반응하나"
│   ├─ C2a. Attention shift after gesture
│   ├─ C2b. Group attention triggered by gesture
│   └─ C2c. Eye contact during object transfer
│
├─ C3. Cross-Modal Temporal Sequence        ← "모달리티 간 시간 순서"
│   ├─ C3a. Which-first (gaze event vs gesture)
│   └─ C3b. What follows across modalities
│
└─ C4. Cross-Modal Person Dynamics          ← "사람 수준 교차 모달 패턴"
    ├─ C4a. Target-participant match
    ├─ C4b. Reciprocal cross-modal response
    └─ C4c. Cross-modal social centrality
```

---

## 3. Category Details

### C1. Temporal Co-occurrence

두 모달리티 이벤트가 시간적으로 겹치는지, 겹친다면 어떤 타입인지.

| Sub | Question Pattern | Format | Answer Source | Difficulty |
|---|---|---|---|---|
| C1a | "During P0's pointing at 21s, is there a gaze event also happening?" | MCQ | timestamp overlap check | Medium |
| C1a | "Which gesture overlaps with the mutual gaze between P1 and P4?" | MCQ | timestamp overlap check | Medium |
| C1b | "What type of gaze interaction co-occurs with P4 showing something at 16.5s?" | MCQ | overlap + gaze event_type | Medium |
| C1b | "What type of gesture is happening during the joint attention at 64s?" | MCQ | overlap + gesture_type | Medium |

### C2. Gaze Response to Gesture

제스처 발생 후 시선 패턴이 어떻게 변하는지. 가장 socially meaningful한 카테고리.

| Sub | Question Pattern | Format | Answer Source | Difficulty |
|---|---|---|---|---|
| C2a | "After P3 points at P4 at 65s, does anyone shift their gaze?" | MCQ | gesture end_time → nearby gaze events | Hard |
| C2a | "After P2 gives something to P0 at 20s, who else shifts attention?" | MCQ | gesture time → attention_capture participants | Hard |
| C2b | "After P4 shows an object at 16.5s, do multiple people look at the same area?" | MCQ | gesture → nearby joint_attention | Hard |
| C2b | "Is there a group attention event within 3 seconds of P3's pointing gesture?" | MCQ | temporal proximity check | Hard |
| C2c | "During the exchange from P1 to P4 at 4.0s, are they making eye contact?" | MCQ | giving time ∩ mutual_gaze with same persons | Hard |
| C2c | "When P1 shows a card to P0 at 103s, are P1 and P0 looking at each other?" | MCQ | showing time ∩ mutual_gaze with same persons | Hard |

### C3. Cross-Modal Temporal Sequence

서로 다른 모달리티 이벤트 간의 시간 순서.

| Sub | Question Pattern | Format | Answer Source | Difficulty |
|---|---|---|---|---|
| C3a | "Which happens first: P0's sudden gaze shift, or P0 reaching for an object at 45s?" | MCQ | start_time comparison across modalities | Medium |
| C3a | "Does eye contact between P4 and P3 start before or after P4's giving gesture?" | MCQ | mutual_gaze start vs giving start | Medium |
| C3b | "What gesture happens right after the joint attention event ends at 19s?" | MCQ | gaze end_time → next gesture by start_time | Hard |
| C3b | "What gaze event occurs right after P3's pointing gesture ends at 68s?" | MCQ | gesture end_time → next gaze event | Hard |

### C4. Cross-Modal Person Dynamics

같은 사람이 두 모달리티에서 어떻게 나타나는지.

| Sub | Question Pattern | Format | Answer Source | Difficulty |
|---|---|---|---|---|
| C4a | "P3 points at P4 at 65s. Is P4 involved in any gaze event around that time?" | MCQ | gesture target → gaze persons_involved check | Hard |
| C4a | "There is joint attention among P0, P1, P3 at 64s. Does anyone gesture toward the group at that time?" | MCQ | gaze persons → gesture target check | Hard |
| C4b | "P1 gives an object to P4 at 4s. Does P4 look back at P1?" | MCQ | gesture initiator/target → mutual_gaze check | Hard |
| C4b | "After P0 reaches for an object at 45s, does anyone look at P0?" | MCQ | gesture initiator → gaze persons_involved | Hard |
| C4c | "Who is both most frequently gestured-at AND most involved in gaze events?" | short_answer | count across both modalities | Hard |
| C4c | "Who appears in the most events across both gaze and gesture data?" | short_answer | total event count per person | Hard |

---

## 4. Cross-Modal Interaction Patterns

어떤 gaze × gesture 조합이 자연스러운 cross-modal QA를 만드는지.

| Gesture Type | Naturally Paired Gaze Events | Cross-modal QA Focus |
|---|---|---|
| **pointing** → person | `joint_attention`, `attention_capture`, `gaze_following` | 가리킨 방향으로 다른 사람들이 보는가? |
| **showing** → person | `joint_attention`, `mutual_gaze` | 보여줄 때 상대가 보고 있는가? 그룹이 함께 보는가? |
| **giving** → person | `mutual_gaze`, `gaze_following` | 주고받을 때 눈 마주치는가? |
| **reaching** → object | `sudden_gaze_shift` | reach 직전에 gaze shift 있었나? |

> **Generation hint**: 위 매핑을 prompt에 넣어서 LLM이 자연스러운 cross-modal pair를 찾도록 유도.

---

## 5. Difficulty & Distribution

| Level | Categories | Target % |
|---|---|---|
| Medium | C1a-b, C3a | 30–35% |
| Hard | C2a-c, C3b, C4a-c | 65–70% |

> Cross-modal은 본질적으로 두 데이터를 동시에 참조해야 하므로 Easy 레벨 없음. 가장 쉬운 것도 Medium.

---

## 6. QA JSON Format

```json
{
  "video_id": "152_1_001",
  "dataset": "Werewolf",
  "category": "C2c",
  "format": "mcq",
  "question": "When Person 1 gives something to Person 4 at 4.0s, are they making eye contact?",
  "options": ["A) Yes, they are looking at each other", "B) Only Person 1 is looking at Person 4", "C) Only Person 4 is looking at Person 1", "D) No, they are not looking at each other"],
  "answer": "A",
  "answer_text": "Yes, they are looking at each other",
  "source_gaze_event_ids": [4001],
  "source_gesture_indices": [0],
  "time_range": [3.0, 6.0]
}
```

> **필드 추가**: `source_gaze_event_ids` + `source_gesture_indices` 둘 다 포함. 이전 단독 taxonomy와 호환.

---

## 7. API Prompts

### 7.1 Master Prompt (Cross-Modal 전용 시스템 프롬프트)

```
You are a QA generator for a cross-modal social reasoning dataset.
You will receive TWO types of annotations from the same video clip:
1. GAZE EVENTS: eye contact, joint attention, gaze following, gaze shifts, attention capture
2. GESTURE EVENTS: pointing, showing, giving, reaching

Your job: generate question-answer pairs that REQUIRE reasoning across BOTH modalities.

CRITICAL: Every question must need information from BOTH gaze AND gesture data to answer.
- BAD (single modality): "What type of gesture does P0 perform at 21s?" → answerable from gesture alone
- GOOD (cross-modal): "When P1 gives something to P4 at 4s, are they making eye contact?" → needs gesture (who gives to whom) + gaze (mutual_gaze at that time)

STRICT RULES:
1. Answers must be SHORT and FACTUAL.
   - MCQ: output the correct option letter only (A/B/C/D)
   - Short-answer: 1-5 words only
2. NEVER put reasoning, speculation, or interpretation in the answer.
   Banned words in answers: "suggesting", "indicating", "likely", "because", "implies", "seems", "probably"
3. Questions must require WATCHING THE VIDEO to answer.
4. Use natural language. Say "eye contact" not "mutual gaze event". Say "hands something to" not "initiates a giving gesture".
5. MCQ must have exactly 4 options with plausible distractors.
6. Person references: "Person {id}" format.
7. Include specific timestamps in questions.
8. "target_description" in gesture data is for context only. Do NOT use as GT answer.

CROSS-MODAL MATCHING RULES:
- Two events "co-occur" if their time ranges overlap by any amount.
- Two events are "temporally adjacent" if one ends within 3 seconds of the other starting.
- Person match: gesture target_person_id ∈ gaze persons_involved, or gesture initiator_id ∈ gaze persons_involved.

NATURAL CROSS-MODAL PATTERNS TO LOOK FOR:
- pointing + joint_attention/attention_capture: Does the group look where someone points?
- showing + joint_attention/mutual_gaze: Does the audience look at the shown object?
- giving + mutual_gaze: Do giver and receiver make eye contact during transfer?
- reaching + sudden_gaze_shift: Does a gaze shift precede a reaching action?
- Any gesture + gaze_following: Does someone follow another's gaze after a gesture?

OUTPUT: Return a JSON array. Each element:
{
  "category": "C1a" | "C1b" | "C2a" | "C2b" | "C2c" | "C3a" | "C3b" | "C4a" | "C4b" | "C4c",
  "format": "mcq" | "short_answer",
  "question": "...",
  "options": ["A) ...", "B) ...", "C) ...", "D) ..."],  // only for mcq
  "answer": "A",
  "answer_text": "...",
  "source_gaze_event_ids": [1000],
  "source_gesture_indices": [0],
  "time_range": [3.0, 6.0]
}
```

---

### 7.2 Cross-Modal Prompt (Main)

```
TASK: Generate 5-8 cross-modal QA pairs from the gaze and gesture data below.
Every question MUST require information from BOTH gaze and gesture data.

REQUIRED CATEGORY COVERAGE:
- At least 1 from C1 (Co-occurrence): Does a gaze event overlap with a gesture?
- At least 1 from C2 (Gaze Response): How does gaze change around a gesture?
  (only if gaze events exist within 3s of any gesture)
- At least 1 from C3 (Temporal Sequence): Which comes first across modalities?
- At least 1 from C4 (Person Dynamics): Same person across both modalities
  (only if gesture target or initiator appears in gaze events)

DIFFICULTY MIX:
- 2-3 Medium (C1, C3a): co-occurrence checks, temporal ordering
- 3-5 Hard (C2, C3b, C4): response patterns, causal chains, person dynamics

FORMAT MIX: ~65% MCQ, ~35% short_answer

VIDEO METADATA:
- video_id: {video_id}
- dataset: {dataset}
- persons_in_video: {person_ids}
- video_duration: {duration}s

GAZE EVENTS:
{gaze_events_json}

GESTURE EVENTS:
{gesture_events_json}

Generate the QA pairs now. Return ONLY the JSON array, no explanation.
```

---

### 7.3 Full Example

**입력:**

```
VIDEO METADATA:
- video_id: 152_1_001
- dataset: Werewolf
- persons_in_video: [0, 1, 2, 3, 4]
- video_duration: 120.0s

GAZE EVENTS:
[
  {"event_id": 4001, "event_type": "mutual_gaze",
   "start_time": 3.0, "end_time": 6.0,
   "persons_involved": [1, 4],
   "details": {"duration": 3.0, "num_frames": 6}},

  {"event_id": 1001, "event_type": "joint_attention",
   "start_time": 15.0, "end_time": 19.0,
   "persons_involved": [0, 1, 4],
   "details": {"mean_convergence": 0.78, "duration": 4.0, "num_persons": 3}},

  {"event_id": 3001, "event_type": "attention_capture",
   "start_time": 20.0, "end_time": 22.0,
   "persons_involved": [0, 1, 3],
   "details": {"num_persons": 3, "mean_velocity": 0.55,
               "person_velocities": {"0": 0.80, "1": 0.45, "3": 0.40}}},

  {"event_id": 1, "event_type": "sudden_gaze_shift",
   "start_time": 44.5, "end_time": 45.5,
   "persons_involved": [0],
   "details": {"max_velocity": 1.10, "duration": 1.0, "num_frames": 2}},

  {"event_id": 1002, "event_type": "joint_attention",
   "start_time": 64.0, "end_time": 68.5,
   "persons_involved": [0, 1, 3],
   "details": {"mean_convergence": 0.82, "duration": 4.5, "num_persons": 3}},

  {"event_id": 4002, "event_type": "mutual_gaze",
   "start_time": 69.0, "end_time": 71.0,
   "persons_involved": [3, 4],
   "details": {"duration": 2.0, "num_frames": 4}},

  {"event_id": 2001, "event_type": "gaze_following",
   "start_time": 80.0, "end_time": 82.0,
   "persons_involved": [1, 2],
   "details": {"leader_id": 2, "follower_id": 1, "lag_seconds": 1.2, "gaze_distance": 0.05}},

  {"event_id": 4003, "event_type": "mutual_gaze",
   "start_time": 103.0, "end_time": 106.0,
   "persons_involved": [0, 1],
   "details": {"duration": 3.0, "num_frames": 6}}
]

GESTURE EVENTS:
[
  {"gesture_type": "giving", "initiator_id": 1,
   "start_time": 4.0, "end_time": 5.1,
   "target_type": "person", "target_person_id": 4,
   "target_description": "P1 gives a card to P4", "confidence": 0.95},

  {"gesture_type": "showing", "initiator_id": 4,
   "start_time": 16.5, "end_time": 18.6,
   "target_type": "person", "target_person_id": 0,
   "target_description": "P4 holds up card toward P0", "confidence": 0.9},

  {"gesture_type": "giving", "initiator_id": 2,
   "start_time": 20.0, "end_time": 21.6,
   "target_type": "person", "target_person_id": 0,
   "target_description": "P2 gives a card to P0", "confidence": 0.95},

  {"gesture_type": "reaching", "initiator_id": 0,
   "start_time": 45.0, "end_time": 46.1,
   "target_type": "object", "target_person_id": null,
   "target_description": "yellow can", "confidence": 0.95},

  {"gesture_type": "pointing", "initiator_id": 3,
   "start_time": 65.0, "end_time": 68.0,
   "target_type": "person", "target_person_id": 4,
   "target_description": "P3 points at P4", "confidence": 0.85},

  {"gesture_type": "giving", "initiator_id": 4,
   "start_time": 69.0, "end_time": 70.6,
   "target_type": "person", "target_person_id": 3,
   "target_description": "P4 gives a card to P3", "confidence": 0.95},

  {"gesture_type": "giving", "initiator_id": 2,
   "start_time": 73.5, "end_time": 75.0,
   "target_type": "person", "target_person_id": 3,
   "target_description": "P2 gives a card to P3", "confidence": 0.95},

  {"gesture_type": "giving", "initiator_id": 2,
   "start_time": 80.0, "end_time": 81.5,
   "target_type": "person", "target_person_id": 1,
   "target_description": "P2 gives a card to P1", "confidence": 0.95},

  {"gesture_type": "showing", "initiator_id": 1,
   "start_time": 103.4, "end_time": 105.0,
   "target_type": "person", "target_person_id": 0,
   "target_description": "P1 shows a card to P0", "confidence": 0.9},

  {"gesture_type": "reaching", "initiator_id": 0,
   "start_time": 115.5, "end_time": 116.5,
   "target_type": "object", "target_person_id": null,
   "target_description": "object on table", "confidence": 0.9}
]
```

**기대 output:**

```json
[
  {
    "category": "C1a", "format": "mcq",
    "question": "While Person 1 is handing something to Person 4 around 4.0s, is there also a gaze interaction happening between them?",
    "options": ["A) Yes, they are making eye contact", "B) Yes, but only Person 1 is looking at Person 4", "C) No gaze interaction between them at that time", "D) They are both looking at someone else"],
    "answer": "A", "answer_text": "Yes, they are making eye contact",
    "source_gaze_event_ids": [4001], "source_gesture_indices": [0],
    "time_range": [3.0, 6.0]
  },

  {
    "category": "C1b", "format": "mcq",
    "question": "What type of gaze interaction is happening while Person 4 shows something to Person 0 between 16.5s and 18.6s?",
    "options": ["A) Eye contact between Person 4 and Person 0", "B) Multiple people looking at the same area", "C) A sudden gaze shift by Person 4", "D) No gaze event during this time"],
    "answer": "B", "answer_text": "Multiple people looking at the same area",
    "source_gaze_event_ids": [1001], "source_gesture_indices": [1],
    "time_range": [15.0, 19.0]
  },

  {
    "category": "C2a", "format": "mcq",
    "question": "When Person 2 gives something to Person 0 at 20.0s, do other people in the group shift their attention?",
    "options": ["A) Yes, Person 1 and Person 3 also shift attention", "B) Yes, but only Person 1 shifts attention", "C) No, no one else reacts", "D) Only Person 4 shifts attention"],
    "answer": "A", "answer_text": "Yes, Person 1 and Person 3 also shift attention",
    "source_gaze_event_ids": [3001], "source_gesture_indices": [2],
    "time_range": [20.0, 22.0]
  },

  {
    "category": "C2b", "format": "mcq",
    "question": "While Person 3 is pointing at Person 4 between 65s and 68s, are others in the group also looking in that direction?",
    "options": ["A) Yes, Person 0, Person 1, and Person 3 share attention toward that area", "B) Only Person 0 looks in that direction", "C) No one else looks where Person 3 is pointing", "D) Everyone including Person 4 is looking at Person 3"],
    "answer": "A", "answer_text": "Yes, Person 0, Person 1, and Person 3 share attention toward that area",
    "source_gaze_event_ids": [1002], "source_gesture_indices": [4],
    "time_range": [64.0, 68.5]
  },

  {
    "category": "C2c", "format": "mcq",
    "question": "When Person 4 gives something to Person 3 at 69.0s, are Person 4 and Person 3 making eye contact?",
    "options": ["A) Yes, they are looking at each other", "B) Only Person 4 looks at Person 3", "C) Only Person 3 looks at Person 4", "D) Neither is looking at the other"],
    "answer": "A", "answer_text": "Yes, they are looking at each other",
    "source_gaze_event_ids": [4002], "source_gesture_indices": [5],
    "time_range": [69.0, 71.0]
  },

  {
    "category": "C3a", "format": "mcq",
    "question": "Which happens first: Person 0's sudden gaze shift, or Person 0 reaching for an object around 45s?",
    "options": ["A) The gaze shift happens first", "B) The reaching happens first", "C) They happen at the same time", "D) Neither event occurs"],
    "answer": "A", "answer_text": "The gaze shift happens first",
    "source_gaze_event_ids": [1], "source_gesture_indices": [3],
    "time_range": [44.5, 46.1]
  },

  {
    "category": "C3b", "format": "mcq",
    "question": "What gesture happens right after the joint attention event among Person 0, 1, and 4 ends around 19.0s?",
    "options": ["A) Person 2 hands something to Person 0", "B) Person 0 reaches for an object", "C) Person 3 points at Person 4", "D) Person 4 hands something to Person 3"],
    "answer": "A", "answer_text": "Person 2 hands something to Person 0",
    "source_gaze_event_ids": [1001], "source_gesture_indices": [2],
    "time_range": [19.0, 21.6]
  },

  {
    "category": "C4a", "format": "mcq",
    "question": "Person 3 points at Person 4 from 65s to 68s. Is Person 4 involved in any gaze interaction around that time?",
    "options": ["A) Yes, Person 4 is part of the group's shared attention", "B) Yes, Person 4 makes eye contact with Person 3", "C) No, Person 4 is not in any gaze event at that time", "D) Person 4 is involved in a gaze event but only after 69s"],
    "answer": "D", "answer_text": "Person 4 is involved in a gaze event but only after 69s",
    "source_gaze_event_ids": [1002, 4002], "source_gesture_indices": [4],
    "time_range": [64.0, 71.0]
  },

  {
    "category": "C4b", "format": "mcq",
    "question": "When Person 1 shows a card to Person 0 at 103.4s, does Person 0 look back at Person 1?",
    "options": ["A) Yes, they make eye contact", "B) Person 0 looks at the card but not at Person 1", "C) Person 0 looks away", "D) No gaze data exists for this moment"],
    "answer": "A", "answer_text": "Yes, they make eye contact",
    "source_gaze_event_ids": [4003], "source_gesture_indices": [8],
    "time_range": [103.0, 106.0]
  },

  {
    "category": "C4c", "format": "short_answer",
    "question": "Which person is involved in the most events across both gaze and gesture data combined in this clip?",
    "answer": "Person 0", "answer_text": "Person 0",
    "source_gaze_event_ids": [1001, 3001, 1, 1002, 4003],
    "source_gesture_indices": [1, 2, 3, 8, 9],
    "time_range": [0.0, 120.0]
  }
]
```

**정답 검증:**

| QA | Category | Cross-modal 근거 |
|---|---|---|
| 1 (C1a) | giving P1→P4 (4.0-5.1) overlaps mutual_gaze P1,P4 (3.0-6.0) | ✓ gesture initiator+target = gaze participants |
| 2 (C1b) | showing P4→P0 (16.5-18.6) overlaps joint_attention P0,P1,P4 (15.0-19.0) | ✓ gesture target P0 ∈ gaze participants |
| 3 (C2a) | giving P2→P0 (20.0-21.6) co-occurs with attention_capture P0,P1,P3 (20.0-22.0) | ✓ attention shift coincides with giving |
| 4 (C2b) | pointing P3→P4 (65-68) overlaps joint_attention P0,P1,P3 (64-68.5) | ✓ group attention during pointing |
| 5 (C2c) | giving P4→P3 (69.0-70.6) overlaps mutual_gaze P3,P4 (69.0-71.0) | ✓ eye contact during transfer |
| 6 (C3a) | sudden_gaze_shift P0 starts 44.5 < reaching P0 starts 45.0 | ✓ gaze first, then reach |
| 7 (C3b) | joint_attention ends ~19.0, next gesture = giving P2→P0 at 20.0 | ✓ 1.0s gap |
| 8 (C4a) | P3 points at P4, but P4 NOT in joint_attention(64-68.5)={P0,P1,P3}. P4 appears in mutual_gaze(69-71) which is after | ✓ target not in co-occurring gaze, but in later gaze |
| 9 (C4b) | showing P1→P0 (103.4-105) overlaps mutual_gaze P0,P1 (103-106) | ✓ reciprocal gaze response |
| 10 (C4c) | P0: gaze 5 events (1001,3001,1,1002,4003) + gesture 5 (idx 1,2,3,8,9 as target/initiator) = 10 total | ✓ most events |

---

### 7.4 Category-Specific Addon Prompts

#### C2 전용 (giving/showing 제스처와 mutual_gaze가 동시에 존재할 때)

```
ADDITIONAL TASK: Generate 2 more QA pairs focused on C2 (Gaze Response to Gesture).
The video has person-to-person gestures that temporally overlap with gaze events.

Focus on:
- C2c: During object transfers (giving/showing), do the participants make eye contact?
- C2b: After a pointing gesture, does the group's attention converge?

Find gesture-gaze pairs where time ranges overlap or are within 3 seconds.
Same rules as before. Return JSON array.
```

#### C4 전용 (같은 사람이 양쪽 모달리티에 3번 이상 등장할 때)

```
ADDITIONAL TASK: Generate 2 more QA pairs focused on C4 (Cross-Modal Person Dynamics).
Some people appear frequently in both gaze and gesture events.

Focus on:
- C4b: After Person X gestures toward Person Y, does Person Y respond via gaze (look at X)?
- C4c: Who is the most socially central person based on combined gaze + gesture data?

Same rules as before. Return JSON array.
```

---

## 8. Generation Script 로직 (pseudo-code)

```python
def generate_crossmodal_qa(video_id, gaze_events, gesture_events,
                           person_ids, duration, dataset):
    # 1. Filter by confidence
    gaze_events = [e for e in gaze_events if e["confidence"] >= 0.9]
    gesture_events = [g for g in gesture_events if g["confidence"] >= 0.85]

    # 2. Check if cross-modal QA is feasible
    if len(gaze_events) == 0 or len(gesture_events) == 0:
        return []  # need both modalities

    # 3. Find temporal overlaps / adjacencies
    cross_pairs = find_cross_modal_pairs(gaze_events, gesture_events, max_gap=3.0)
    if len(cross_pairs) < 2:
        return []  # not enough cross-modal signal

    # 4. Determine valid categories
    categories = {"C1", "C3"}  # always valid if cross_pairs exist

    # C2: need gaze events near gestures
    person_gestures = [g for g in gesture_events if g["target_type"] == "person"]
    if any(has_nearby_gaze(g, gaze_events, max_gap=3.0) for g in person_gestures):
        categories.add("C2")

    # C4: need same person in both modalities
    gaze_persons = set()
    for e in gaze_events:
        gaze_persons.update(e["persons_involved"])
    gesture_persons = set()
    for g in gesture_events:
        gesture_persons.add(g["initiator_id"])
        if g["target_person_id"] is not None:
            gesture_persons.add(g["target_person_id"])
    if gaze_persons & gesture_persons:  # intersection
        categories.add("C4")

    # 5. Fill prompt
    filled = CROSSMODAL_SYSTEM_PROMPT + CROSSMODAL_MAIN_PROMPT.format(
        video_id=video_id,
        dataset=dataset,
        person_ids=person_ids,
        duration=duration,
        gaze_events_json=json.dumps(gaze_events, indent=2),
        gesture_events_json=json.dumps(gesture_events, indent=2)
    )

    # 6. Call API
    response = call_llm_api(filled)
    qa_pairs = json.loads(response)

    # 7. Optional addon
    mutual_gaze_during_gesture = find_overlapping_pairs(
        [g for g in gesture_events if g["gesture_type"] in ("giving", "showing")],
        [e for e in gaze_events if e["event_type"] == "mutual_gaze"]
    )
    if len(mutual_gaze_during_gesture) >= 2:
        addon = call_llm_api(filled + C2_ADDON_PROMPT)
        qa_pairs.extend(json.loads(addon))

    # 8. Validate
    qa_pairs = validate_crossmodal(qa_pairs, gaze_events, gesture_events, person_ids)

    # 9. Metadata
    for qa in qa_pairs:
        qa["video_id"] = video_id
        qa["dataset"] = dataset

    return qa_pairs


def find_cross_modal_pairs(gaze_events, gesture_events, max_gap=3.0):
    """Find (gaze, gesture) pairs that overlap or are within max_gap seconds."""
    pairs = []
    for gi, gaze in enumerate(gaze_events):
        for gj, gesture in enumerate(gesture_events):
            overlap = (min(gaze["end_time"], gesture["end_time"])
                      - max(gaze["start_time"], gesture["start_time"]))
            if overlap >= 0:  # overlapping
                pairs.append((gi, gj, "overlap", overlap))
            else:
                gap = -overlap
                if gap <= max_gap:
                    pairs.append((gi, gj, "adjacent", gap))
    return pairs


def validate_crossmodal(qa_pairs, gaze_events, gesture_events, person_ids):
    valid = []
    for qa in qa_pairs:
        # Basic answer checks (same as single modality)
        if qa["format"] == "mcq" and len(qa["answer"]) != 1:
            continue
        if qa["format"] == "short_answer" and len(qa["answer"].split()) > 10:
            continue
        banned = ["suggesting", "indicating", "likely", "because",
                  "implies", "emotion", "probably"]
        if any(w in qa["answer"].lower() for w in banned):
            continue

        # Person ID check
        mentioned_ids = re.findall(r'Person (\d+)', qa["question"] + qa["answer_text"])
        if any(int(pid) not in person_ids for pid in mentioned_ids):
            continue

        # CROSS-MODAL CHECK: must reference both modalities
        has_gaze_ref = len(qa.get("source_gaze_event_ids", [])) > 0
        has_gesture_ref = len(qa.get("source_gesture_indices", [])) > 0
        if not (has_gaze_ref and has_gesture_ref):
            continue  # not truly cross-modal → reject

        # Index bounds
        if any(idx >= len(gesture_events) for idx in qa.get("source_gesture_indices", [])):
            continue

        valid.append(qa)
    return valid
```

---

## 9. MCQ Distractor Strategy

| Question type | Correct source | Distractor strategy |
|---|---|---|
| "Eye contact during giving?" | mutual_gaze ∩ giving time | "Only X looks at Y", "Neither looks", "Both look elsewhere" |
| "Group looks where pointed?" | joint_attention ∩ pointing time | "Only one person looks", "No one looks", "They look at pointer instead" |
| "Gaze shift before reach?" | sudden_gaze_shift before reaching | "Reach first", "Same time", "Neither happens" |
| "What gesture after gaze event?" | next gesture by time | Other gestures from the clip, "No gesture follows" |
| "Which first: gaze or gesture?" | start_time comparison | Swap order, "Same time", "Neither happens" |
| "Is target in gaze event?" | persons_involved check | "Yes in different event", "No", "At a different time" |
| "Most central person?" | combined count | 2nd/3rd most frequent persons |

---

## 10. Quality Filters

| Filter | Rule |
|---|---|
| Answer length | MCQ: 1 char. Short: ≤10 words |
| Cross-modal requirement | Must have both `source_gaze_event_ids` and `source_gesture_indices` non-empty |
| Hallucination | No emotions, dialogue, objects not in data |
| Data match | Person IDs, timestamps, event types traceable to input JSON |
| Temporal validity | claimed overlap/adjacency must actually exist in data (re-check timestamps) |
| Not single-modal | Question must NOT be answerable from one modality alone |
| Distractor quality | MCQ options distinct and plausible |
| Dedup | >60% unigram overlap with another Q from same video → drop |
| No target_description leak | Answer must NOT copy gesture target_description verbatim |

---

## 11. 전체 QA Pipeline 요약 (3-Layer)

```
Video (with gaze + gesture annotations)
│
├─ Layer 1: Gaze-only QA         (T1-T5, 기존 taxonomy)
│   └─ 입력: gaze_events만
│
├─ Layer 2: Gesture-only QA      (G1-G5, 기존 taxonomy)
│   └─ 입력: gesture_events만
│
└─ Layer 3: Cross-modal QA       (C1-C4, 이 문서)
    └─ 입력: gaze_events + gesture_events 동시
    └─ 조건: 양쪽 모달리티에 이벤트가 있고, temporal overlap/adjacency가 2쌍 이상

각 Layer는 독립적으로 생성 가능.
최종 QA set = Layer 1 ∪ Layer 2 ∪ Layer 3
```

### 예상 비율

| Layer | QA/video (avg) | Notes |
|---|---|---|
| Gaze-only (T1-T5) | 4-8 | ~46K videos |
| Gesture-only (G1-G5) | 3-6 | gesture 있는 비디오만 |
| Cross-modal (C1-C4) | 3-5 | 양쪽 모달리티 이벤트 + overlap 있는 비디오만 |
