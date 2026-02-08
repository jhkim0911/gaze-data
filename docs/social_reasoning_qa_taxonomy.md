# Social Gaze QA Taxonomy for Video-LLM Post-Training (GRPO)

> Gaze event JSON → GPT/Gemini → QA generation → Video + QA → GRPO training
>
> **핵심**: 모델은 비디오를 보고 답한다. Answer는 짧고 factual. CoT는 GRPO가 알아서 학습.

---

## 0. Data Inventory

| Event Type | Fields |
|---|---|
| 공통 | `event_type`, `start_time`, `end_time`, `confidence`, `persons_involved: [int]` |
| `sudden_gaze_shift` | `max_velocity`, `duration`, `num_frames` |
| `joint_attention` | `max_convergence`, `mean_convergence`, `duration`, `attention_center`, `num_persons` |
| `gaze_following` | `leader_id`, `follower_id`, `lag_seconds`, `gaze_distance`, `gaze_target` |
| `attention_capture` | `num_persons`, `mean_velocity`, `person_velocities` |
| `mutual_gaze` | `duration`, `num_frames` |

---

## 1. Design Principles

- **Answer**: 짧고 factual. MCQ는 A/B/C/D. Short-answer는 1-5 words. No reasoning in answer.
- **Question**: 비디오를 봐야 답할 수 있고, JSON에서 GT를 뽑을 수 있어야 함.
- **Format**: MCQ 50-60%, Short-answer 30-40%, Open-ended ≤10%
- **Yes/no**: 최소화 (≤5%)

---

## 2. Taxonomy

```
SOCIAL GAZE QA
│
├─ T1. Gaze Target Identification          ← "누가 누구를 보고 있나"
│   ├─ T1a. Person-to-Person gaze target
│   └─ T1b. Gaze target change detection
│
├─ T2. Gaze Event Recognition              ← "어떤 종류의 gaze event인가"
│   ├─ T2a. Event type classification
│   ├─ T2b. Event detection
│   └─ T2c. Participant identification
│
├─ T3. Temporal Gaze Reasoning             ← "언제, 얼마나, 무엇이 먼저"
│   ├─ T3a. Duration
│   ├─ T3b. Ordering
│   └─ T3c. Before/After
│
├─ T4. Inter-Person Dynamics               ← "누가 리드, 누가 팔로우, 누가 짝"
│   ├─ T4a. Leader–Follower
│   ├─ T4b. Mutual gaze pairs
│   └─ T4c. Gaze asymmetry
│
└─ T5. Group Gaze Patterns                 ← "그룹 전체 패턴"
    ├─ T5a. Shared attention (who's in/out)
    ├─ T5b. Subgroup identification
    └─ T5c. Attention shift patterns
```

---

## 3. Event Type → Category Mapping

| | T1 (Target) | T2 (Recognition) | T3 (Temporal) | T4 (Person) | T5 (Group) |
|---|:---:|:---:|:---:|:---:|:---:|
| sudden_gaze_shift | ✓ change | ✓ classify | ✓ timing | — | — |
| joint_attention | — | ✓ classify, detect | ✓ duration | — | ✓ shared, subgroup |
| gaze_following | ✓ target | ✓ classify | ✓ lag, order | ✓ leader/follower | — |
| attention_capture | — | ✓ classify, detect | ✓ timing | — | ✓ who's in/out |
| mutual_gaze | ✓ target | ✓ classify | ✓ duration | ✓ pairs, asymmetry | ✓ in/out |

---

## 4. Difficulty & Distribution

| Level | Categories | Target % |
|---|---|---|
| Easy | T1a-b, T2b | 25–30% |
| Medium | T2a, T2c, T3a, T3b | 35–40% |
| Hard | T3c, T4a-c, T5a-c | 30–35% |

---

## 5. QA JSON Format

```json
{"video_id": "str", "dataset": "str", "category": "T1a",
 "event_ids": [4000], "time_range": [13.0, 15.5],
 "format": "mcq",
 "question": "At 14.0s, who is Person 1 looking at?",
 "options": ["Person 0", "Person 2", "Person 3", "No one in particular"],
 "answer": "C", "answer_text": "Person 3"}
```

```json
{"video_id": "str", "dataset": "str", "category": "T3a",
 "format": "short_answer",
 "question": "How long do Person 1 and Person 3 maintain eye contact?",
 "answer": "About 2.5 seconds"}
```

---

## 6. API Prompts

### 6.1 Master Prompt (모든 비디오에 공통으로 붙는 시스템 프롬프트)

```
You are a QA generator for a social gaze reasoning dataset.
You will receive gaze event annotations detected in a video clip. Each event has a type, timestamps, involved persons, and type-specific details.

Your job: generate question-answer pairs that a Video-LLM must answer by WATCHING the video.

STRICT RULES:
1. Answers must be SHORT and FACTUAL.
   - MCQ: output the correct option letter only (A/B/C/D)
   - Short-answer: 1-5 words only
   - Open-ended: max 2 sentences, no interpretation
2. NEVER put reasoning, speculation, or interpretation in the answer.
   Banned words in answers: "suggesting", "indicating", "likely", "because", "implies", "seems"
3. Questions must require WATCHING THE VIDEO to answer — they should not be answerable from the question text alone.
4. Use natural language in questions. Say "eye contact" not "mutual gaze event". Say "looking at the same thing" not "joint attention with high convergence".
5. MCQ must have exactly 4 options. Distractors must be plausible (other person IDs present in the video, other event types that didn't happen, swapped roles, etc.)
6. All person references must use "Person {id}" format matching the event data.
7. Include specific timestamps from the event data in questions.

OUTPUT: Return a JSON array. Each element:
{
  "category": "T1a",
  "format": "mcq" | "short_answer",
  "question": "...",
  "options": ["A) ...", "B) ...", "C) ...", "D) ..."],  // only for mcq
  "answer": "C",           // letter for mcq, text for short_answer
  "answer_text": "...",    // full text of correct answer (for both formats)
  "source_event_ids": [4000],
  "time_range": [13.0, 15.5]
}
```

---

### 6.2 Single-Event Prompt (이벤트 1-2개인 비디오, ~20K videos)

카테고리 T1, T2만 생성. 가장 기본적인 QA.

```
TASK: Generate 2-3 QA pairs from the gaze event(s) below.

CATEGORIES TO COVER:
- T1a (Person-to-Person gaze target): "Who is Person X looking at at time T?" → MCQ
- T1b (Gaze change detection): "Does Person X change gaze direction between T1 and T2?" → MCQ
- T2a (Event type classification): "What best describes the gaze interaction at time T?" → MCQ
- T2b (Event detection): "Is there [eye contact / gaze following / etc.] in this clip?" → MCQ
- T2c (Participant identification): "Who is involved in [event] at time T?" → MCQ or short_answer

FORMAT MIX: at least 1 MCQ, rest can be MCQ or short_answer.

VIDEO METADATA:
- video_id: {video_id}
- dataset: {dataset}
- total_persons_in_video: {person_ids}
- video_duration: {duration}s

GAZE EVENTS:
{events_json}

Generate the QA pairs now. Return ONLY the JSON array, no explanation.
```

**활용 예시 (mutual_gaze only):**

```
VIDEO METADATA:
- video_id: friends_mmc_2341
- dataset: FriendsMMC
- total_persons_in_video: [0, 1, 2]
- video_duration: 45.0s

GAZE EVENTS:
[
  {"event_id": 4000, "event_type": "mutual_gaze",
   "start_time": 8.0, "end_time": 10.5,
   "persons_involved": [0, 2],
   "details": {"duration": 2.5, "num_frames": 5}}
]
```

**기대 output:**
```json
[
  {"category": "T1a", "format": "mcq",
   "question": "At 9.0s, who is Person 0 looking at?",
   "options": ["A) Person 1", "B) Person 2", "C) No one in particular", "D) Looking away from all people"],
   "answer": "B", "answer_text": "Person 2",
   "source_event_ids": [4000], "time_range": [8.0, 10.5]},

  {"category": "T2a", "format": "mcq",
   "question": "What best describes the gaze interaction between Person 0 and Person 2 from 8.0s to 10.5s?",
   "options": ["A) Person 0 looks at Person 2 but Person 2 looks away", "B) Both are looking at each other", "C) Both are looking at Person 1", "D) They are looking in opposite directions"],
   "answer": "B", "answer_text": "Both are looking at each other",
   "source_event_ids": [4000], "time_range": [8.0, 10.5]},

  {"category": "T2c", "format": "short_answer",
   "question": "Which two people are making eye contact between 8.0s and 10.5s?",
   "answer": "Person 0 and Person 2", "answer_text": "Person 0 and Person 2",
   "source_event_ids": [4000], "time_range": [8.0, 10.5]}
]
```

---

### 6.3 Multi-Event Prompt (이벤트 3개 이상, ~26K videos)

T1-T5 전체 커버. 난이도 mix.

```
TASK: Generate 6-10 QA pairs from the gaze events below.
Cover a MIX of difficulty levels and categories.

REQUIRED CATEGORY COVERAGE:
- At least 1 from T1 (Gaze Target): who is looking at whom
- At least 1 from T2 (Event Recognition): what type of gaze event
- At least 1 from T3 (Temporal): duration, ordering, before/after
- At least 1 from T4 (Inter-Person): leader/follower, pairs, asymmetry
  (only if gaze_following or mutual_gaze events exist)
- At least 1 from T5 (Group Patterns): who's in/out, subgroups
  (only if joint_attention or attention_capture events exist)

DIFFICULTY MIX:
- 2-3 Easy (T1, T2b): single-event, direct identification
- 3-4 Medium (T2a, T2c, T3a, T3b): classification, duration, ordering
- 2-3 Hard (T3c, T4, T5): cross-event reasoning, dynamics, group patterns

FORMAT MIX: ~60% MCQ, ~40% short_answer

VIDEO METADATA:
- video_id: {video_id}
- dataset: {dataset}
- total_persons_in_video: {person_ids}
- video_duration: {duration}s

GAZE EVENTS:
{events_json}

Generate the QA pairs now. Return ONLY the JSON array, no explanation.
```

**활용 예시 (4 events):**

```
VIDEO METADATA:
- video_id: werewolf_015_seg3
- dataset: Werewolf
- total_persons_in_video: [0, 1, 2, 3]
- video_duration: 120.0s

GAZE EVENTS:
[
  {"event_id": 3000, "event_type": "attention_capture",
   "start_time": 4.5, "end_time": 4.5,
   "persons_involved": [0, 2, 3],
   "details": {"num_persons": 3, "mean_velocity": 0.50,
               "person_velocities": {"0": 0.93, "2": 0.62, "3": 0.96}}},

  {"event_id": 1000, "event_type": "joint_attention",
   "start_time": 5.0, "end_time": 10.0,
   "persons_involved": [0, 2, 3],
   "details": {"mean_convergence": 0.81, "duration": 5.0, "num_persons": 3}},

  {"event_id": 2000, "event_type": "gaze_following",
   "start_time": 10.5, "end_time": 12.0,
   "persons_involved": [0, 3],
   "details": {"leader_id": 0, "follower_id": 3, "lag_seconds": 1.5,
               "gaze_distance": 0.002}},

  {"event_id": 4000, "event_type": "mutual_gaze",
   "start_time": 13.0, "end_time": 15.5,
   "persons_involved": [1, 3],
   "details": {"duration": 2.5, "num_frames": 6}}
]
```

**기대 output:**
```json
[
  {"category": "T1a", "format": "mcq",
   "question": "At 14.0s, who is Person 1 looking at?",
   "options": ["A) Person 0", "B) Person 2", "C) Person 3", "D) No one in particular"],
   "answer": "C", "answer_text": "Person 3",
   "source_event_ids": [4000], "time_range": [13.0, 15.5]},

  {"category": "T1b", "format": "mcq",
   "question": "Does Person 1 abruptly change their gaze direction between 6.0s and 6.5s?",
   "options": ["A) Yes, they shift rapidly", "B) No, their gaze stays steady"],
   "answer": "A", "answer_text": "Yes, they shift rapidly",
   "source_event_ids": [0], "time_range": [6.0, 6.5]},

  {"category": "T2a", "format": "mcq",
   "question": "Between 10.5s and 12.0s, Person 3 looks at the same spot Person 0 was already looking at. What type of gaze behavior is this?",
   "options": ["A) Mutual eye contact", "B) Gaze following", "C) Simultaneous attention shift", "D) Gaze avoidance"],
   "answer": "B", "answer_text": "Gaze following",
   "source_event_ids": [2000], "time_range": [10.5, 12.0]},

  {"category": "T2c", "format": "short_answer",
   "question": "Who is NOT part of the simultaneous gaze shift at 4.5s?",
   "answer": "Person 1", "answer_text": "Person 1",
   "source_event_ids": [3000], "time_range": [4.5, 4.5]},

  {"category": "T3a", "format": "mcq",
   "question": "How long do Person 1 and Person 3 maintain eye contact?",
   "options": ["A) Less than 1 second", "B) About 1 second", "C) About 2-3 seconds", "D) More than 5 seconds"],
   "answer": "C", "answer_text": "About 2-3 seconds",
   "source_event_ids": [4000], "time_range": [13.0, 15.5]},

  {"category": "T3c", "format": "mcq",
   "question": "What happens right after the group's shared attention ends at 10.0s?",
   "options": ["A) Person 3 follows Person 0's gaze to a new target", "B) Person 1 and Person 3 make eye contact", "C) Everyone shifts gaze simultaneously again", "D) The group continues shared attention"],
   "answer": "A", "answer_text": "Person 3 follows Person 0's gaze to a new target",
   "source_event_ids": [2000, 1000], "time_range": [10.0, 12.0]},

  {"category": "T4a", "format": "mcq",
   "question": "Between 10.5s and 12.0s, one person looks at a target and another follows. Who looks first?",
   "options": ["A) Person 0", "B) Person 3", "C) They look at the same time", "D) Neither is looking at the target"],
   "answer": "A", "answer_text": "Person 0",
   "source_event_ids": [2000], "time_range": [10.5, 12.0]},

  {"category": "T4b", "format": "mcq",
   "question": "Which pair of people makes eye contact in this clip?",
   "options": ["A) Person 0 and Person 2", "B) Person 0 and Person 3", "C) Person 1 and Person 3", "D) Person 2 and Person 3"],
   "answer": "C", "answer_text": "Person 1 and Person 3",
   "source_event_ids": [4000], "time_range": [13.0, 15.5]},

  {"category": "T5a", "format": "mcq",
   "question": "Person 0, 2, and 3 share attention from 5.0s to 10.0s. Is Person 1 also looking at the same area?",
   "options": ["A) Yes", "B) No"],
   "answer": "B", "answer_text": "No",
   "source_event_ids": [1000], "time_range": [5.0, 10.0]},

  {"category": "T5b", "format": "mcq",
   "question": "Person 0, 2, 3 share attention and react together. Person 1 only has eye contact with Person 3. Which best describes the group structure?",
   "options": ["A) All four form one cohesive group", "B) Person 0, 2, 3 form a subgroup; Person 1 connects only through Person 3", "C) Person 1 and 3 are the main pair; others are separate", "D) Everyone acts independently"],
   "answer": "B", "answer_text": "Person 0, 2, 3 form a subgroup; Person 1 connects only through Person 3",
   "source_event_ids": [1000, 3000, 4000], "time_range": [4.5, 15.5]}
]
```

---

### 6.4 Category-Specific Addon Prompts

multi-event prompt 뒤에 추가로 붙여서 특정 카테고리를 더 생성할 때 사용.

#### T4 전용 (gaze_following 이벤트가 2개 이상일 때)

```
ADDITIONAL TASK: Generate 2 more QA pairs focused on T4 (Inter-Person Dynamics).
The video has multiple gaze_following events. Use cross-event information.

Focus on:
- T4a: Who is the most frequent gaze leader across events?
- T4c: Is the gaze-following relationship one-directional or mutual?

Same rules as before. Return JSON array.
```

#### T5 전용 (3인 이상 비디오, joint_attention + attention_capture 모두 있을 때)

```
ADDITIONAL TASK: Generate 2 more QA pairs focused on T5 (Group Patterns).
The video has both joint_attention and attention_capture events.

Focus on:
- T5a: Who is consistently included in or excluded from group-level events?
- T5b: Do the same people appear together across multiple group events?

Same rules as before. Return JSON array.
```

#### T3c 전용 (이벤트 3개 이상, 시간적으로 연속)

```
ADDITIONAL TASK: Generate 2 more QA pairs focused on T3c (Before/After reasoning).
The video has multiple events in temporal sequence.

Focus on:
- What event immediately follows another?
- What changes after a specific event ends?

Use ONLY adjacent events in the timeline. Same rules. Return JSON array.
```

---

## 7. Generation Script 로직 (pseudo-code)

```python
def generate_qa_for_video(video_id, events, person_ids, duration, dataset):
    # 1. Filter events
    events = [e for e in events if e["confidence"] >= 0.9]

    # 2. Pick prompt based on event count
    if len(events) <= 2:
        prompt = SINGLE_EVENT_PROMPT
        n_qa = 2-3
        categories = ["T1", "T2"]
    else:
        prompt = MULTI_EVENT_PROMPT
        n_qa = 6-10
        categories = ["T1", "T2", "T3", "T4", "T5"]

    # 3. Check which categories are valid
    event_types = {e["event_type"] for e in events}
    if "gaze_following" not in event_types and "mutual_gaze" not in event_types:
        categories.remove("T4")
    if "joint_attention" not in event_types and "attention_capture" not in event_types:
        categories.remove("T5")

    # 4. Fill prompt template
    filled = SYSTEM_PROMPT + prompt.format(
        video_id=video_id,
        dataset=dataset,
        person_ids=person_ids,
        duration=duration,
        events_json=json.dumps(events, indent=2)
    )

    # 5. Call API
    response = call_llm_api(filled)
    qa_pairs = json.loads(response)

    # 6. Optional: add category-specific addon
    gf_count = sum(1 for e in events if e["event_type"] == "gaze_following")
    if gf_count >= 2:
        addon = call_llm_api(filled + T4_ADDON_PROMPT)
        qa_pairs.extend(json.loads(addon))

    # 7. Post-process & validate
    qa_pairs = validate_and_filter(qa_pairs, events, person_ids)

    # 8. Attach metadata
    for qa in qa_pairs:
        qa["video_id"] = video_id
        qa["dataset"] = dataset

    return qa_pairs

def validate_and_filter(qa_pairs, events, person_ids):
    valid = []
    for qa in qa_pairs:
        # Answer length check
        if qa["format"] == "mcq" and len(qa["answer"]) != 1:
            continue
        if qa["format"] == "short_answer" and len(qa["answer"].split()) > 10:
            continue
        # Hallucination check
        banned = ["suggesting", "indicating", "likely", "because", "implies", "emotion"]
        if any(w in qa["answer"].lower() for w in banned):
            continue
        # Person ID check
        mentioned_ids = re.findall(r'Person (\d+)', qa["question"] + qa["answer_text"])
        if any(int(pid) not in person_ids for pid in mentioned_ids):
            continue
        valid.append(qa)
    return valid
```

---

## 8. 생성 가능량 추정

| Videos | Count | QA/video | Total QA |
|---|---|---|---|
| 1-2 events | ~20K | 2-3 | ~50K |
| 3-5 events | ~18K | 5-7 | ~110K |
| 6+ events | ~8K | 8-12 | ~80K |
| **Total** | **46K** | | **~240K** |

---

## 9. MCQ Distractor Strategy

| Question type | Correct source | Distractor strategy |
|---|---|---|
| "Who is P1 looking at?" | `persons_involved` | Other person IDs in video |
| "What type of event?" | `event_type` | Other 4 event types |
| "Who follows whose gaze?" | `leader_id`/`follower_id` | Swap roles, other persons |
| "What happens after X?" | Next event in time | Other events, "nothing" |
| "How long?" | `duration` | ±50%, other event durations |
| "Who's not in the group?" | Missing from `persons_involved` | Persons who ARE involved |

---

## 10. Quality Filters

| Filter | Rule |
|---|---|
| Answer length | MCQ: 1 char. Short: ≤10 words. OE: ≤30 words |
| Hallucination | No emotions, dialogue, objects, scene context |
| Data match | Person IDs, timestamps, numbers traceable to input JSON |
| Distractor quality | MCQ options distinct and plausible |
| Video-dependence | Q not answerable from question text alone |
| Dedup | >60% unigram overlap with another Q from same video → drop |
