# Gaze Event Statistics Summary

## Overview

| Metric | Value |
|--------|-------|
| **Total Videos** | 46,158 |
| **Total Events (conf ≥ 0.9)** | 215,378 |
| **Avg Events per Video** | 4.67 |
| **Datasets** | 7 |

---

## Event Taxonomy

### 1. Sudden Gaze Shift

**Description**: Rapid head/gaze movement by a single person  
**Detection Rule**: Gaze velocity > 0.4 threshold with confidence based on max velocity  
**Use case**: Detecting reactions, attention shifts, surprise moments

```json
{
  "event_id": 0,
  "event_type": "sudden_gaze_shift",
  "start_time": 6.0,
  "end_time": 6.5,
  "confidence": 0.83,
  "persons_involved": [1],
  "details": {
    "max_velocity": 1.16,
    "duration": 0.5,
    "num_frames": 2
  }
}
```

---

### 2. Joint Attention

**Description**: Multiple people looking at the same region simultaneously  
**Detection Rule**: Convergence score ≥ 0.6 for 2+ people, sustained for ≥ 0.5s  
**Use case**: Shared focus on objects, speakers, or events

```json
{
  "event_id": 1000,
  "event_type": "joint_attention",
  "start_time": 0.0,
  "end_time": 5.0,
  "confidence": 0.81,
  "persons_involved": [0, 2, 3],
  "details": {
    "max_convergence": 0.98,
    "mean_convergence": 0.81,
    "duration": 5.0,
    "attention_center": [0.723, 0.202],
    "num_persons": 3
  }
}
```

---

### 3. Gaze Following

**Description**: Person B looks where Person A just looked (temporal sequence)  
**Detection Rule**: Gaze target distance < 0.1 with temporal lag of 0.5-2.0s  
**Use case**: Social referencing, following directions, visual cues

```json
{
  "event_id": 2000,
  "event_type": "gaze_following",
  "start_time": 5.0,
  "end_time": 6.5,
  "confidence": 0.93,
  "persons_involved": [0, 3],
  "details": {
    "leader_id": 0,
    "follower_id": 3,
    "lag_seconds": 1.5,
    "gaze_distance": 0.002,
    "gaze_target": [0.725, 0.202]
  }
}
```

---

### 4. Attention Capture

**Description**: Multiple people suddenly shift gaze to same location simultaneously  
**Detection Rule**: 2+ people with gaze velocity > 0.4 in same frame, converging  
**Use case**: Salient events, interruptions, surprising stimuli

```json
{
  "event_id": 3000,
  "event_type": "attention_capture",
  "start_time": 4.5,
  "end_time": 4.5,
  "confidence": 1.0,
  "persons_involved": [0, 2, 3],
  "details": {
    "num_persons": 3,
    "mean_velocity": 0.50,
    "person_velocities": {
      "0": 0.93, "2": 0.62, "3": 0.96
    }
  }
}
```

---

### 5. Mutual Gaze

**Description**: Two people looking directly at each other  
**Detection Rule**: Bidirectional gaze alignment for ≥ 3 consecutive frames  
**Use case**: Eye contact, conversation engagement, social bonding

```json
{
  "event_id": 4000,
  "event_type": "mutual_gaze",
  "start_time": 13.0,
  "end_time": 15.5,
  "confidence": 1.0,
  "persons_involved": [1, 3],
  "details": {
    "duration": 2.5,
    "num_frames": 6
  }
}
```

---

## Dataset Statistics (conf ≥ 0.9)

| Dataset | Videos | Events | Avg/Video |
|---------|-------:|-------:|----------:|
| AVSBench | 1,884 | 1,979 | 1.05 |
| Embody3D | 3,854 | 17,201 | 4.46 |
| FriendsMMC | 5,077 | 13,239 | 2.61 |
| Social Gesture | 1,424 | 28,011 | 19.67 |
| Social-IQ | 1,114 | 3,124 | 2.80 |
| TVQA | 30,185 | 52,802 | 1.75 |
| Werewolf | 2,620 | 99,022 | 37.79 |
| **Total** | **46,158** | **215,378** | **4.67** |

---

## Event Distribution by Type (conf ≥ 0.9)

| Event Type | Count | % of Total |
|------------|------:|-----------:|
| Gaze Following | 67,108 | 31.2% |
| Attention Capture | 49,758 | 23.1% |
| Joint Attention | 38,830 | 18.0% |
| Mutual Gaze | 31,813 | 14.8% |
| Sudden Gaze Shift | 27,869 | 12.9% |

---

## Confidence Threshold Impact

| Threshold | Total Events | Retention % |
|-----------|-------------:|------------:|
| All (≥0.0) | 429,663 | 100% |
| ≥0.6 | 409,783 | 95.4% |
| ≥0.7 | 353,636 | 82.3% |
| ≥0.8 | 285,176 | 66.4% |
| **≥0.9** | **215,378** | **50.1%** |

> **Note**: Using conf ≥ 0.9 retains ~50% of events with higher precision for downstream QA generation.

---

## Event JSON Format

Each video produces a `*_sam3rf_gaze_events.json` file:

```json
{
  "video_path": "/path/to/video.mp4",
  "video_fps": 23.98,
  "sample_fps": 2.0,
  "num_events": 2,
  "event_counts": {
    "sudden_gaze_shift": 1,
    "joint_attention": 1
  },
  "events": [
    {
      "event_id": 0,
      "event_type": "sudden_gaze_shift",
      "start_time": 6.0,
      "end_time": 6.5,
      "start_frame": 143,
      "end_frame": 155,
      "confidence": 0.83,
      "persons_involved": [1],
      "details": {
        "max_velocity": 1.16,
        "duration": 0.5,
        "num_frames": 2
      }
    }
  ]
}
```

### Field Descriptions

| Field | Type | Description |
|-------|------|-------------|
| `event_id` | int | Unique ID (0-999: shift, 1000-1999: joint_att, etc.) |
| `event_type` | str | One of 5 taxonomy types |
| `start_time` / `end_time` | float | Event timestamps in seconds |
| `start_frame` / `end_frame` | int | Original video frame indices |
| `confidence` | float | Detection confidence (0.0-1.0) |
| `persons_involved` | list[int] | Person IDs participating in the event |
| `details` | dict | Event-specific metrics (varies by type) |

### Event-Specific Details

| Event Type | Detail Fields |
|------------|---------------|
| `sudden_gaze_shift` | `max_velocity`, `duration`, `num_frames` |
| `joint_attention` | `max_convergence`, `mean_convergence`, `attention_center`, `num_persons` |
| `gaze_following` | `leader_id`, `follower_id`, `delay_seconds` |
| `attention_capture` | `capture_point`, `num_persons`, `max_velocity` |
| `mutual_gaze` | `person_pair`, `duration`, `num_frames` |
