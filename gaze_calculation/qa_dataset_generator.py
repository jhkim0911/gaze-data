#!/usr/bin/env python3
"""
QA Dataset Generator for Video-LLM Training

Generates question-answer pairs from gaze and gesture annotations
for training video-LLMs to understand non-verbal social cues.

Taxonomy:
A. Gaze-based Prediction
B. Gesture-Gaze Cross-Reasoning  
C. Social Interaction Reasoning
D. Object-centric Social Attention

Usage:
    python qa_dataset_generator.py \
        --gaze_json /path/to/gaze.json \
        --events_json /path/to/events.json \
        --gestures_json /path/to/gestures.json \
        --output_json /path/to/qa_dataset.json
"""

import argparse
import json
import os
import random
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
import numpy as np


@dataclass
class QAPair:
    """A single question-answer pair."""
    question_id: str
    category: str  # A, B, C, D
    subcategory: str  # A1, A2, etc.
    task_type: str  # "classification", "regression", "binary"
    question: str
    answer: str
    # Metadata
    video_path: Optional[str] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    persons_involved: Optional[List[int]] = None


class QADatasetGenerator:
    """Generate QA pairs from gaze and gesture annotations.
    
    Supports enhanced gestures.json format (v2) with embedded temporal info:
    - event_type, start_time, end_time, start_frame, end_frame, persons_involved
    - No need to join with events.json for gesture-related QA
    """
    
    def __init__(
        self,
        gaze_data: Dict,
        events_data: Optional[Dict] = None,
        gestures_data: Optional[Dict] = None,
    ):
        self.gaze_data = gaze_data
        self.events_data = events_data or {}
        self.gestures_data = gestures_data or {}
        self.video_path = gaze_data.get("video_path", "")
        self.sample_fps = gaze_data.get("sample_fps", 2.0)
        self.video_fps = gaze_data.get("video_fps", 30.0)
        
        # Parse frame annotations
        self.frames = gaze_data.get("frames", gaze_data.get("frame_annotations", []))
        self.events = events_data.get("events", []) if events_data else []
        self.classifications = gestures_data.get("classifications", []) if gestures_data else []
        
        # Build event lookup by event_id (for fallback if gestures don't have temporal info)
        self.event_lookup = {e.get("event_id"): e for e in self.events}
        
        # Build classification lookup by event_id
        self.classification_lookup = {
            c.get("event_id"): c for c in self.classifications
        }
    
    def generate_all(self) -> List[QAPair]:
        """Generate all QA pairs."""
        qa_pairs = []
        
        # A. Gaze-based Prediction
        qa_pairs.extend(self._generate_gaze_prediction_qa())
        
        # B. Gesture-Gaze Cross-Reasoning
        qa_pairs.extend(self._generate_gesture_gaze_qa())
        
        # C. Social Interaction Reasoning
        qa_pairs.extend(self._generate_social_interaction_qa())
        
        # D. Object-centric Social Attention
        qa_pairs.extend(self._generate_object_attention_qa())
        
        return qa_pairs
    
    # ========================================================================
    # A. Gaze-based Prediction
    # ========================================================================
    
    def _generate_gaze_prediction_qa(self) -> List[QAPair]:
        """Generate gaze prediction QA pairs."""
        qa_pairs = []
        
        for i, frame in enumerate(self.frames):
            frame_idx = frame.get("frame_idx", frame.get("frame_index", i))
            timestamp = frame.get("timestamp", frame_idx / self.sample_fps)
            persons = frame.get("persons", [])
            
            for person in persons:
                person_id = person.get("person_id")
                gaze_point = person.get("gaze_point")
                
                if gaze_point is None:
                    continue
                
                # A2: Gaze shift direction (need previous frame)
                if i > 0:
                    prev_frame = self.frames[i - 1]
                    prev_persons = {p["person_id"]: p for p in prev_frame.get("persons", [])}
                    
                    if person_id in prev_persons:
                        prev_gaze = prev_persons[person_id].get("gaze_point")
                        if prev_gaze:
                            direction = self._compute_gaze_direction(prev_gaze, gaze_point)
                            if direction:
                                qa_pairs.append(QAPair(
                                    question_id=f"A2_{frame_idx}_{person_id}",
                                    category="A",
                                    subcategory="A2",
                                    task_type="classification",
                                    question=f"At {timestamp:.1f}s, which direction did P{person_id}'s gaze shift?",
                                    answer=direction,
                                    video_path=self.video_path,
                                    start_time=timestamp - 0.5,
                                    end_time=timestamp + 0.5,
                                    persons_involved=[person_id],
                                ))
                            
                            # A3: Gaze shift magnitude
                            magnitude = self._compute_gaze_magnitude(prev_gaze, gaze_point)
                            magnitude_label = self._quantize_magnitude(magnitude)
                            qa_pairs.append(QAPair(
                                question_id=f"A3_{frame_idx}_{person_id}",
                                category="A",
                                subcategory="A3",
                                task_type="classification",
                                question=f"At {timestamp:.1f}s, how much did P{person_id}'s gaze shift?",
                                answer=magnitude_label,
                                video_path=self.video_path,
                                start_time=timestamp - 0.5,
                                end_time=timestamp + 0.5,
                                persons_involved=[person_id],
                            ))
        
        return qa_pairs
    
    def _compute_gaze_direction(self, prev_gaze: List[float], curr_gaze: List[float]) -> Optional[str]:
        """Compute 8-direction gaze shift."""
        dx = curr_gaze[0] - prev_gaze[0]
        dy = curr_gaze[1] - prev_gaze[1]
        
        # Threshold for "no movement"
        if abs(dx) < 0.02 and abs(dy) < 0.02:
            return None
        
        # 8 directions
        angle = np.arctan2(dy, dx) * 180 / np.pi
        if -22.5 <= angle < 22.5:
            return "right"
        elif 22.5 <= angle < 67.5:
            return "down-right"
        elif 67.5 <= angle < 112.5:
            return "down"
        elif 112.5 <= angle < 157.5:
            return "down-left"
        elif angle >= 157.5 or angle < -157.5:
            return "left"
        elif -157.5 <= angle < -112.5:
            return "up-left"
        elif -112.5 <= angle < -67.5:
            return "up"
        else:
            return "up-right"
    
    def _compute_gaze_magnitude(self, prev_gaze: List[float], curr_gaze: List[float]) -> float:
        """Compute L2 distance of gaze shift."""
        dx = curr_gaze[0] - prev_gaze[0]
        dy = curr_gaze[1] - prev_gaze[1]
        return np.sqrt(dx**2 + dy**2)
    
    def _quantize_magnitude(self, magnitude: float) -> str:
        """Quantize magnitude to small/medium/large."""
        if magnitude < 0.1:
            return "small"
        elif magnitude < 0.3:
            return "medium"
        else:
            return "large"
    
    # ========================================================================
    # B. Gesture-Gaze Cross-Reasoning
    # ========================================================================
    
    def _generate_gesture_gaze_qa(self) -> List[QAPair]:
        """Generate gesture-gaze cross-reasoning QA pairs.
        
        Uses enhanced classifications format (v2) with embedded temporal info.
        Falls back to events.json lookup for legacy format.
        """
        qa_pairs = []
        
        for classification in self.classifications:
            event_id = classification.get("event_id")
            
            # Get temporal info from classification first (enhanced format v2)
            # Fallback to event lookup for legacy format
            start_time = classification.get("start_time")
            end_time = classification.get("end_time")
            persons_involved = classification.get("persons_involved", [])
            
            if start_time is None or end_time is None:
                # Legacy format: need to join with events.json
                event = self.event_lookup.get(event_id, {})
                start_time = event.get("start_time", 0)
                end_time = event.get("end_time", 0)
                persons_involved = event.get("persons_involved", [])
            
            # B1: Iterate over deictic_gestures for gesture-specific QA
            deictic_gestures = classification.get("deictic_gestures", [])
            for gesture in deictic_gestures:
                gesture_type = gesture.get("gesture_type")
                initiator_id = gesture.get("initiator_id")
                target_type = gesture.get("target_type")
                target_person_id = gesture.get("target_person_id")
                target_desc = gesture.get("target_description")
                
                # B1: Who initiated the gesture?
                if initiator_id is not None:
                    qa_pairs.append(QAPair(
                        question_id=f"B1_{event_id}_{initiator_id}",
                        category="B",
                        subcategory="B1",
                        task_type="classification",
                        question=f"Between {start_time:.1f}s and {end_time:.1f}s, who initiated the {gesture_type} gesture?",
                        answer=f"P{initiator_id}",
                        video_path=self.video_path,
                        start_time=start_time,
                        end_time=end_time,
                        persons_involved=persons_involved,
                    ))
                
                # B2: Gesture target identification
                if target_type == "person" and target_person_id is not None:
                    qa_pairs.append(QAPair(
                        question_id=f"B2_{event_id}_{initiator_id}",
                        category="B",
                        subcategory="B2",
                        task_type="classification",
                        question=f"Between {start_time:.1f}s and {end_time:.1f}s, who is P{initiator_id}'s {gesture_type} directed at?",
                        answer=f"P{target_person_id}",
                        video_path=self.video_path,
                        start_time=start_time,
                        end_time=end_time,
                        persons_involved=persons_involved,
                    ))
                elif target_type in ["object", "location"] and target_desc:
                    qa_pairs.append(QAPair(
                        question_id=f"B2_{event_id}_{initiator_id}",
                        category="B",
                        subcategory="B2",
                        task_type="classification",
                        question=f"Between {start_time:.1f}s and {end_time:.1f}s, what is P{initiator_id}'s {gesture_type} directed at?",
                        answer=target_desc,
                        video_path=self.video_path,
                        start_time=start_time,
                        end_time=end_time,
                        persons_involved=persons_involved,
                    ))
            
            # B3: Gaze-gesture alignment (did gesture cause gaze shift?)
            caused_shift = classification.get("caused_gaze_shift", False)
            if deictic_gestures:  # Only ask if there were gestures detected
                qa_pairs.append(QAPair(
                    question_id=f"B3_{event_id}",
                    category="B",
                    subcategory="B3",
                    task_type="binary",
                    question=f"Between {start_time:.1f}s and {end_time:.1f}s, did any gesture cause others to shift their gaze?",
                    answer="yes" if caused_shift else "no",
                    video_path=self.video_path,
                    start_time=start_time,
                    end_time=end_time,
                    persons_involved=persons_involved,
                ))
        
        return qa_pairs
    
    # ========================================================================
    # C. Social Interaction Reasoning
    # ========================================================================
    
    def _generate_social_interaction_qa(self) -> List[QAPair]:
        """Generate social interaction reasoning QA pairs."""
        qa_pairs = []
        
        for event in self.events:
            event_id = event.get("event_id")
            event_type = event.get("event_type")
            start_time = event.get("start_time", 0)
            end_time = event.get("end_time", 0)
            persons = event.get("persons_involved", [])
            
            # Get classification if available
            classification = self.classification_lookup.get(event_id, {})
            
            # C1: Who is being looked at (for mutual_gaze, joint_attention)
            if event_type in ["mutual_gaze", "gaze_following"]:
                details = event.get("details", {})
                target_id = details.get("target_person_id") or details.get("followed_person_id")
                if target_id is not None:
                    qa_pairs.append(QAPair(
                        question_id=f"C1_{event_id}",
                        category="C",
                        subcategory="C1",
                        task_type="classification",
                        question=f"Between {start_time:.1f}s and {end_time:.1f}s, who is being looked at?",
                        answer=f"P{target_id}",
                        video_path=self.video_path,
                        start_time=start_time,
                        end_time=end_time,
                        persons_involved=persons,
                    ))
            
            # C2: Social response prediction (did someone respond with gaze shift?)
            responder_ids = classification.get("responder_ids", [])
            has_response = len(responder_ids) > 0
            qa_pairs.append(QAPair(
                question_id=f"C2_{event_id}",
                category="C",
                subcategory="C2",
                task_type="binary",
                question=f"Between {start_time:.1f}s and {end_time:.1f}s, did anyone respond with a gaze shift?",
                answer="yes" if has_response else "no",
                video_path=self.video_path,
                start_time=start_time,
                end_time=end_time,
                persons_involved=persons,
            ))
            
            # C3: Interaction pair detection
            if event_type == "mutual_gaze" and len(persons) >= 2:
                qa_pairs.append(QAPair(
                    question_id=f"C3_{event_id}",
                    category="C",
                    subcategory="C3",
                    task_type="classification",
                    question=f"Between {start_time:.1f}s and {end_time:.1f}s, which pair is engaged in mutual gaze?",
                    answer=f"P{persons[0]} and P{persons[1]}",
                    video_path=self.video_path,
                    start_time=start_time,
                    end_time=end_time,
                    persons_involved=persons,
                ))
        
        return qa_pairs
    
    # ========================================================================
    # D. Object-centric Social Attention
    # ========================================================================
    
    def _generate_object_attention_qa(self) -> List[QAPair]:
        """Generate object-centric social attention QA pairs.
        
        Uses enhanced classifications format (v2) with embedded temporal info.
        """
        qa_pairs = []
        
        for classification in self.classifications:
            event_id = classification.get("event_id")
            
            # Get temporal info from classification first (enhanced format v2)
            start_time = classification.get("start_time")
            end_time = classification.get("end_time")
            persons_involved = classification.get("persons_involved", [])
            
            if start_time is None or end_time is None:
                # Legacy format: need to join with events.json
                event = self.event_lookup.get(event_id, {})
                if not event:
                    continue
                start_time = event.get("start_time", 0)
                end_time = event.get("end_time", 0)
                persons_involved = event.get("persons_involved", [])
            
            # Iterate over deictic gestures to find object-targeted ones
            deictic_gestures = classification.get("deictic_gestures", [])
            for gesture in deictic_gestures:
                target_type = gesture.get("target_type")
                if target_type in ["object", "location"]:
                    target_desc = gesture.get("target_description")
                    if target_desc:
                        gesture_type = gesture.get("gesture_type")
                        initiator_id = gesture.get("initiator_id")
                        
                        # D1: Object of attention
                        qa_pairs.append(QAPair(
                            question_id=f"D1_{event_id}_{initiator_id}",
                            category="D",
                            subcategory="D1",
                            task_type="classification",
                            question=f"Between {start_time:.1f}s and {end_time:.1f}s, what {target_type} is P{initiator_id} {gesture_type} at?",
                            answer=target_desc,
                            video_path=self.video_path,
                            start_time=start_time,
                            end_time=end_time,
                            persons_involved=persons_involved,
                        ))
            
            # D2: Object-driven gaze shift (if any object-targeted gesture)
            object_gestures = [g for g in deictic_gestures if g.get("target_type") in ["object", "location"]]
            if object_gestures:
                caused_shift = classification.get("caused_gaze_shift", False)
                qa_pairs.append(QAPair(
                    question_id=f"D2_{event_id}",
                    category="D",
                    subcategory="D2",
                    task_type="binary",
                    question=f"Between {start_time:.1f}s and {end_time:.1f}s, did pointing to an object cause others to shift their gaze?",
                    answer="yes" if caused_shift else "no",
                    video_path=self.video_path,
                    start_time=start_time,
                    end_time=end_time,
                    persons_involved=persons_involved,
                ))
        
        return qa_pairs


def save_qa_dataset(qa_pairs: List[QAPair], output_path: str) -> None:
    """Save QA dataset to JSON."""
    # Count by category
    category_counts = {}
    for qa in qa_pairs:
        key = qa.subcategory
        category_counts[key] = category_counts.get(key, 0) + 1
    
    output = {
        "num_qa_pairs": len(qa_pairs),
        "category_counts": category_counts,
        "qa_pairs": [asdict(qa) for qa in qa_pairs],
    }
    
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"Saved {len(qa_pairs)} QA pairs to: {output_path}")
    print("Category breakdown:")
    for cat, count in sorted(category_counts.items()):
        print(f"  {cat}: {count}")


def main():
    parser = argparse.ArgumentParser(description="Generate QA dataset from gaze/gesture annotations")
    parser.add_argument("--gaze_json", type=str, required=True, help="Path to gaze annotation JSON")
    parser.add_argument("--events_json", type=str, default=None, help="Path to candidate events JSON")
    parser.add_argument("--gestures_json", type=str, default=None, help="Path to gesture classifications JSON")
    parser.add_argument("--output_json", type=str, default=None, help="Output path for QA dataset")
    
    args = parser.parse_args()
    
    if args.output_json is None:
        base = os.path.splitext(args.gaze_json)[0].replace("_gaze", "").replace("_sam3rf", "")
        args.output_json = f"{base}_qa_dataset.json"
    
    # Load data
    print(f"Loading gaze data from: {args.gaze_json}")
    with open(args.gaze_json, 'r') as f:
        gaze_data = json.load(f)
    
    events_data = None
    if args.events_json and os.path.exists(args.events_json):
        print(f"Loading events from: {args.events_json}")
        with open(args.events_json, 'r') as f:
            events_data = json.load(f)
    
    gestures_data = None
    if args.gestures_json and os.path.exists(args.gestures_json):
        print(f"Loading gestures from: {args.gestures_json}")
        with open(args.gestures_json, 'r') as f:
            gestures_data = json.load(f)
    
    # Generate QA pairs
    generator = QADatasetGenerator(gaze_data, events_data, gestures_data)
    qa_pairs = generator.generate_all()
    
    # Save
    save_qa_dataset(qa_pairs, args.output_json)


if __name__ == "__main__":
    main()
