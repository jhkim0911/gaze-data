#!/usr/bin/env python3
"""
Video Segmentation Script using PySceneDetect

Splits videos into scene-based segments using AdaptiveDetector with configurable
minimum (60s default) and maximum (180s default) segment lengths.

Usage:
    python segment_videos.py

Dependencies:
    pip install scenedetect[opencv]
"""

import os
import logging
from pathlib import Path
from typing import List, Tuple, Optional
from dataclasses import dataclass

from scenedetect import detect, AdaptiveDetector, open_video, split_video_ffmpeg
from scenedetect.frame_timecode import FrameTimecode


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class SegmentConfig:
    """Configuration for video segmentation."""
    min_segment_seconds: int = 60  # Minimum segment length in seconds
    max_segment_seconds: int = 180  # Maximum segment length in seconds (3 minutes)
    adaptive_threshold: float = 3.0  # AdaptiveDetector threshold
    min_content_val: float = 15.0  # Minimum content difference to detect a scene


def get_video_fps(video_path: str) -> float:
    """Get the FPS of a video file."""
    video = open_video(video_path)
    fps = video.frame_rate
    video.release()
    return fps


def enforce_max_segment_length(
    scenes: List[Tuple[FrameTimecode, FrameTimecode]], 
    max_seconds: float,
    fps: float
) -> List[Tuple[FrameTimecode, FrameTimecode]]:
    """
    Post-process scenes to enforce maximum segment length.
    Splits long scenes at max_seconds boundaries.
    
    Args:
        scenes: List of (start, end) FrameTimecode pairs
        max_seconds: Maximum segment length in seconds
        fps: Video frame rate
        
    Returns:
        List of scenes with max length enforced
    """
    if not scenes:
        return scenes
    
    processed_scenes = []
    max_frames = int(max_seconds * fps)
    
    for start, end in scenes:
        scene_frames = end.get_frames() - start.get_frames()
        
        if scene_frames <= max_frames:
            # Scene is within limits, keep as-is
            processed_scenes.append((start, end))
        else:
            # Split into chunks of max_frames
            current_start_frame = start.get_frames()
            end_frame = end.get_frames()
            
            while current_start_frame < end_frame:
                chunk_end_frame = min(current_start_frame + max_frames, end_frame)
                
                chunk_start = FrameTimecode(current_start_frame, fps=fps)
                chunk_end = FrameTimecode(chunk_end_frame, fps=fps)
                
                processed_scenes.append((chunk_start, chunk_end))
                current_start_frame = chunk_end_frame
    
    return processed_scenes


def segment_video(
    input_path: str,
    output_dir: str,
    config: SegmentConfig = None,
    show_progress: bool = True
) -> List[Tuple[FrameTimecode, FrameTimecode]]:
    """
    Detect scenes in video and split into segments.
    
    Args:
        input_path: Path to input video file
        output_dir: Directory to save output segments
        config: Segmentation configuration
        show_progress: Show progress bar
        
    Returns:
        List of (start, end) tuples for each segment
    """
    if config is None:
        config = SegmentConfig()
    
    input_path = str(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get video FPS
    fps = get_video_fps(input_path)
    logger.info(f"Processing: {input_path} (FPS: {fps:.2f})")
    
    # Calculate frame counts for min/max segment lengths
    min_scene_len_frames = int(config.min_segment_seconds * fps)
    
    # Detect scenes using AdaptiveDetector
    detector = AdaptiveDetector(
        adaptive_threshold=config.adaptive_threshold,
        min_scene_len=min_scene_len_frames,
        min_content_val=config.min_content_val
    )
    
    logger.info(f"Detecting scenes with min_scene_len={config.min_segment_seconds}s...")
    scenes = detect(input_path, detector, show_progress=show_progress)
    
    if not scenes:
        logger.warning(f"No scenes detected in {input_path}")
        # If no scenes detected, treat entire video as one scene
        video = open_video(input_path)
        total_frames = video.duration.get_frames()
        scenes = [(FrameTimecode(0, fps=fps), FrameTimecode(total_frames, fps=fps))]
        video.release()
    
    logger.info(f"Detected {len(scenes)} initial scenes")
    
    # Enforce maximum segment length
    scenes = enforce_max_segment_length(scenes, config.max_segment_seconds, fps)
    logger.info(f"After max length enforcement: {len(scenes)} segments")
    
    # Log segment information
    for i, (start, end) in enumerate(scenes):
        duration = (end.get_frames() - start.get_frames()) / fps
        logger.info(f"  Segment {i+1}: {start.get_timecode()} - {end.get_timecode()} ({duration:.1f}s)")
    
    # Split video using ffmpeg
    video_name = Path(input_path).stem
    output_template = f"{video_name}_segment_$SCENE_NUMBER.mp4"
    
    logger.info(f"Splitting video into {len(scenes)} segments...")
    split_video_ffmpeg(
        input_video_path=input_path,
        scene_list=scenes,
        output_dir=output_dir,
        output_file_template=output_template,
        show_progress=show_progress
    )
    
    logger.info(f"Segments saved to: {output_dir}")
    return scenes


def process_directory(
    input_dir: str,
    output_dir: str,
    config: SegmentConfig = None,
    video_extensions: Tuple[str, ...] = ('.mp4', '.avi', '.mkv', '.mov'),
    preserve_structure: bool = True
) -> dict:
    """
    Process all videos in a directory.
    
    Args:
        input_dir: Directory containing input videos
        output_dir: Directory to save output segments
        config: Segmentation configuration
        video_extensions: File extensions to process
        preserve_structure: If True, preserve subdirectory structure in output
        
    Returns:
        Dictionary mapping input paths to segment lists
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    
    if not input_dir.exists():
        raise ValueError(f"Input directory does not exist: {input_dir}")
    
    results = {}
    
    # Find all video files
    video_files = []
    for ext in video_extensions:
        video_files.extend(input_dir.rglob(f"*{ext}"))
    
    logger.info(f"Found {len(video_files)} videos in {input_dir}")
    
    for video_path in sorted(video_files):
        try:
            # Calculate output directory
            if preserve_structure:
                relative_path = video_path.relative_to(input_dir).parent
                video_output_dir = output_dir / relative_path
            else:
                video_output_dir = output_dir
            
            segments = segment_video(
                input_path=str(video_path),
                output_dir=str(video_output_dir),
                config=config
            )
            results[str(video_path)] = segments
            
        except Exception as e:
            logger.error(f"Error processing {video_path}: {e}")
            results[str(video_path)] = None
    
    return results


def main():
    """Main function to process TVQA and SocialGesture videos."""
    
    config = SegmentConfig(
        min_segment_seconds=60,   # 60 seconds minimum
        max_segment_seconds=180,  # 3 minutes maximum
    )
    
    # ==========================================================================
    # TVQA Videos
    # ==========================================================================
    tvqa_input_dir = "/projects/illinois/eng/cs/jrehg/users/arkimjh/gaze_social/tvqa/mp4_videos"
    tvqa_output_dir = "/projects/illinois/eng/cs/jrehg/users/arkimjh/gaze_social/tvqa/video_segments"
    
    if Path(tvqa_input_dir).exists():
        logger.info("=" * 60)
        logger.info("Processing TVQA videos...")
        logger.info("=" * 60)
        
        tvqa_results = process_directory(
            input_dir=tvqa_input_dir,
            output_dir=tvqa_output_dir,
            config=config,
            preserve_structure=True
        )
        
        success_count = sum(1 for v in tvqa_results.values() if v is not None)
        logger.info(f"TVQA: Processed {success_count}/{len(tvqa_results)} videos successfully")
    else:
        logger.warning(f"TVQA input directory not found: {tvqa_input_dir}")
    
    # ==========================================================================
    # SocialGesture Videos
    # ==========================================================================
    # NOTE: Update this path if the SocialGesture videos are in a different location
    socialgesture_input_dir = "/projects/illinois/eng/cs/jrehg/datasets-social/SocialGesture/socialgesture_5fps_videos"
    socialgesture_output_dir = "/projects/illinois/eng/cs/jrehg/users/arkimjh/gaze_social/social_gesture/video_segments"
    
    if Path(socialgesture_input_dir).exists():
        logger.info("=" * 60)
        logger.info("Processing SocialGesture videos...")
        logger.info("=" * 60)
        
        socialgesture_results = process_directory(
            input_dir=socialgesture_input_dir,
            output_dir=socialgesture_output_dir,
            config=config,
            preserve_structure=True
        )
        
        success_count = sum(1 for v in socialgesture_results.values() if v is not None)
        logger.info(f"SocialGesture: Processed {success_count}/{len(socialgesture_results)} videos successfully")
    else:
        logger.warning(f"SocialGesture input directory not found: {socialgesture_input_dir}")
        logger.warning("To process SocialGesture videos, update the path in main() or use process_directory() directly")


if __name__ == "__main__":
    main()
