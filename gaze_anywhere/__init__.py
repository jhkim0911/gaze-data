"""
GazeAnywhere - Gaze prediction model for data annotation

Usage:
    from gaze_anywhere import GazePredictor

    predictor = GazePredictor(checkpoint_path="path/to/gazeanywhere.pth")
    
    # Single image prediction
    result = predictor.predict("image.jpg", text_prompt="person description")
    gaze_x, gaze_y = result["gaze_point"]  # normalized [0, 1]
    is_inframe = result["inout"]  # bool
    head_bbox = result["head_bbox"]  # (x1, y1, x2, y2) normalized
    
    # Video prediction
    results = predictor.predict_video("video.mp4", text_prompt="person description")
    for frame in results["frames"]:
        print(f"Frame {frame['frame_idx']}: gaze at {frame['gaze_point']}")
    
    # Save/load results
    save_results_json(results, "output.json")
    results = load_results_json("output.json")
"""

from .inference import (
    GazePredictor,
    visualize_gaze,
    visualize_video,
    save_results_json,
    load_results_json,
)

__all__ = [
    "GazePredictor",
    "visualize_gaze",
    "visualize_video",
    "save_results_json",
    "load_results_json",
]
