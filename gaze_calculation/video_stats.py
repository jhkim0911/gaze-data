#!/usr/bin/env python3
"""Get video duration statistics for all datasets."""
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

base_dir = '/projects/illinois/eng/cs/jrehg/users/arkimjh/gaze_social'
datasets = ['avsbench', 'embody3d', 'friendsmmc', 'social_gesture', 'social-iq', 'tvqa', 'werewolf']

def get_frame_count(vpath):
    try:
        result = subprocess.run(
            ['ffprobe', '-v', 'error', '-count_frames', '-select_streams', 'v:0',
             '-show_entries', 'stream=nb_read_frames', '-of', 'csv=p=0', vpath],
            capture_output=True, text=True, timeout=30
        )
        return int(result.stdout.strip())
    except:
        return 0

print(f"{'Dataset':<15} {'Videos':>8} {'Total Dur (hrs)':>15} {'Avg Dur (s)':>12}")
print("-" * 55)

grand_total_dur = 0
grand_total_vids = 0

for ds in datasets:
    bbox_dir = os.path.join(base_dir, ds, 'bbox_videos')
    if not os.path.exists(bbox_dir):
        print(f"{ds:<15} {'N/A':>8}")
        continue
    
    videos = [os.path.join(bbox_dir, f) for f in os.listdir(bbox_dir) if f.endswith('.mp4')]
    count = len(videos)
    
    # Use parallel processing
    total_frames = 0
    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = {executor.submit(get_frame_count, v): v for v in videos}
        for future in as_completed(futures):
            total_frames += future.result()
    
    total_dur = total_frames / 2.0  # 2fps
    hrs = total_dur / 3600
    avg = total_dur / count if count > 0 else 0
    print(f"{ds:<15} {count:>8} {hrs:>15.1f} {avg:>12.1f}")
    grand_total_dur += total_dur
    grand_total_vids += count

print("-" * 55)
print(f"{'TOTAL':<15} {grand_total_vids:>8} {grand_total_dur/3600:>15.1f} {grand_total_dur/grand_total_vids:>12.1f}")
