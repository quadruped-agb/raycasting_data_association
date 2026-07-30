"""
save_debug_frames.py

Generates debug files (image copy, cropped_map.ply, pixel_to_point.npy)
for SPECIFIC frame_ids, WITHOUT touching the checkpoint or the shared
color buffer at all. Safe to run any time, even after a full main
pipeline run has already completed and marked every frame as
processed -- this never reads or writes output/checkpoint_02/, so it
can't accidentally double-count or corrupt the already-finished
colorized map.

Reads dataset_index_glim_2.json and global_map_glim_2.ply from
output/ (unchanged), but writes its debug files into tests/ instead
of output/, to keep test artifacts separate from the main pipeline's
own output folder.

Usage (run from the project root, e.g. ~/Downloads/data_association):
    python3 tests/save_debug_frames.py 100 200 300 400 500
"""

import os
import sys
import json
import shutil
import time
import numpy as np
import open3d as o3d

import raycasting_glim as cgm

# Where THIS script's debug files get written -- separate from
# colorize_global_map's own OUTPUT_DIR (which stays "output" and is
# left completely alone, so the checkpoint/final colorized map are
# never touched).
DEBUG_OUTPUT_DIR = "tests"

# IMPORTANT: step1_crop_to_fov / step2_raycast (imported below) both
# read OUTPUT_DIR from colorize_global_map's OWN module namespace when
# save_debug=True -- they don't take it as a parameter. So to make
# their debug writes land in tests/ instead of output/, we override
# the attribute on the imported module itself, before calling them.
# This does NOT affect colorize_global_map.py's own CHECKPOINT_DIR
# (that was already computed once, at import time, from the ORIGINAL
# OUTPUT_DIR = "output" -- changing the attribute afterwards doesn't
# retroactively change CHECKPOINT_DIR's already-fixed string value).
cgm.OUTPUT_DIR = DEBUG_OUTPUT_DIR


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 tests/save_debug_frames.py <frame_id> [<frame_id> ...]")
        sys.exit(1)

    target_frame_ids = {int(x) for x in sys.argv[1:]}

    os.makedirs(DEBUG_OUTPUT_DIR, exist_ok=True)

    with open(cgm.DATASET_INDEX_PATH, "r") as f:
        dataset = json.load(f)

    entries_by_id = {e["frame_id"]: e for e in dataset}
    missing = target_frame_ids - set(entries_by_id.keys())
    if missing:
        print(f"WARNING: these frame_ids don't exist in {cgm.DATASET_INDEX_PATH}: {sorted(missing)}")

    print(f"Loading global map once: {cgm.GLOBAL_MAP_PATH}")
    global_cloud = o3d.io.read_point_cloud(cgm.GLOBAL_MAP_PATH)
    map_points = np.asarray(global_cloud.points)
    print(f"Loaded global map with {map_points.shape[0]} points\n")

    overall_start = time.time()

    for fid in sorted(target_frame_ids & set(entries_by_id.keys())):
        frame_start = time.time()
        entry = entries_by_id[fid]
        print(f"\n=== Frame {fid} (debug-only, writing to {DEBUG_OUTPUT_DIR}/, checkpoint untouched) ===")

        image_ext = os.path.splitext(entry["image_file"])[1]
        image_copy_path = os.path.join(DEBUG_OUTPUT_DIR, f"frame_{fid}_image{image_ext}")
        shutil.copy(entry["image_file"], image_copy_path)
        print(f"Saved copy of camera image: {image_copy_path}")

        lidar_points, global_indices, T_map_lidar = cgm.step1_crop_to_fov(
            entry, fid, map_points, save_debug=True
        )
        cgm.step2_raycast(lidar_points, global_indices, fid, save_debug=True)

        frame_elapsed = time.time() - frame_start
        print(f"[timing] Frame {fid} took {frame_elapsed:.1f}s")

    total_elapsed = time.time() - overall_start
    print(f"\nDone. All debug files written to {DEBUG_OUTPUT_DIR}/.")
    print(f"[timing] Total: {total_elapsed:.1f}s for {len(target_frame_ids)} frames "
          f"({total_elapsed / len(target_frame_ids):.1f}s average per frame)")
    print("The main colorized map / checkpoint in output/ were NOT touched by this script.")


if __name__ == "__main__":
    main()