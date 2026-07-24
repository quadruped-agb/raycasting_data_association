"""
  1. CROP    - crop the global map to this frame's camera FOV
               (same method as crop_fov.py), output in LIDAR-local frame.
               *** Also keeps the GLOBAL INDEX of every surviving point,
               so we can always trace a point back to its row in the
               one, single global map array. ***
  2. RAYCAST - two-stage:
       a) fast vectorized projection to find which pixels have ANY
          candidate coverage at all
       b) raycasting (pixel -> ray -> nearest point within
          max_perp_dist), but ONLY for that narrowed candidate set.
       Every match now carries the point's GLOBAL index, not just its
       index within this frame's cropped subset.
  3. COLORIZE - sample RGB from the full original image at each
       matched pixel and write it directly into a GLOBAL color buffer,
       indexed by that point's global index. This buffer starts as a
       GRAYSCALE version of global_map_glim.ply's own (height-based)
       colors -- converted via standard luminance weighting -- and is
       shared across every frame, so there is no "merge all frames"
       step and no duplicate points from different frames -- each
       global map point either gets a real camera color or it doesn't,
       and it's always the SAME point (same row) that gets updated no
       matter which frame colored it.

       POINTS WITH NO CAMERA MATCH stay GRAYSCALE (their original
       height color desaturated to gray), so the full map's shape
       stays visible while making it unambiguous which points are
       real photographic color vs which aren't -- a gray point can
       never be mistaken for a genuine but dim camera color, since
       only real camera hits ever have unequal R/G/B channels. The
       output has the exact same N points as the input map -- nothing
       is dropped.

       CONFLICT HANDLING: if a global point has already been colored
       by an earlier frame, and a later frame proposes a different
       color for that SAME point, we only accept it if the new color
       is within COLOR_MATCH_THRESHOLD of the existing one (running
       average in that case). If the new color differs by more than
       the threshold, it's treated as a bad observation (occlusion,
       reflection, projection error, etc.) and is rejected -- the
       existing color is kept as-is.

IMPORTANT NOTE ON PER-FRAME DEBUG FILES:
       EVERY frame in the dataset is always cropped, raycasted, and
       colorized into the ONE shared global color buffer -- that part
       never changes based on frame count. What's limited is only the
       optional per-frame DEBUG artifacts (the copied camera image and
       the frame_<id>_cropped_map.ply), which are written to disk for
       AT MOST the first MAX_DEBUG_FRAMES frames (see below), purely so
       a human has a handful of example frames to inspect. Frames
       beyond that limit still fully contribute their color data to
       global_map_glim_colorized.ply -- they just don't get their own
       per-frame files saved.

Output:
    output/frame_<id>_cropped_map.ply         (debug, first MAX_DEBUG_FRAMES frames only, from step 1, lidar-local frame)
    output/frame_<id>_pixel_to_point.npy      (debug, first MAX_DEBUG_FRAMES frames only, from step 2, full-image (u,v) keys)
    output/frame_<id>_image.*                 (debug, first MAX_DEBUG_FRAMES frames only, copy of the source camera image)
    output/global_map_glim_colorized.ply      (FINAL - same N points as global_map_glim.ply,
                                                camera RGB where matched, grayscale
                                                everywhere else. Input file is
                                                NOT modified. Includes ALL frames,
                                                regardless of the debug-file limit above.)
    output/checkpoint/                        (small resumable state: global_colors.npy,
                                                colored_mask.npy, color_counts.npy,
                                                processed_frames.json -- lets a later run
                                                resume instead of starting over)
"""

import os
import json
import numpy as np
import open3d as o3d
import cv2
from scipy.spatial import cKDTree
import time
import shutil
# FRAME_ID = 100

# Calibration (same as crop_fov.py / raycasting.py)
#
# NOTE: T_baselink_lidar has been removed. GLIM's poses (from
# traj_lidar.txt, via dataset_index_glim.json) are already map->lidar
# directly - GLIM has no base_link concept at all (its own sensor
# config only defines lidar/imu/camera extrinsics). Composing through
# a lidar->baselink calibration here would double-transform every
# point, same issue as in the original global_map.py.

# Maps a point expressed in CAMERA frame into LIDAR frame:
#   p_lidar = R_cam_lidar @ p_cam + t_cam_lidar
T_cam_lidar = np.array([
    [0.906, 0.000, -0.423, 0.142],
    [0.000, 1.000,  0.000, 0.000],
    [0.423, 0.000,  0.906, 0.005],
    [0.000, 0.000,  0.000, 1.000],
])
R_cam_lidar = T_cam_lidar[:3, :3]
t_cam_lidar = T_cam_lidar[:3, 3]
R_lidar_cam = R_cam_lidar.T  # inverse rotation: lidar -> camera

K = np.array([
    [1219.92, 0.0,     960.0],
    [0.0,     1219.92, 540.0],
    [0.0,     0.0,     1.0],
])

IMG_W, IMG_H = 1920, 1080
MAX_RANGE = 70.0     # metres, Livox Mid360 spec
MIN_DEPTH = 0.05     # metres, ignore points at/behind the camera
MAX_PERP_DIST = 0.15 # metres, raycasting tolerance

# NEW: how close (in normalized 0-1 RGB space, euclidean distance)
# a newly proposed color must be to a point's EXISTING color for it
# to be accepted as "the same surface, different frame". Tune this:
# smaller = stricter (more conflicts rejected), larger = looser.
COLOR_MATCH_THRESHOLD = 0.12

# Master on/off switch for per-frame DEBUG files (the image copy
# frame_<id>_image.* and the cropped map frame_<id>_cropped_map.ply
# and the frame_<id>_pixel_to_point.npy). These exist purely so a
# human can inspect a specific frame later; the actual colorization
# only ever uses the in-memory lidar_points / pixel_to_point / the
# ORIGINAL image path (entry["image_file"]), never these saved copies.
# Set to False to skip writing them entirely and save disk space --
# global_map_glim_colorized.ply comes out byte-for-byte identical
# either way. Set to True (default) to write them, but only for the
# first MAX_DEBUG_FRAMES frames -- see below.
SAVE_PER_FRAME_DEBUG_FILES = True

MAX_DEBUG_FRAMES = 20

# paths
DATASET_INDEX_PATH = "output/dataset_index_glim_2.json"
GLOBAL_MAP_PATH = "output/global_map_glim_2.ply"
OUTPUT_DIR = "output"
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoint")

# NEW: save a small checkpoint (global_colors/colored_mask/color_counts
# + list of processed frame ids) every N frames, so if the run stops
# for any reason (disk full, crash, manual interrupt), the NEXT run
# automatically resumes from the first un-processed frame instead of
# starting over or losing progress. This is independent of
# SAVE_PER_FRAME_DEBUG_FILES -- the checkpoint is a handful of
# map-sized arrays, not a per-frame file, so it stays small regardless
# of how many frames have been processed.
CHECKPOINT_EVERY_N_FRAMES = 20


# This converts a quaternion (the 4-number rotation format /tf publishes)
# into a 3x3 rotation matrix,
# since everywhere else in code (like R_cam_lidar) uses rotation matrices, not quaternions
def quaternion_to_matrix(x, y, z, w):
    return np.array([
        [1 - 2*(y**2 + z**2), 2*(x*y - z*w),       2*(x*z + y*w)],
        [2*(x*y + z*w),       1 - 2*(x**2 + z**2), 2*(y*z - x*w)],
        [2*(x*z - y*w),       2*(y*z + x*w),       1 - 2*(x**2 + y**2)],
    ])


# Takes a translation (x,y,z) and a quaternion (x,y,z,w)
# stores and combines them into one 4x4 matrix
def build_transform_matrix(translation, quaternion_xyzw):
    tx, ty, tz = translation
    x, y, z, w = quaternion_xyzw
    T = np.eye(4)
    T[:3, :3] = quaternion_to_matrix(x, y, z, w)
    T[:3, 3] = [tx, ty, tz]
    return T


def transform_points(points_local, T):
    n = points_local.shape[0]
    points_h = np.hstack([points_local, np.ones((n, 1))])
    return (T @ points_h.T).T[:, :3]


# transform a pixel (u,v) into a ray vector 3D
def pixel_to_ray_camera_frame(u, v):
    """
    Converts a pixel (u, v) into a normalized 3D ray direction in the
    CAMERA's own coordinate frame (origin at camera center, ray pointing
    into the scene along +Z by convention).
    """
    K_inv = np.linalg.inv(K)
    pixel_h = np.array([u, v, 1.0])  # homogeneous coordinates require a 3rd coordinate
    ray_dir_cam = K_inv @ pixel_h  # direction in camera frame
    ray_dir_cam = ray_dir_cam / np.linalg.norm(ray_dir_cam)  # normalize direction (only direction matters)
    return ray_dir_cam


# transforms from camera frame to lidar frame
def ray_to_lidar_frame(ray_dir_cam):
    """
    A ray has origin and direction.
    Transforms the camera-frame ray (origin + direction) into the LiDAR frame.
    Origin: camera center (0,0,0 in cam frame)
    Direction: rotate only, no translation (directions aren't affected by
    translation)
    """
    ray_origin_lidar = t_cam_lidar.copy()  # camera origin expressed in lidar frame
    ray_dir_lidar = R_cam_lidar @ ray_dir_cam  # rotate direction only, no translation
    ray_dir_lidar = ray_dir_lidar / np.linalg.norm(ray_dir_lidar)
    return ray_origin_lidar, ray_dir_lidar


def find_nearest_point_along_ray(points, ray_origin, ray_dir,
                                   max_perp_dist=0.15, min_depth=0.05):
    """
    Finds the LiDAR point that is the closest "first hit" along the ray.
    (Kept for reference / fallback use - the fast path below uses
    find_nearest_point_near_anchor instead.)
    """
    vecs = points - ray_origin
    depth = vecs @ ray_dir

    valid = depth > min_depth
    if not np.any(valid):
        return None, None, None, None

    proj_points = ray_origin + np.outer(depth, ray_dir)
    perp_dist = np.linalg.norm(points - proj_points, axis=1)
    candidate_mask = valid & (perp_dist < max_perp_dist)
    if not np.any(candidate_mask):
        return None, None, None, None
    candidate_idx = np.where(candidate_mask)[0]
    best_idx = candidate_idx[np.argmin(depth[candidate_idx])]

    return points[best_idx], depth[best_idx], perp_dist[best_idx], best_idx


def find_nearest_point_near_anchor(tree, points, ray_origin, ray_dir, anchor,
                                    max_perp_dist=MAX_PERP_DIST, min_depth=MIN_DEPTH):
    """
    Same math as find_nearest_point_along_ray, but only checked
    against points close to a known anchor position instead of the
    full point array. anchor is the 3D point whose projection
    generated this pixel, so the true nearest point is almost always
    within a small radius of it.

    NOTE: returns an index LOCAL to `points` (this frame's cropped,
    lidar-local subset). The caller is responsible for mapping that
    back to the point's GLOBAL index via `global_indices`.
    """
    idxs = tree.query_ball_point(anchor, r=max_perp_dist * 3)
    if not idxs:
        return None, None, None, None
    idxs = np.array(idxs)
    candidates = points[idxs]
    vecs = candidates - ray_origin
    depth = vecs @ ray_dir
    valid = depth > min_depth
    if not np.any(valid):
        return None, None, None, None

    proj_points = ray_origin + np.outer(depth, ray_dir)
    perp_dist = np.linalg.norm(candidates - proj_points, axis=1)

    mask = valid & (perp_dist < max_perp_dist)
    if not np.any(mask):
        return None, None, None, None

    local_best = np.where(mask)[0]
    best = local_best[np.argmin(depth[local_best])]
    global_idx_in_subset = idxs[best]  # still local to `points`, despite the name
    return points[global_idx_in_subset], depth[best], perp_dist[best], global_idx_in_subset


# STEP 1: crop the global map to this frame's camera FOV
# Result is in LIDAR-LOCAL frame (that's what raycasting needs), but
# now we ALSO return `global_indices`: for every surviving point,
# its row index in the original, full-size `map_points` array. This
# is what lets step 3 write colors back onto the ONE global map
# instead of a per-frame copy.
#
# `save_debug` controls ONLY whether frame_<id>_cropped_map.ply gets
# written for THIS frame -- it never affects cropping/raycasting
# itself, so every frame is processed identically either way.
def step1_crop_to_fov(entry, fid, map_points, save_debug=False):
    pose = entry["pose"]

    T_map_lidar = build_transform_matrix(pose["translation"], pose["quaternion"])
    T_map_cam = T_map_lidar @ T_cam_lidar
    T_cam_map = np.linalg.inv(T_map_cam)

    points_cam = transform_points(map_points, T_cam_map)
    depth = points_cam[:, 2]
    in_range = (depth > MIN_DEPTH) & (depth < MAX_RANGE)

    uvw = (K @ points_cam.T).T
    safe_depth = np.where(in_range, uvw[:, 2], 1.0)
    u = uvw[:, 0] / safe_depth
    v = uvw[:, 1] / safe_depth

    in_fov = in_range & (u >= 0) & (u < IMG_W) & (v >= 0) & (v < IMG_H)
    print(f"[crop] Points inside FOV : {in_fov.sum()} / {map_points.shape[0]}")

    # indices into the FULL global map, for every point that survives
    # the crop -- this is the thread that keeps us tied to "one point,
    # one row, forever" all the way through raycasting and colorizing.
    global_indices = np.where(in_fov)[0]

    cropped_points_map = map_points[in_fov]

    T_lidar_map = np.linalg.inv(T_map_lidar)
    cropped_points_lidar = transform_points(cropped_points_map, T_lidar_map)

    if save_debug:
        cropped_cloud = o3d.geometry.PointCloud()
        cropped_cloud.points = o3d.utility.Vector3dVector(cropped_points_lidar)
        cropped_map_path = os.path.join(OUTPUT_DIR, f"frame_{fid}_cropped_map.ply")
        o3d.io.write_point_cloud(cropped_map_path, cropped_cloud)
        print(f"[crop] Saved: {cropped_map_path}")

    return cropped_points_lidar, global_indices, T_map_lidar


# STEP 2: raycasting, two-stage, ALL in full-image (u,v) coordinates.
# `global_indices[i]` gives the row in the full global map that
# `lidar_points[i]` corresponds to, so every match we record here
# carries a "global_index" alongside the old lidar-local one.
#
# `save_debug` controls ONLY whether frame_<id>_pixel_to_point.npy
# gets written for THIS frame -- raycasting itself is unaffected.
def step2_raycast(lidar_points, global_indices, fid, save_debug=False):
    points_cam = (R_lidar_cam @ (lidar_points - t_cam_lidar).T).T
    depth = points_cam[:, 2]
    valid_depth = depth > MIN_DEPTH

    uvw = (K @ points_cam.T).T
    safe_depth = np.where(valid_depth, depth, 1.0)
    u_fast = np.round(uvw[:, 0] / safe_depth).astype(np.int64)
    v_fast = np.round(uvw[:, 1] / safe_depth).astype(np.int64)

    in_bounds = (u_fast >= 0) & (u_fast < IMG_W) & (v_fast >= 0) & (v_fast < IMG_H)
    valid = valid_depth & in_bounds

    if not np.any(valid):
        print("[raycast] No candidate pixels found at all -- check geometry/extrinsics.")
        return {}

    MARGIN = 2
    u_valid = u_fast[valid]
    v_valid = v_fast[valid]
    anchors_valid = lidar_points[valid]

    candidate_pixels = {}
    for uu, vv, anchor in zip(u_valid, v_valid, anchors_valid):
        for du in range(-MARGIN, MARGIN + 1):
            for dv in range(-MARGIN, MARGIN + 1):
                pu, pv = uu + du, vv + dv
                if 0 <= pu < IMG_W and 0 <= pv < IMG_H:
                    key = (int(pu), int(pv))
                    if key not in candidate_pixels:
                        candidate_pixels[key] = anchor

    print(f"[raycast] Fast pass found {valid.sum()} point projections -> "
          f"{len(candidate_pixels)} candidate pixels to actually raycast")

    tree = cKDTree(lidar_points)

    pixel_to_point = {}
    total = len(candidate_pixels)
    done = 0
    raycast_start = time.time()
    for (u, v), anchor in candidate_pixels.items():
        ray_dir_cam = pixel_to_ray_camera_frame(u, v)
        ray_origin_lidar, ray_dir_lidar = ray_to_lidar_frame(ray_dir_cam)

        point, pt_depth, perp_dist, local_idx = find_nearest_point_near_anchor(
            tree, lidar_points, ray_origin_lidar, ray_dir_lidar, anchor
        )

        if point is not None:
            pixel_to_point[(u, v)] = {
                "global_index": int(global_indices[local_idx]),  # <-- NEW: row in the full global map
                "point": point,
                "depth": float(pt_depth),
                "perp_dist": float(perp_dist),
            }

        done += 1
        if done % 100000 == 0:
            elapsed = time.time() - raycast_start
            print(f"[raycast]   {done}/{total} candidate pixels raycasted... ({elapsed:.1f}s elapsed)")

    print(f"[raycast] Matched pixels (raycasting): {len(pixel_to_point)} / {total}")

    if save_debug:
        pixel_to_point_path = os.path.join(OUTPUT_DIR, f"frame_{fid}_pixel_to_point.npy")
        np.save(pixel_to_point_path, pixel_to_point, allow_pickle=True)
        print(f"[raycast] Saved: {pixel_to_point_path}")

    return pixel_to_point


# STEP 3 (REPLACES old per-frame step3_colorize): sample RGB from the
# full original image at each matched pixel, and write it straight
# into the GLOBAL color buffer at that point's global index.
#
# global_colors : (N, 3) float array, shared across all frames.
#                 Starts as a GRAYSCALE version of the original height-based colors
#                 from global_map_glim.ply -- so any row this function
#                 never touches keeps a neutral gray, never a real color.
# colored_mask   : (N,)   bool array, True once a CAMERA (not gray)
#                  color has been written to that point. This is only
#                  used internally to decide "first camera hit -> just
#                  set it" vs. "repeat camera hit -> apply threshold",
#                  it does NOT control what gets written to the output
#                  file -- every point is written out regardless.
# color_counts   : (N,)   int array, how many camera frames have
#                  contributed to a point's color (for running-average
#                  blending)
#
# Conflict rule: if colored_mask[idx] is already True and the new
# color differs from the existing one by more than
# COLOR_MATCH_THRESHOLD (euclidean distance in normalized RGB), the
# new observation is REJECTED and the existing color is left
# untouched. If it's within the threshold, it's treated as "the same
# surface seen again" and blended in via a running average.
#
# NOTE: this function ALWAYS runs, for every single frame -- it is
# never gated by save_debug/MAX_DEBUG_FRAMES. The debug-file limit
# only ever affects optional files written to disk for human
# inspection; it never skips a frame's actual contribution to the
# shared global color buffer.
def update_global_colors(pixel_to_point, image_file, global_colors, colored_mask, color_counts):
    image = cv2.imread(image_file)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_file}")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    new_count = 0
    updated_count = 0
    conflict_count = 0

    for (u, v), info in pixel_to_point.items():
        idx = info["global_index"]
        new_color = image_rgb[v, u, :].astype(np.float64) / 255.0

        if not colored_mask[idx]:
            # first time this global point has ever been colored
            global_colors[idx] = new_color
            colored_mask[idx] = True
            color_counts[idx] = 1
            new_count += 1
        else:
            existing_color = global_colors[idx]
            diff = np.linalg.norm(new_color - existing_color)

            if diff <= COLOR_MATCH_THRESHOLD:
                # close enough to the existing color -> accept,
                # blend via running average so no single frame
                # dominates
                n = color_counts[idx]
                global_colors[idx] = (existing_color * n + new_color) / (n + 1)
                color_counts[idx] = n + 1
                updated_count += 1
            else:
                # too different -> likely a bad match (occlusion,
                # reflection, wrong projection, etc.) -- reject and
                # keep the existing color as-is
                conflict_count += 1

    return new_count, updated_count, conflict_count


def load_checkpoint(n_points, map_colors_original):
    """
    If output/checkpoint/ exists (from a previous run, or from
    reconstruct_from_saved_frames.py), load it and return
    (global_colors, colored_mask, color_counts, processed_frame_ids_set).
    Otherwise, return a fresh, empty state.
    """
    required = ["global_colors.npy", "colored_mask.npy", "color_counts.npy", "processed_frames.json"]
    if not all(os.path.exists(os.path.join(CHECKPOINT_DIR, f)) for f in required):
        print("No checkpoint found -- starting fresh.")
        global_colors = rgb_to_grayscale(map_colors_original)
        colored_mask = np.zeros(n_points, dtype=bool)
        color_counts = np.zeros(n_points, dtype=np.int32)
        return global_colors, colored_mask, color_counts, set()

    global_colors = np.load(os.path.join(CHECKPOINT_DIR, "global_colors.npy"))
    colored_mask = np.load(os.path.join(CHECKPOINT_DIR, "colored_mask.npy"))
    color_counts = np.load(os.path.join(CHECKPOINT_DIR, "color_counts.npy"))
    with open(os.path.join(CHECKPOINT_DIR, "processed_frames.json"), "r") as f:
        processed_frame_ids = set(json.load(f))

    print(f"Checkpoint found: {len(processed_frame_ids)} frames already processed "
          f"-- resuming instead of starting over.")
    return global_colors, colored_mask, color_counts, processed_frame_ids


def save_checkpoint(global_colors, colored_mask, color_counts, processed_frame_ids):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    np.save(os.path.join(CHECKPOINT_DIR, "global_colors.npy"), global_colors)
    np.save(os.path.join(CHECKPOINT_DIR, "colored_mask.npy"), colored_mask)
    np.save(os.path.join(CHECKPOINT_DIR, "color_counts.npy"), color_counts)
    with open(os.path.join(CHECKPOINT_DIR, "processed_frames.json"), "w") as f:
        json.dump(sorted(processed_frame_ids), f)


def rgb_to_grayscale(colors):
    """
    Converts (N,3) RGB colors in [0,1] to grayscale using standard
    luminance weighting, replicated across all 3 channels so it's
    still a valid (N,3) color array (just R==G==B per point). Used to
    turn the map's original height-based colormap into a neutral
    black-and-white backdrop, so the overall map shape/silhouette
    stays visible, while leaving the color channel free to show real
    camera color clearly wherever it exists -- no risk of a gray
    height-colored point being mistaken for a genuine (but dim/gray)
    camera color, since only actual camera hits ever have unequal
    R/G/B values.
    """
    luminance = 0.299 * colors[:, 0] + 0.587 * colors[:, 1] + 0.114 * colors[:, 2]
    return np.stack([luminance, luminance, luminance], axis=1)


def replay_orphaned_pixel_to_point_files(dataset, global_colors, colored_mask, color_counts, processed_frame_ids):
    """
    RECOVERY STEP,. If a previous run saved frame_<id>_pixel_to_point.npy
    files to disk (e.g. from back when SAVE_PER_FRAME_DEBUG_FILES was
    True, or before checkpointing existed) for frames that are NOT
    already marked processed in the checkpoint, replay them here --
    same update_global_colors() call the main loop uses -- so their
    work gets folded into global_colors/colored_mask/color_counts
    without re-doing any cropping/raycasting.

    Frames already in processed_frame_ids are skipped even if a
    leftover .npy exists for them, so nothing gets double-counted.
    After this runs once, it's safe to delete those .npy files -- the
    checkpoint saved right after this will have their contribution
    permanently folded in.
    """
    replayed = []
    for entry in dataset:
        fid = entry["frame_id"]
        if fid in processed_frame_ids:
            continue
        npy_path = os.path.join(OUTPUT_DIR, f"frame_{fid}_pixel_to_point.npy")
        if not os.path.exists(npy_path):
            continue
        pixel_to_point = np.load(npy_path, allow_pickle=True).item()
        update_global_colors(pixel_to_point, entry["image_file"], global_colors, colored_mask, color_counts)
        processed_frame_ids.add(fid)
        replayed.append(fid)

    if replayed:
        print(f"[recovery] Found and replayed {len(replayed)} leftover "
              f"pixel_to_point.npy file(s) from a previous run "
              f"(frame_id {min(replayed)}-{max(replayed)}) -- folded into "
              f"the color buffer. Safe to delete those .npy files after this run.")

    return replayed


def main():
    overall_start = time.time()

    with open(DATASET_INDEX_PATH, "r") as f:
        dataset = json.load(f)

    print(f"Loading global map once: {GLOBAL_MAP_PATH}")
    global_cloud = o3d.io.read_point_cloud(GLOBAL_MAP_PATH)
    map_points = np.asarray(global_cloud.points)
    map_colors_original = np.asarray(global_cloud.colors)  # height-based colors already in the file
    n_points = map_points.shape[0]
    print(f"Loaded global map with {n_points} points\n")

    # ONE persistent color buffer for the ENTIRE global map, shared
    # across every frame in the dataset. It starts as a GRAYSCALE
    # version of the map's own original (height-based) colors, so any
    # point that never gets a camera match keeps a neutral gray in
    # the final output -- visible as part of the map's shape, but
    # never confusable with a real camera color. Nothing is zeroed
    # out or dropped.
    # If a checkpoint from a previous (interrupted) run exists, this
    # loads it instead, so we resume rather than start over.
    global_colors, colored_mask, color_counts, processed_frame_ids = load_checkpoint(
        n_points, map_colors_original
    )

    # fold in any leftover per-frame .npy files from a previous run
    # before continuing -- no separate reconstruction script needed
    replayed = replay_orphaned_pixel_to_point_files(dataset, global_colors, colored_mask, color_counts, processed_frame_ids)
    if replayed:
        save_checkpoint(global_colors, colored_mask, color_counts, processed_frame_ids)

    total_new = 0
    total_updated = 0
    total_conflicts = 0
    frames_done_this_run = 0

    # `debug_index` walks dataset order (0, 1, 2, ...) independent of
    # how many frames actually get processed this run (some may be
    # skipped because they're already in the checkpoint). This is what
    # "first MAX_DEBUG_FRAMES frames" is measured against, so re-runs
    # against the same dataset always pick the same frames for debug
    # output, regardless of resume state.
    for debug_index, entry in enumerate(dataset):
        fid = entry["frame_id"]

        if fid in processed_frame_ids:
            # already folded into the checkpoint by a previous run (or
            # by reconstruct_from_saved_frames.py) -- skip re-doing it
            continue

        print(f"\n=== Frame {fid} ===")

        # Debug files are only written for the first MAX_DEBUG_FRAMES
        # frames (in dataset order). Every frame past that point is
        # still fully cropped, raycasted, and colorized below -- it
        # just doesn't get its own image/ply/npy saved to disk.
        save_debug = SAVE_PER_FRAME_DEBUG_FILES and (debug_index < MAX_DEBUG_FRAMES)

        if save_debug:
            image_ext = os.path.splitext(entry["image_file"])[1]
            image_copy_path = os.path.join(OUTPUT_DIR, f"frame_{fid}_image{image_ext}")
            shutil.copy(entry["image_file"], image_copy_path)
            print(f"Saved copy of camera image: {image_copy_path}")

        lidar_points, global_indices, T_map_lidar = step1_crop_to_fov(entry, fid, map_points, save_debug=save_debug)
        pixel_to_point = step2_raycast(lidar_points, global_indices, fid, save_debug=save_debug)

        new_c, upd_c, conf_c = update_global_colors(
            pixel_to_point, entry["image_file"], global_colors, colored_mask, color_counts
        )
        total_new += new_c
        total_updated += upd_c
        total_conflicts += conf_c
        processed_frame_ids.add(fid)
        frames_done_this_run += 1
        print(f"[colorize] frame {fid}: {new_c} newly colored, "
              f"{upd_c} blended into existing color, "
              f"{conf_c} rejected as conflicting (> {COLOR_MATCH_THRESHOLD} threshold)")

        if frames_done_this_run % CHECKPOINT_EVERY_N_FRAMES == 0:
            save_checkpoint(global_colors, colored_mask, color_counts, processed_frame_ids)
            partial_cloud = o3d.geometry.PointCloud()
            partial_cloud.points = o3d.utility.Vector3dVector(map_points)
            partial_cloud.colors = o3d.utility.Vector3dVector(global_colors)
            o3d.io.write_point_cloud(os.path.join(OUTPUT_DIR, "global_map_glim_colorized.ply"), partial_cloud)
            print(f"[checkpoint] Saved progress ({len(processed_frame_ids)} frames total processed so far) "
                  f"+ updated global_map_glim_colorized.ply")

    # final checkpoint save, in case the last save was more than
    # CHECKPOINT_EVERY_N_FRAMES frames ago
    save_checkpoint(global_colors, colored_mask, color_counts, processed_frame_ids)

    # Build the FINAL colorized global map: SAME N points as the
    # input file, always. Points with a camera match now carry real
    # RGB (in global_colors); points with no match still carry
    # whatever was already in global_colors for them, which is a
    # neutral grayscale version of the original height color (never
    # given a fake/desaturated "camera" color). Nothing is
    # dropped, and global_map_glim.ply itself is never modified --
    # this writes a brand new file. This includes the contribution of
    # EVERY frame processed above, not just the ones that got debug
    # files written.
    final_cloud = o3d.geometry.PointCloud()
    final_cloud.points = o3d.utility.Vector3dVector(map_points)
    final_cloud.colors = o3d.utility.Vector3dVector(global_colors)

    final_path = os.path.join(OUTPUT_DIR, "global_map_glim_colorized.ply")
    o3d.io.write_point_cloud(final_path, final_cloud)

    print(f"\nSaved final colorized global map: {final_path}")
    print(f"  Total points in file      : {n_points} (same as input map)")
    print(f"  Points with real camera color : {int(colored_mask.sum())}")
    print(f"  Points kept as grayscale (no camera match) : {int((~colored_mask).sum())}")
    print(f"Totals across all frames: {total_new} first-time colors, "
          f"{total_updated} accepted blends, {total_conflicts} rejected conflicts")
    print(f"Total batch time: {time.time() - overall_start:.1f}s for {len(dataset)} frames")


if __name__ == "__main__":
    main()