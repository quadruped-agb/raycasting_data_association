"""
PIPELINE:
  0. VOXELIZE - merge tightly-clustered near-duplicate points into
       single representative points BEFORE any raycasting happens
       Each voxel's position is the
       average of every original point inside it. point_to_voxel
       maps every ORIGINAL point to its voxel, for broadcasting
       colors back at the end.
  1. CROP    - crop the VOXELIZED map to this frame's camera FOV,
       output in LIDAR-local frame. Keeps the VOXEL INDEX of every
       surviving voxel-point.
  2. RAYCAST (GPU) - same voxel-grid-accelerated raycasting as
       raycasting_gpu.py pipeline, embedded here, operating on voxel points
       instead of raw points -- "closest point wins" now picks among
       clean, de-duplicated targets. Returns "voxel_index" per match
  3. COLORIZE - sample RGB at each matched pixel, write into a GLOBAL
       VOXEL color buffer.
  4. BROADCAST - every voxel's final color gets copied onto ALL the
       original raw points belonging to that voxel. Points whose
       voxel never got a real color keep the original grayscale
       fallback. Output has the SAME N points as
       global_map_glim_2.ply, just far better
       coverage since a whole duplicate-cluster now shares one
       voxel's color instead of only whichever single point won
       before.

Output:
    output/frame_<id>_cropped_map.ply           
    output/frame_<id>_pixel_to_point.npy        
    output/frame_<id>_image.*                  
    output/global_map_glim_colorized_voxel_gpu.ply 
    output/checkpoint_voxel_gpu/                
"""

import os
import json
import numpy as np
import torch
import open3d as o3d
import cv2
import time
import shutil

# ---------------------------------------------------------------
# Calibration 
T_cam_lidar = np.array([
    [0.906, 0.000, -0.423, 0.142],
    [0.000, 1.000,  0.000, 0.000],
    [0.423, 0.000,  0.906, 0.005],
    [0.000, 0.000,  0.000, 1.000],
])
R_cam_lidar = T_cam_lidar[:3, :3]
t_cam_lidar = T_cam_lidar[:3, 3]
R_lidar_cam = R_cam_lidar.T

K = np.array([
    [1219.92, 0.0,     960.0],
    [0.0,     1219.92, 540.0],
    [0.0,     0.0,     1.0],
])
K_inv = np.linalg.inv(K)

IMG_W, IMG_H = 1920, 1080
MAX_RANGE = 70.0
MIN_DEPTH = 0.05

MAX_PERP_DIST = 0.15   # metres, raycasting tolerance -- value confirmed
                        # best-performing via testing with different thresholds
RAYCAST_MARGIN = 1      # best-performing margin selected
                  

COLOR_MATCH_THRESHOLD = 0.20  # loosened from 0.12 
                               # comment: 0.12 rejected ~2x more legitimate
                               # repeat observations than it accepted

VOXEL_SIZE = 0.03  # metres, for merging near-duplicate map points before raycasting

# GPU spatial-hash acceleration settings 
GPU_HASH_VOXEL_SIZE = 0.5
NEIGHBOR_RING = 1
ANCHOR_CHUNK_SIZE = 2048
NEIGHBOR_CAP = 512
_COORD_OFFSET = 1 << 20

SAVE_PER_FRAME_DEBUG_FILES = True
MAX_DEBUG_FRAMES = 20

DATASET_INDEX_PATH = "output/dataset_index_glim_2.json"
GLOBAL_MAP_PATH = "output/global_map_glim_2.ply"
OUTPUT_DIR = "output"
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoint_voxel_gpu")  # own dir -- never touches other checkpoints
CHECKPOINT_EVERY_N_FRAMES = 20

_device = "cuda" if torch.cuda.is_available() else "cpu"
if _device == "cpu":
    print("[voxel-gpu] WARNING: CUDA not available -- running on CPU tensors "
          "(still batched/parallel, just not on GPU cores).")

_K_inv_t = torch.tensor(K_inv, dtype=torch.float32, device=_device)
_R_cam_lidar_t = torch.tensor(R_cam_lidar, dtype=torch.float32, device=_device)
_t_cam_lidar_t = torch.tensor(t_cam_lidar, dtype=torch.float32, device=_device)

# Pose / transform helpers
def quaternion_to_matrix(x, y, z, w):
    return np.array([
        [1 - 2*(y**2 + z**2), 2*(x*y - z*w),       2*(x*z + y*w)],
        [2*(x*y + z*w),       1 - 2*(x**2 + z**2), 2*(y*z - x*w)],
        [2*(x*z - y*w),       2*(y*z + x*w),       1 - 2*(x**2 + y**2)],
    ])


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


# ---------------------------------------------------------------
# STEP 0: voxelize the map ONCE, up front, before any frame processing
# ---------------------------------------------------------------
def voxelize_map(map_points, voxel_size):
    print(f"Voxelizing {map_points.shape[0]} points at {voxel_size}m resolution...")
    voxel_coords = np.floor(map_points / voxel_size).astype(np.int64)

    unique_voxels, point_to_voxel, counts = np.unique(
        voxel_coords, axis=0, return_inverse=True, return_counts=True
    )
    num_voxels = unique_voxels.shape[0]

    voxel_positions = np.zeros((num_voxels, 3), dtype=np.float64)
    np.add.at(voxel_positions, point_to_voxel, map_points)
    voxel_positions /= counts[:, None]

    print(f"  -> {num_voxels} voxels "
          f"({map_points.shape[0] / num_voxels:.1f}x reduction, "
          f"avg {counts.mean():.1f} original points per voxel)")

    return voxel_positions, point_to_voxel


# ---------------------------------------------------------------
# STEP 1: crop the VOXELIZED map to this frame's camera FOV
# ---------------------------------------------------------------
def step1_crop_to_fov(entry, fid, voxel_positions, save_debug=False):
    pose = entry["pose"]

    T_map_lidar = build_transform_matrix(pose["translation"], pose["quaternion"])
    T_map_cam = T_map_lidar @ T_cam_lidar
    T_cam_map = np.linalg.inv(T_map_cam)

    points_cam = transform_points(voxel_positions, T_cam_map)
    depth = points_cam[:, 2]
    in_range = (depth > MIN_DEPTH) & (depth < MAX_RANGE)

    uvw = (K @ points_cam.T).T
    safe_depth = np.where(in_range, uvw[:, 2], 1.0)
    u = uvw[:, 0] / safe_depth
    v = uvw[:, 1] / safe_depth

    in_fov = in_range & (u >= 0) & (u < IMG_W) & (v >= 0) & (v < IMG_H)
    print(f"[crop] Voxels inside FOV : {in_fov.sum()} / {voxel_positions.shape[0]}")

    voxel_indices = np.where(in_fov)[0]

    cropped_voxels_map = voxel_positions[in_fov]
    T_lidar_map = np.linalg.inv(T_map_lidar)
    cropped_voxels_lidar = transform_points(cropped_voxels_map, T_lidar_map)

    if save_debug:
        cropped_cloud = o3d.geometry.PointCloud()
        cropped_cloud.points = o3d.utility.Vector3dVector(cropped_voxels_lidar)
        cropped_map_path = os.path.join(OUTPUT_DIR, f"frame_{fid}_cropped_map.ply")
        o3d.io.write_point_cloud(cropped_map_path, cropped_cloud)
        print(f"[crop] Saved: {cropped_map_path}")

    return cropped_voxels_lidar, voxel_indices, T_map_lidar


# ---------------------------------------------------------------
# GPU voxel-grid raycasting helpers (embedded, spatial-hash accelerator)
# ---------------------------------------------------------------
def _pack_voxel_keys(voxel_coords):
    shifted = voxel_coords + _COORD_OFFSET
    keys = (shifted[:, 0].astype(np.int64) << 42) \
         | (shifted[:, 1].astype(np.int64) << 21) \
         | (shifted[:, 2].astype(np.int64))
    return keys


def _build_voxel_grid(points, voxel_size):
    voxel_coords = np.floor(points / voxel_size).astype(np.int64)
    keys = _pack_voxel_keys(voxel_coords)
    order = np.argsort(keys, kind="stable")
    sorted_keys = keys[order]
    unique_keys, start_idx, counts = np.unique(
        sorted_keys, return_index=True, return_counts=True
    )
    return order, unique_keys, start_idx, counts


def _query_neighbors(anchor_voxel_coords, order, unique_keys, start_idx, counts):
    m = anchor_voxel_coords.shape[0]
    neighbor_lists = [[] for _ in range(m)]

    offsets = range(-NEIGHBOR_RING, NEIGHBOR_RING + 1)
    for dx in offsets:
        for dy in offsets:
            for dz in offsets:
                shifted_coords = anchor_voxel_coords + np.array([dx, dy, dz])
                neighbor_keys = _pack_voxel_keys(shifted_coords)
                pos = np.searchsorted(unique_keys, neighbor_keys)
                pos_clipped = np.clip(pos, 0, len(unique_keys) - 1)
                found = unique_keys[pos_clipped] == neighbor_keys
                for i in range(m):
                    if found[i]:
                        s = start_idx[pos_clipped[i]]
                        c = counts[pos_clipped[i]]
                        neighbor_lists[i].append(order[s:s + c])

    result = []
    for i in range(m):
        if neighbor_lists[i]:
            idxs = np.concatenate(neighbor_lists[i])
        else:
            idxs = np.empty((0,), dtype=np.int64)
        if len(idxs) > NEIGHBOR_CAP:
            idxs = idxs[:NEIGHBOR_CAP]
        result.append(idxs)
    return result


# ---------------------------------------------------------------
# STEP 2: GPU raycasting, operating on VOXEL points.
# `voxel_indices[i]` gives the row in the FULL voxel array that
# `lidar_points[i]` corresponds to. 
#This returns
# "voxel_index" 
# ---------------------------------------------------------------
def step2_raycast_gpu(lidar_points, voxel_indices, fid, save_debug=False):
    t0 = time.time()

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
        print("[raycast-gpu] No candidate pixels found at all -- check geometry/extrinsics.")
        return {}

    u_valid = u_fast[valid]
    v_valid = v_fast[valid]
    anchors_valid = lidar_points[valid]

    candidate_pixels = {}
    for uu, vv, anchor in zip(u_valid, v_valid, anchors_valid):
        for du in range(-RAYCAST_MARGIN, RAYCAST_MARGIN + 1):
            for dv in range(-RAYCAST_MARGIN, RAYCAST_MARGIN + 1):
                pu, pv = uu + du, vv + dv
                if 0 <= pu < IMG_W and 0 <= pv < IMG_H:
                    key = (int(pu), int(pv))
                    if key not in candidate_pixels:
                        candidate_pixels[key] = anchor

    print(f"[raycast-gpu] Fast pass found {valid.sum()} voxel projections -> "
          f"{len(candidate_pixels)} candidate pixels to actually raycast "
          f"(MARGIN={RAYCAST_MARGIN}, MAX_PERP_DIST={MAX_PERP_DIST})")

    keys = list(candidate_pixels.keys())
    us = np.array([k[0] for k in keys], dtype=np.float32)
    vs = np.array([k[1] for k in keys], dtype=np.float32)
    anchors = np.array([candidate_pixels[k] for k in keys])

    unique_anchors, inverse = np.unique(anchors, axis=0, return_inverse=True)
    inverse = inverse.reshape(-1)
    A = unique_anchors.shape[0]
    print(f"[raycast-gpu] {len(keys)} pixels -> {A} unique anchors "
          f"({len(keys) / max(A, 1):.1f}x dedup)")

    order, unique_keys, start_idx, counts = _build_voxel_grid(lidar_points, GPU_HASH_VOXEL_SIZE)
    anchor_voxel_coords = np.floor(unique_anchors / GPU_HASH_VOXEL_SIZE).astype(np.int64)

    lidar_points_t = torch.tensor(lidar_points, dtype=torch.float32, device=_device)

    rep_pixel_for_anchor = np.full(A, -1, dtype=np.int64)
    seen_anchor = np.zeros(A, dtype=bool)
    for pixel_i, anchor_i in enumerate(inverse):
        if not seen_anchor[anchor_i]:
            rep_pixel_for_anchor[anchor_i] = pixel_i
            seen_anchor[anchor_i] = True

    anchor_best_voxel_row = np.full(A, -1, dtype=np.int64)
    anchor_best_point = np.zeros((A, 3), dtype=np.float32)
    anchor_best_depth = np.zeros(A, dtype=np.float32)
    anchor_best_perp = np.zeros(A, dtype=np.float32)
    anchor_has_match = np.zeros(A, dtype=bool)

    total_chunks = (A + ANCHOR_CHUNK_SIZE - 1) // ANCHOR_CHUNK_SIZE
    chunk_start = time.time()

    for start in range(0, A, ANCHOR_CHUNK_SIZE):
        end = min(start + ANCHOR_CHUNK_SIZE, A)
        m = end - start

        chunk_voxel_coords = anchor_voxel_coords[start:end]

        neighbor_idx_lists = _query_neighbors(
            chunk_voxel_coords, order, unique_keys, start_idx, counts
        )
        max_k = max((len(x) for x in neighbor_idx_lists), default=0)
        if max_k == 0:
            continue

        padded_idx = np.zeros((m, max_k), dtype=np.int64)
        pad_mask = np.zeros((m, max_k), dtype=bool)
        for i, idxs in enumerate(neighbor_idx_lists):
            k = len(idxs)
            if k > 0:
                padded_idx[i, :k] = idxs
                pad_mask[i, :k] = True

        padded_idx_t = torch.tensor(padded_idx, dtype=torch.long, device=_device)
        pad_mask_t = torch.tensor(pad_mask, dtype=torch.bool, device=_device)

        neighbor_points_t = lidar_points_t[padded_idx_t]

        rep_pixel_idx = rep_pixel_for_anchor[start:end]

        u_chunk = torch.tensor(us[rep_pixel_idx], device=_device)
        v_chunk = torch.tensor(vs[rep_pixel_idx], device=_device)

        ones = torch.ones_like(u_chunk)
        pix_h = torch.stack([u_chunk, v_chunk, ones], dim=0)
        ray_dir_cam = _K_inv_t @ pix_h
        ray_dir_cam = ray_dir_cam / ray_dir_cam.norm(dim=0, keepdim=True)

        ray_dir_lidar = _R_cam_lidar_t @ ray_dir_cam
        ray_dir_lidar = ray_dir_lidar / ray_dir_lidar.norm(dim=0, keepdim=True)
        ray_dir_lidar = ray_dir_lidar.T
        ray_origin_lidar = _t_cam_lidar_t

        vecs = neighbor_points_t - ray_origin_lidar.view(1, 1, 3)
        depth_t = torch.einsum('mkd,md->mk', vecs, ray_dir_lidar)
        proj = ray_origin_lidar.view(1, 1, 3) + depth_t.unsqueeze(-1) * ray_dir_lidar.unsqueeze(1)
        perp_dist_t = (neighbor_points_t - proj).norm(dim=-1)

        valid_t = pad_mask_t & (depth_t > MIN_DEPTH) & (perp_dist_t < MAX_PERP_DIST)
        depth_masked = torch.where(valid_t, depth_t, torch.full_like(depth_t, float("inf")))
        best_depth, best_local_idx = depth_masked.min(dim=1)
        has_match = torch.isfinite(best_depth)

        best_local_idx_cpu = best_local_idx.cpu().numpy()
        best_depth_cpu = best_depth.cpu().numpy()
        has_match_cpu = has_match.cpu().numpy()
        perp_at_best = perp_dist_t[torch.arange(m, device=_device), best_local_idx].cpu().numpy()

        for i in range(m):
            if not has_match_cpu[i]:
                continue
            # index LOCAL to lidar_points (this frame's cropped voxel subset)
            local_voxel_row = padded_idx[i, best_local_idx_cpu[i]]
            global_anchor_i = start + i
            anchor_has_match[global_anchor_i] = True
            anchor_best_voxel_row[global_anchor_i] = local_voxel_row
            anchor_best_point[global_anchor_i] = lidar_points[local_voxel_row]
            anchor_best_depth[global_anchor_i] = best_depth_cpu[i]
            anchor_best_perp[global_anchor_i] = perp_at_best[i]

        chunk_num = start // ANCHOR_CHUNK_SIZE + 1
        if chunk_num % 10 == 0 or chunk_num == total_chunks:
            elapsed_so_far = time.time() - chunk_start
            anchors_done = min(end, A)
            rate = anchors_done / max(elapsed_so_far, 1e-6)
            print(f"[raycast-gpu]   anchor-chunk {chunk_num}/{total_chunks} "
                  f"({anchors_done}/{A} unique anchors, {elapsed_so_far:.1f}s elapsed, "
                  f"{rate:.0f} anchors/sec)")

    pixel_to_point = {}
    total_matched = 0
    for pixel_i, anchor_i in enumerate(inverse):
        if not anchor_has_match[anchor_i]:
            continue
        key = keys[pixel_i]
        local_voxel_row = anchor_best_voxel_row[anchor_i]
        pixel_to_point[key] = {
            # FIX: "voxel_index" (row in the FULL voxel array, via
            # voxel_indices passed in), not "global_index" -- matches
            # what update_voxel_colors expects.
            "voxel_index": int(voxel_indices[local_voxel_row]),
            "point": anchor_best_point[anchor_i],
            "depth": float(anchor_best_depth[anchor_i]),
            "perp_dist": float(anchor_best_perp[anchor_i]),
        }
        total_matched += 1

    elapsed = time.time() - t0
    print(f"[raycast-gpu] Matched pixels (raycasting): {total_matched} / {len(keys)} "
          f"in {elapsed:.2f}s ({len(keys) / max(elapsed, 1e-6):.0f} pixels/sec)")

    if save_debug:
        pixel_to_point_path = os.path.join(OUTPUT_DIR, f"frame_{fid}_pixel_to_point.npy")
        np.save(pixel_to_point_path, pixel_to_point, allow_pickle=True)
        print(f"[raycast-gpu] Saved: {pixel_to_point_path}")

    return pixel_to_point


# ---------------------------------------------------------------
# STEP 3: colorize -- writes into the VOXEL-level color buffer
# ---------------------------------------------------------------
def update_voxel_colors(pixel_to_point, image_file, voxel_colors, voxel_colored_mask, voxel_color_counts):
    image = cv2.imread(image_file)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_file}")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    new_count = 0
    updated_count = 0
    conflict_count = 0

    for (u, v), info in pixel_to_point.items():
        idx = info["voxel_index"]
        new_color = image_rgb[v, u, :].astype(np.float64) / 255.0

        if not voxel_colored_mask[idx]:
            voxel_colors[idx] = new_color
            voxel_colored_mask[idx] = True
            voxel_color_counts[idx] = 1
            new_count += 1
        else:
            existing_color = voxel_colors[idx]
            diff = np.linalg.norm(new_color - existing_color)

            if diff <= COLOR_MATCH_THRESHOLD:
                n = voxel_color_counts[idx]
                voxel_colors[idx] = (existing_color * n + new_color) / (n + 1)
                voxel_color_counts[idx] = n + 1
                updated_count += 1
            else:
                conflict_count += 1

    return new_count, updated_count, conflict_count


def rgb_to_grayscale(colors):
    luminance = 0.299 * colors[:, 0] + 0.587 * colors[:, 1] + 0.114 * colors[:, 2]
    return np.stack([luminance, luminance, luminance], axis=1)


def format_duration(seconds):
    """Human-readable H:M:S, so long full runs (hours) are easy to
    read at a glance instead of just a raw seconds count."""
    hours, remainder = divmod(int(seconds), 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def load_checkpoint(num_voxels):
    required = ["voxel_colors.npy", "voxel_colored_mask.npy", "voxel_color_counts.npy", "processed_frames.json"]
    if not all(os.path.exists(os.path.join(CHECKPOINT_DIR, f)) for f in required):
        print("No voxel-gpu checkpoint found -- starting fresh.")
        voxel_colors = np.zeros((num_voxels, 3), dtype=np.float64)
        voxel_colored_mask = np.zeros(num_voxels, dtype=bool)
        voxel_color_counts = np.zeros(num_voxels, dtype=np.int32)
        return voxel_colors, voxel_colored_mask, voxel_color_counts, set()

    voxel_colors = np.load(os.path.join(CHECKPOINT_DIR, "voxel_colors.npy"))
    voxel_colored_mask = np.load(os.path.join(CHECKPOINT_DIR, "voxel_colored_mask.npy"))
    voxel_color_counts = np.load(os.path.join(CHECKPOINT_DIR, "voxel_color_counts.npy"))
    with open(os.path.join(CHECKPOINT_DIR, "processed_frames.json"), "r") as f:
        processed_frame_ids = set(json.load(f))

    print(f"Voxel-gpu checkpoint found: {len(processed_frame_ids)} frames already processed "
          f"-- resuming instead of starting over.")
    return voxel_colors, voxel_colored_mask, voxel_color_counts, processed_frame_ids


def save_checkpoint(voxel_colors, voxel_colored_mask, voxel_color_counts, processed_frame_ids):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    np.save(os.path.join(CHECKPOINT_DIR, "voxel_colors.npy"), voxel_colors)
    np.save(os.path.join(CHECKPOINT_DIR, "voxel_colored_mask.npy"), voxel_colored_mask)
    np.save(os.path.join(CHECKPOINT_DIR, "voxel_color_counts.npy"), voxel_color_counts)
    with open(os.path.join(CHECKPOINT_DIR, "processed_frames.json"), "w") as f:
        json.dump(sorted(processed_frame_ids), f)


# ---------------------------------------------------------------
# STEP 4: broadcast voxel colors back onto all original points
# ---------------------------------------------------------------
def build_and_save_point_level_output(map_points, map_colors_original, point_to_voxel,
                                       voxel_colors, voxel_colored_mask, output_path):
    n_points = map_points.shape[0]
    print(f"\nBroadcasting voxel colors back onto {n_points} original points...")

    final_colors = rgb_to_grayscale(map_colors_original)

    point_voxel_colored = voxel_colored_mask[point_to_voxel]
    final_colors[point_voxel_colored] = voxel_colors[point_to_voxel[point_voxel_colored]]

    final_cloud = o3d.geometry.PointCloud()
    final_cloud.points = o3d.utility.Vector3dVector(map_points)
    final_cloud.colors = o3d.utility.Vector3dVector(final_colors)
    o3d.io.write_point_cloud(output_path, final_cloud)

    num_colored_points = int(point_voxel_colored.sum())
    print(f"Saved: {output_path}")
    print(f"  Total points in file : {n_points} (same as input map)")
    print(f"  Points with real camera color (via their voxel) : {num_colored_points} "
          f"({100.0 * num_colored_points / n_points:.1f}%)")
    print(f"  Points kept as grayscale : {n_points - num_colored_points}")


def main():
    overall_start = time.time()

    with open(DATASET_INDEX_PATH, "r") as f:
        dataset = json.load(f)

    print(f"Loading global map once: {GLOBAL_MAP_PATH}")
    global_cloud = o3d.io.read_point_cloud(GLOBAL_MAP_PATH)
    map_points = np.asarray(global_cloud.points)
    map_colors_original = np.asarray(global_cloud.colors)
    n_points = map_points.shape[0]
    print(f"Loaded global map with {n_points} points\n")

    voxel_positions, point_to_voxel = voxelize_map(map_points, VOXEL_SIZE)
    num_voxels = voxel_positions.shape[0]

    voxel_colors, voxel_colored_mask, voxel_color_counts, processed_frame_ids = load_checkpoint(num_voxels)

    total_new = 0
    total_updated = 0
    total_conflicts = 0
    frames_done_this_run = 0

    for debug_index, entry in enumerate(dataset):
        fid = entry["frame_id"]

        if fid in processed_frame_ids:
            continue

        frame_start = time.time()
        print(f"\n=== Frame {fid} (voxel + GPU raycasting) ===")

        save_debug = SAVE_PER_FRAME_DEBUG_FILES and (debug_index < MAX_DEBUG_FRAMES)

        if save_debug:
            image_ext = os.path.splitext(entry["image_file"])[1]
            image_copy_path = os.path.join(OUTPUT_DIR, f"frame_{fid}_image{image_ext}")
            shutil.copy(entry["image_file"], image_copy_path)
            print(f"Saved copy of camera image: {image_copy_path}")

        lidar_points, voxel_indices, T_map_lidar = step1_crop_to_fov(
            entry, fid, voxel_positions, save_debug=save_debug
        )
        pixel_to_point = step2_raycast_gpu(lidar_points, voxel_indices, fid, save_debug=save_debug)

        new_c, upd_c, conf_c = update_voxel_colors(
            pixel_to_point, entry["image_file"], voxel_colors, voxel_colored_mask, voxel_color_counts
        )
        total_new += new_c
        total_updated += upd_c
        total_conflicts += conf_c
        processed_frame_ids.add(fid)
        frames_done_this_run += 1
        print(f"[colorize] frame {fid}: {new_c} newly colored voxels, "
              f"{upd_c} blended, {conf_c} rejected as conflicting (> {COLOR_MATCH_THRESHOLD} threshold)")

        frame_elapsed = time.time() - frame_start
        print(f"[timing] Frame {fid} took {frame_elapsed:.1f}s total")

        if frames_done_this_run % CHECKPOINT_EVERY_N_FRAMES == 0:
            save_checkpoint(voxel_colors, voxel_colored_mask, voxel_color_counts, processed_frame_ids)
            build_and_save_point_level_output(
                map_points, map_colors_original, point_to_voxel,
                voxel_colors, voxel_colored_mask,
                os.path.join(OUTPUT_DIR, "global_map_glim_colorized_voxel_gpu.ply")
            )
            elapsed_so_far = time.time() - overall_start
            print(f"[checkpoint] Saved progress ({len(processed_frame_ids)} frames total processed so far) "
                  f"-- elapsed so far: {format_duration(elapsed_so_far)}")

    save_checkpoint(voxel_colors, voxel_colored_mask, voxel_color_counts, processed_frame_ids)

    build_and_save_point_level_output(
        map_points, map_colors_original, point_to_voxel,
        voxel_colors, voxel_colored_mask,
        os.path.join(OUTPUT_DIR, "global_map_glim_colorized_voxel_gpu.ply")
    )

    total_elapsed = time.time() - overall_start
    print(f"\n=== FINAL SUMMARY ===")
    print(f"Frames in dataset               : {len(dataset)}")
    print(f"Frames processed this run        : {frames_done_this_run}")
    print(f"First-time voxel colors          : {total_new}")
    print(f"Accepted blends                  : {total_updated}")
    print(f"Rejected conflicts               : {total_conflicts}")
    print(f"Total time                       : {format_duration(total_elapsed)} "
          f"({total_elapsed:.1f}s)")
    if frames_done_this_run > 0:
        print(f"Average time per frame           : {format_duration(total_elapsed / frames_done_this_run)}")


if __name__ == "__main__":
    main()
