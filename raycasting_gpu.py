"""
raycasting_gpu.py

GLIM colorization pipeline: crop -> GPU raycast -> colorize into ONE
shared global color buffer.

  1. CROP    - crop the global map to this frame's camera FOV,
               output in LIDAR-local frame. Also keeps the GLOBAL
               INDEX of every surviving point, so we can always trace
               a point back to its row in the one, single global map
               array.
  2. RAYCAST - two-stage:
       a) fast vectorized projection to find which pixels have ANY
          candidate coverage at all
       b) raycasting (pixel -> ray -> nearest point within
          max_perp_dist), but ONLY for that narrowed candidate set,
          accelerated by a per-frame voxel grid so each pixel only
          searches its own 0.5m cell plus the 26 neighbours.
       Every match carries the point's GLOBAL index.
  3. COLORIZE - sample RGB from the full original image at each
       matched pixel and write it into a GLOBAL color buffer, indexed
       by that point's global index. The buffer starts as a GRAYSCALE
       version of the map's own height-based colors, so any point that
       never gets a camera match stays neutral gray -- the map's shape
       stays visible, and a gray point can never be mistaken for a
       genuine but dim camera color, since only real camera hits ever
       have unequal R/G/B channels. Output has the exact same N points
       as the input map; nothing is dropped.

       CONFLICT HANDLING: if a point was already colored by an earlier
       frame and a later frame proposes a different color, the new
       color is only accepted if within COLOR_MATCH_THRESHOLD of the
       existing one (running average in that case). Beyond that it's
       treated as a bad observation (occlusion, reflection, projection
       error) and rejected; the existing color is kept.

WHY THIS FILE WAS REWRITTEN
    The previous on-disk version was two files concatenated, and
    defined T_cam_lidar TWICE -- correct at the top, stale/broken
    further down. Python resolves globals at call time, so at runtime:
      - the raycast fast pass used the numpy R_cam_lidar  -> BROKEN
      - the GPU ray construction used _R_cam_lidar_t, a torch tensor
        built at import time from the top (correct) matrix, never
        redefined                                        -> CORRECT
      - step1_crop_to_fov used T_cam_lidar               -> BROKEN
    So candidate pixels were computed with one geometry and rays were
    built through those pixels with a different one. Every point sat
    far off its own ray, perp_dist blew past MAX_PERP_DIST, and the
    run reported "Matched pixels: 0" on every single frame. That file
    also imported itself (`from raycasting_gpu import ...`) and had a
    corrupted docstring. This version defines everything exactly once.

CALIBRATION NOTE
    T_cam_lidar is NOT the raw Ry(-25 deg) matrix from
    sensor_calibration.docx. The doc value describes the transform
    between two BODY-convention frames (X forward, Y left, Z up),
    while the intrinsics K assume the OPTICAL convention (Z forward,
    X right, Y down). Composing only the body-frame tilt, without the
    body->optical axis remap, puts 0% of near-field points in FOV.
    The matrix here is Ry(-25 deg) @ R_body_from_optical with
    X_opt = -Y_body, Y_opt = -Z_body, Z_opt = X_body. Row 2 must read
    [-1, 0, 0]. Do not "correct" this back to the doc values.

    Note also that `unitree_go2/front_cam` is therefore not a standard
    optical frame -- anything else consuming these extrinsics (the
    segmentation node, tree-ID work) hits the same trap.

PER-FRAME DEBUG FILES
    Every frame in the dataset is always cropped, raycasted, and
    colorized into the ONE shared global buffer. What's limited to the
    first MAX_DEBUG_FRAMES frames is only the optional per-frame
    artifacts written for human inspection. Frames beyond that limit
    still contribute fully to the final map.

Output:
    output/frame_<id>_cropped_map.ply     (debug, first MAX_DEBUG_FRAMES frames, lidar-local frame)
    output/frame_<id>_pixel_to_point.npy  (debug, first MAX_DEBUG_FRAMES frames, full-image (u,v) keys)
    output/frame_<id>_image.*             (debug, first MAX_DEBUG_FRAMES frames, copy of source camera image)
    output/frame_<id>_colorized.ply       (debug, first MAX_DEBUG_FRAMES frames, this frame's own
                                            cropped points colorized from its own camera image --
                                            real RGB where matched, flat white elsewhere)
    output/global_map_glim_colorized.ply  (FINAL - same N points as the input map, camera RGB
                                            where matched, grayscale elsewhere. Input NOT modified.)
    output/checkpoint/                    (resumable state: global_colors.npy, colored_mask.npy,
                                            color_counts.npy, processed_frames.json)
"""

import os
import json
import numpy as np
import torch
import open3d as o3d
import cv2
from scipy.spatial import cKDTree
import time
import shutil

# ---------------------------------------------------------------
# Calibration -- DEFINED EXACTLY ONCE. See CALIBRATION NOTE above.
#
# Maps a point expressed in CAMERA (optical) frame into LIDAR frame:
#   p_lidar = R_cam_lidar @ p_cam + t_cam_lidar
#
# NOTE: there is no base_link transform here. GLIM's poses (from
# traj_lidar.txt via the dataset index) are already map->lidar
# directly -- GLIM has no base_link concept at all. Composing through
# a lidar->baselink calibration would double-transform every point.
# ---------------------------------------------------------------
T_cam_lidar = np.array([
    [ 0.0000,  0.4226,  0.9063, 0.142],
    [-1.0000,  0.0000,  0.0000, 0.000],
    [ 0.0000, -0.9063,  0.4226, 0.005],
    [ 0.0000,  0.0000,  0.0000, 1.000],
])
R_cam_lidar = T_cam_lidar[:3, :3]
t_cam_lidar = T_cam_lidar[:3, 3]
R_lidar_cam = R_cam_lidar.T  # inverse rotation: lidar -> camera

K = np.array([
    [1219.92, 0.0,     960.0],
    [0.0,     1219.92, 540.0],
    [0.0,     0.0,     1.0],
])
K_inv = np.linalg.inv(K)

IMG_W, IMG_H = 1920, 1080
MAX_RANGE = 70.0     # metres, Livox Mid360 spec
MIN_DEPTH = 0.05     # metres, ignore points at/behind the camera
MAX_PERP_DIST = 0.15 # metres, raycasting tolerance

# how close (in normalized 0-1 RGB space, euclidean distance) a newly
# proposed color must be to a point's EXISTING color to be accepted as
# "same surface, different frame". Smaller = stricter.
COLOR_MATCH_THRESHOLD = 0.12

# --- GPU voxel-grid neighbour-search settings ---
# VOXEL_SIZE=0.5 with a 3x3x3 neighbour search guarantees at least
# 0.5m coverage in every direction from any point in the centre cell --
# a superset of the 0.45m (MAX_PERP_DIST * 3) radius the CPU version
# used. Extra points beyond the true radius get filtered by the
# perp_dist check anyway, so being generous costs compute but never
# causes wrong results.
VOXEL_SIZE = 0.5
NEIGHBOR_RING = 1  # 1 = 3x3x3 = 27 cells checked per anchor

ANCHOR_CHUNK_SIZE = 2048
NEIGHBOR_CAP = 512

# Large coordinate offset so voxel indices are non-negative before
# packing into a single hashable integer key.
_COORD_OFFSET = 1 << 20

SAVE_PER_FRAME_DEBUG_FILES = True
MAX_DEBUG_FRAMES = 20

# Colour written to points in the per-frame colorized .ply that this
# frame's raycast did NOT match to any pixel. 1.0 = white.
PER_FRAME_UNMATCHED_COLOR = 1.0

# paths
DATASET_INDEX_PATH = "output/dataset_index_glim_2.json"
GLOBAL_MAP_PATH = "output/global_map_glim_2.ply"
OUTPUT_DIR = "output"
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoint")

CHECKPOINT_EVERY_N_FRAMES = 20

_device = "cuda" if torch.cuda.is_available() else "cpu"
if _device == "cpu":
    print("[gpu-raycast] WARNING: CUDA not available -- running on CPU tensors "
          "(still batched/parallel across pixels, just not on GPU cores).")

# calibration tensors, built ONCE from the single matrix above
_K_inv_t = torch.tensor(K_inv, dtype=torch.float32, device=_device)
_R_cam_lidar_t = torch.tensor(R_cam_lidar, dtype=torch.float32, device=_device)
_t_cam_lidar_t = torch.tensor(t_cam_lidar, dtype=torch.float32, device=_device)


def quaternion_to_matrix(x, y, z, w):
    """Quaternion (the 4-number rotation format /tf publishes) -> 3x3
    rotation matrix, since everything else here uses matrices."""
    return np.array([
        [1 - 2*(y**2 + z**2), 2*(x*y - z*w),       2*(x*z + y*w)],
        [2*(x*y + z*w),       1 - 2*(x**2 + z**2), 2*(y*z - x*w)],
        [2*(x*z - y*w),       2*(y*z + x*w),       1 - 2*(x**2 + y**2)],
    ])


def build_transform_matrix(translation, quaternion_xyzw):
    """Translation (x,y,z) + quaternion (x,y,z,w) -> one 4x4 matrix."""
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


def pixel_to_ray_camera_frame(u, v):
    """
    Pixel (u, v) -> normalized 3D ray direction in the CAMERA's own
    frame (origin at camera centre, ray into the scene along +Z).
    """
    pixel_h = np.array([u, v, 1.0])
    ray_dir_cam = K_inv @ pixel_h
    return ray_dir_cam / np.linalg.norm(ray_dir_cam)


def ray_to_lidar_frame(ray_dir_cam):
    """
    Transforms a camera-frame ray (origin + direction) into the LiDAR
    frame. Origin: camera centre. Direction: rotate only -- directions
    aren't affected by translation.
    """
    ray_origin_lidar = t_cam_lidar.copy()
    ray_dir_lidar = R_cam_lidar @ ray_dir_cam
    ray_dir_lidar = ray_dir_lidar / np.linalg.norm(ray_dir_lidar)
    return ray_origin_lidar, ray_dir_lidar


def find_nearest_point_near_anchor(tree, points, ray_origin, ray_dir, anchor,
                                    max_perp_dist=MAX_PERP_DIST, min_depth=MIN_DEPTH):
    """
    CPU reference: nearest first-hit along the ray, checked only
    against points near a known anchor rather than the full array.
    Returns an index LOCAL to `points`. Used by the CPU step2_raycast
    below; main() runs the GPU path.
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
    return points[idxs[best]], depth[best], perp_dist[best], idxs[best]


# ---------------------------------------------------------------
# STEP 1: crop the global map to this frame's camera FOV.
# Returns points in LIDAR-LOCAL frame plus `global_indices`: for every
# surviving point, its row in the original full-size map array. That's
# the thread keeping "one point, one row, forever" through raycasting
# and colorizing.
#
# `save_debug` controls ONLY whether the debug .ply is written --
# cropping itself is identical either way.
# ---------------------------------------------------------------
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


# ---------------------------------------------------------------
# GPU voxel-grid neighbour-search helpers
# ---------------------------------------------------------------
def _pack_voxel_keys(voxel_coords):
    """
    voxel_coords: (N, 3) int64 voxel grid indices (may be negative).
    Returns (N,) int64 hashable keys, packing the 3 coords into one
    int64. Each axis fits comfortably in 21 bits (+/-1M cells * 0.5m
    = +/-500km, far beyond any real scene).
    """
    shifted = voxel_coords + _COORD_OFFSET  # guaranteed non-negative
    keys = (shifted[:, 0].astype(np.int64) << 42) \
         | (shifted[:, 1].astype(np.int64) << 21) \
         | (shifted[:, 2].astype(np.int64))
    return keys


def _build_voxel_grid(points, voxel_size):
    """
    Buckets `points` (N, 3) into a voxel grid as a CSR-like structure:
      - order: point indices sorted by voxel key
      - unique_keys: sorted unique voxel keys present
      - start_idx: start offset into `order` per unique key
      - counts: points per voxel
    """
    voxel_coords = np.floor(points / voxel_size).astype(np.int64)
    keys = _pack_voxel_keys(voxel_coords)
    order = np.argsort(keys, kind="stable")
    sorted_keys = keys[order]
    unique_keys, start_idx, counts = np.unique(
        sorted_keys, return_index=True, return_counts=True
    )
    return order, unique_keys, start_idx, counts, voxel_coords


def _query_neighbors(anchor_voxel_coords, order, unique_keys, start_idx, counts):
    """
    For each anchor voxel coord (m, 3), gathers point indices from its
    own cell + the 26 neighbours (NEIGHBOR_RING=1). Returns a list of
    length m, each a numpy array of point indices, capped at
    NEIGHBOR_CAP.
    """
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
# STEP 2 (GPU): raycasting, two-stage, all in full-image (u,v) coords.
# `global_indices[i]` gives the row in the full global map that
# `lidar_points[i]` corresponds to.
#
# Candidate pixels are deduplicated by shared anchor before any GPU
# work (many pixels share an anchor due to the MARGIN dilation), so
# the neighbour search and distance math run once per unique anchor,
# then broadcast back out to every pixel sharing it.
# ---------------------------------------------------------------
def step2_raycast_gpu(lidar_points, global_indices, fid, save_debug=False):
    t0 = time.time()

    # --- fast pass: plain vectorized projection ---
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

    print(f"[raycast-gpu] Fast pass found {valid.sum()} point projections -> "
          f"{len(candidate_pixels)} candidate pixels to actually raycast")

    # --- dedupe pixels by shared anchor ---
    keys = list(candidate_pixels.keys())
    us = np.array([k[0] for k in keys], dtype=np.float32)
    vs = np.array([k[1] for k in keys], dtype=np.float32)
    anchors = np.array([candidate_pixels[k] for k in keys])  # (M, 3)

    unique_anchors, inverse = np.unique(anchors, axis=0, return_inverse=True)
    inverse = inverse.reshape(-1)
    A = unique_anchors.shape[0]
    print(f"[raycast-gpu] {len(keys)} pixels -> {A} unique anchors "
          f"({len(keys) / max(A, 1):.1f}x dedup)")

    # --- voxel grid over this frame's cropped lidar points ---
    order, unique_keys, start_idx, counts, _ = _build_voxel_grid(lidar_points, VOXEL_SIZE)
    anchor_voxel_coords = np.floor(unique_anchors / VOXEL_SIZE).astype(np.int64)

    lidar_points_t = torch.tensor(lidar_points, dtype=torch.float32, device=_device)

    # ONE representative pixel per unique anchor, computed ONCE up
    # front -- O(total pixels). Doing this inside the chunk loop via
    # np.where(inverse == anchor_idx) would be O(anchors * pixels),
    # catastrophically slow at these sizes.
    rep_pixel_for_anchor = np.full(A, -1, dtype=np.int64)
    seen_anchor = np.zeros(A, dtype=bool)
    for pixel_i, anchor_i in enumerate(inverse):
        if not seen_anchor[anchor_i]:
            rep_pixel_for_anchor[anchor_i] = pixel_i
            seen_anchor[anchor_i] = True

    # per-unique-anchor results. NOTE: anchor_best_local_idx holds an
    # index into THIS frame's lidar_points, not the global map --
    # it gets mapped through global_indices at the end.
    anchor_best_local_idx = np.full(A, -1, dtype=np.int64)
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
            continue  # no anchors in this chunk have ANY neighbour points

        padded_idx = np.zeros((m, max_k), dtype=np.int64)
        pad_mask = np.zeros((m, max_k), dtype=bool)
        for i, idxs in enumerate(neighbor_idx_lists):
            k = len(idxs)
            if k > 0:
                padded_idx[i, :k] = idxs
                pad_mask[i, :k] = True

        padded_idx_t = torch.tensor(padded_idx, dtype=torch.long, device=_device)
        pad_mask_t = torch.tensor(pad_mask, dtype=torch.bool, device=_device)

        neighbor_points_t = lidar_points_t[padded_idx_t]  # (m, max_k, 3)

        # The ray must be built from the PIXEL each anchor maps to, not
        # the anchor's own position. Dedup was on the 3D anchor, and
        # several pixels can share one anchor, so each unique anchor
        # uses the first pixel that produced it as its representative
        # (u, v) -- consistent with a per-pixel ray, since all pixels
        # sharing an anchor are within MARGIN of each other.
        rep_pixel_idx = rep_pixel_for_anchor[start:end]

        u_chunk = torch.tensor(us[rep_pixel_idx], device=_device)
        v_chunk = torch.tensor(vs[rep_pixel_idx], device=_device)

        ones = torch.ones_like(u_chunk)
        pix_h = torch.stack([u_chunk, v_chunk, ones], dim=0)  # (3, m)
        ray_dir_cam = _K_inv_t @ pix_h
        ray_dir_cam = ray_dir_cam / ray_dir_cam.norm(dim=0, keepdim=True)

        ray_dir_lidar = _R_cam_lidar_t @ ray_dir_cam
        ray_dir_lidar = ray_dir_lidar / ray_dir_lidar.norm(dim=0, keepdim=True)
        ray_dir_lidar = ray_dir_lidar.T  # (m, 3)
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
            local_row = padded_idx[i, best_local_idx_cpu[i]]
            global_anchor_i = start + i
            anchor_has_match[global_anchor_i] = True
            anchor_best_local_idx[global_anchor_i] = local_row
            anchor_best_point[global_anchor_i] = lidar_points[local_row]
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

    # --- broadcast each unique anchor's result to every pixel sharing it ---
    pixel_to_point = {}
    total_matched = 0
    for pixel_i, anchor_i in enumerate(inverse):
        if not anchor_has_match[anchor_i]:
            continue
        key = keys[pixel_i]
        local_row = anchor_best_local_idx[anchor_i]
        pixel_to_point[key] = {
            "global_index": int(global_indices[local_row]),
            # row in THIS frame's own lidar_points subset -- lets the
            # per-frame colorized cloud be built by direct indexing.
            # Invariant: global_indices[local_index] == global_index.
            "local_index": int(local_row),
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
# STEP 2 (CPU reference) -- kept for comparison. main() uses the GPU
# path above.
# ---------------------------------------------------------------
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
                "global_index": int(global_indices[local_idx]),
                "local_index": int(local_idx),
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


# ---------------------------------------------------------------
# STEP 3: sample RGB at each matched pixel and write it into the
# GLOBAL color buffer at that point's global index.
#
# global_colors : (N, 3) float, shared across all frames. Starts as a
#                 GRAYSCALE version of the map's original height-based
#                 colors, so untouched rows stay neutral gray.
# colored_mask  : (N,) bool, True once a CAMERA color has been written.
#                 Only used to decide "first hit -> set" vs "repeat hit
#                 -> apply threshold"; it does NOT gate what gets
#                 written to the output file (every point is written).
# color_counts  : (N,) int, how many frames contributed (for the
#                 running average).
#
# ALWAYS runs for every frame -- never gated by the debug-file limit.
# ---------------------------------------------------------------
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
            global_colors[idx] = new_color
            colored_mask[idx] = True
            color_counts[idx] = 1
            new_count += 1
        else:
            existing_color = global_colors[idx]
            diff = np.linalg.norm(new_color - existing_color)

            if diff <= COLOR_MATCH_THRESHOLD:
                n = color_counts[idx]
                global_colors[idx] = (existing_color * n + new_color) / (n + 1)
                color_counts[idx] = n + 1
                updated_count += 1
            else:
                conflict_count += 1

    return new_count, updated_count, conflict_count


# ---------------------------------------------------------------
# DEBUG: save THIS frame's own cropped points as their own colorized
# .ply -- real RGB where this frame's raycast matched a pixel, flat
# WHITE everywhere else (PER_FRAME_UNMATCHED_COLOR). Independent of
# (and doesn't affect) the cumulative global color buffer. Written in
# LIDAR-LOCAL frame, same as frame_<id>_cropped_map.ply, so the two
# overlay exactly.
#
# NOTE: white unmatched points will be invisible against a light
# viewer background -- set the background dark in CloudCompare/Open3D
# to see them, or change PER_FRAME_UNMATCHED_COLOR.
# ---------------------------------------------------------------
def save_frame_colorized_cloud(lidar_points, pixel_to_point, image_file, output_path):
    image = cv2.imread(image_file)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_file}")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    colors = np.full((lidar_points.shape[0], 3), PER_FRAME_UNMATCHED_COLOR)
    matched = np.zeros(lidar_points.shape[0], dtype=bool)
    for (u, v), info in pixel_to_point.items():
        row = info["local_index"]
        colors[row] = image_rgb[v, u, :].astype(np.float64) / 255.0
        matched[row] = True

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(lidar_points)
    cloud.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(output_path, cloud)

    n = lidar_points.shape[0]
    print(f"[per-frame] Saved {output_path}")
    print(f"[per-frame]   {n} cropped points, {int(matched.sum())} with real camera "
          f"color ({100.0 * matched.sum() / max(n, 1):.1f}%), rest flat white")


def rgb_to_grayscale(colors):
    """
    (N,3) RGB in [0,1] -> grayscale via standard luminance weighting,
    replicated across all 3 channels so it stays a valid (N,3) color
    array. Turns the map's height-based colormap into a neutral
    backdrop: the shape stays visible, and only real camera hits ever
    have unequal R/G/B, so gray can never be mistaken for a dim
    camera color.
    """
    luminance = 0.299 * colors[:, 0] + 0.587 * colors[:, 1] + 0.114 * colors[:, 2]
    return np.stack([luminance, luminance, luminance], axis=1)


def load_checkpoint(n_points, map_colors_original):
    """
    Loads output/checkpoint/ if present, else returns fresh state.
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
    print("  NOTE: if this checkpoint predates the calibration fix, those frames "
          "will be SKIPPED with their bad colors intact. Delete "
          f"{CHECKPOINT_DIR} if unsure.")
    return global_colors, colored_mask, color_counts, processed_frame_ids


def save_checkpoint(global_colors, colored_mask, color_counts, processed_frame_ids):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    np.save(os.path.join(CHECKPOINT_DIR, "global_colors.npy"), global_colors)
    np.save(os.path.join(CHECKPOINT_DIR, "colored_mask.npy"), colored_mask)
    np.save(os.path.join(CHECKPOINT_DIR, "color_counts.npy"), color_counts)
    with open(os.path.join(CHECKPOINT_DIR, "processed_frames.json"), "w") as f:
        json.dump(sorted(processed_frame_ids), f)


def replay_orphaned_pixel_to_point_files(dataset, global_colors, colored_mask, color_counts, processed_frame_ids):
    """
    RECOVERY: replay any leftover frame_<id>_pixel_to_point.npy files
    for frames NOT already marked processed, folding their work into
    the color buffer without re-doing cropping/raycasting. Frames
    already processed are skipped, so nothing double-counts.

    WARNING: leftover .npy files written with a DIFFERENT (e.g.
    pre-calibration-fix) T_cam_lidar contain bad associations, and
    replaying them silently folds that bad data in. Delete
    output/frame_*_pixel_to_point.npy before a fresh run if the
    calibration has changed since those files were written.
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
              f"the color buffer.")

    return replayed


def main():
    overall_start = time.time()

    # fail fast on a stale calibration matrix rather than after hours
    if not np.allclose(T_cam_lidar[1, :3], [-1.0, 0.0, 0.0], atol=1e-3):
        raise RuntimeError(
            "T_cam_lidar row 2 is not [-1, 0, 0]. This looks like the old "
            "pre-fix calibration matrix (body-frame tilt without the "
            "body->optical remap). See the CALIBRATION NOTE at the top."
        )

    with open(DATASET_INDEX_PATH, "r") as f:
        dataset = json.load(f)

    print(f"Loading global map once: {GLOBAL_MAP_PATH}")
    global_cloud = o3d.io.read_point_cloud(GLOBAL_MAP_PATH)
    map_points = np.asarray(global_cloud.points)
    map_colors_original = np.asarray(global_cloud.colors)
    n_points = map_points.shape[0]
    print(f"Loaded global map with {n_points} points\n")

    # ONE persistent color buffer for the ENTIRE global map, shared
    # across every frame. Starts as grayscale of the map's own
    # height-based colors, so unmatched points stay neutral gray.
    # Resumes from a checkpoint if one exists.
    global_colors, colored_mask, color_counts, processed_frame_ids = load_checkpoint(
        n_points, map_colors_original
    )

    replayed = replay_orphaned_pixel_to_point_files(
        dataset, global_colors, colored_mask, color_counts, processed_frame_ids
    )
    if replayed:
        save_checkpoint(global_colors, colored_mask, color_counts, processed_frame_ids)

    total_new = 0
    total_updated = 0
    total_conflicts = 0
    frames_done_this_run = 0
    per_frame_clouds_saved = 0

    # `debug_index` walks dataset order independent of how many frames
    # actually get processed this run, so re-runs always pick the same
    # frames for debug output regardless of resume state.
    for debug_index, entry in enumerate(dataset):
        fid = entry["frame_id"]

        if fid in processed_frame_ids:
            continue

        print(f"\n=== Frame {fid} ===")

        save_debug = SAVE_PER_FRAME_DEBUG_FILES and (debug_index < MAX_DEBUG_FRAMES)

        if save_debug:
            image_ext = os.path.splitext(entry["image_file"])[1]
            image_copy_path = os.path.join(OUTPUT_DIR, f"frame_{fid}_image{image_ext}")
            shutil.copy(entry["image_file"], image_copy_path)
            print(f"Saved copy of camera image: {image_copy_path}")

        lidar_points, global_indices, T_map_lidar = step1_crop_to_fov(
            entry, fid, map_points, save_debug=save_debug
        )
        pixel_to_point = step2_raycast_gpu(lidar_points, global_indices, fid, save_debug=save_debug)

        # per-frame standalone colorized cloud, first MAX_DEBUG_FRAMES
        # frames only, written right here as the run proceeds
        if save_debug:
            frame_colorized_path = os.path.join(OUTPUT_DIR, f"frame_{fid}_colorized.ply")
            save_frame_colorized_cloud(
                lidar_points, pixel_to_point, entry["image_file"], frame_colorized_path
            )
            per_frame_clouds_saved += 1

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

    save_checkpoint(global_colors, colored_mask, color_counts, processed_frame_ids)

    # FINAL colorized global map: same N points as the input, always.
    # Matched points carry real RGB; unmatched keep neutral grayscale.
    # The input file itself is never modified.
    final_cloud = o3d.geometry.PointCloud()
    final_cloud.points = o3d.utility.Vector3dVector(map_points)
    final_cloud.colors = o3d.utility.Vector3dVector(global_colors)

    final_path = os.path.join(OUTPUT_DIR, "global_map_glim_colorized.ply")
    o3d.io.write_point_cloud(final_path, final_cloud)

    print(f"\nSaved final colorized global map: {final_path}")
    print(f"  Total points in file      : {n_points} (same as input map)")
    print(f"  Points with real camera color : {int(colored_mask.sum())}")
    print(f"  Points kept as grayscale (no camera match) : {int((~colored_mask).sum())}")
    print(f"  Per-frame colorized clouds saved : {per_frame_clouds_saved} (limit {MAX_DEBUG_FRAMES})")
    print(f"Totals across all frames: {total_new} first-time colors, "
          f"{total_updated} accepted blends, {total_conflicts} rejected conflicts")
    print(f"Total batch time: {time.time() - overall_start:.1f}s for {len(dataset)} frames")


if __name__ == "__main__":
    main()