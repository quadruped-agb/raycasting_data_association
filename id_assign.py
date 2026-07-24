import os
import json
import numpy as np
import cv2
import open3d as o3d
import pycocotools.mask as mask_util

from raycasting import associate_pixel_to_lidar
from raycasting import build_transform_matrix, lidar_point_to_world

#path
FRAMES = {
    "frame_a": {
        "image": "output/images/frame_a.png",
        "pcd": "output/pointclouds/frame_a.pcd",
        "pose": "output/poses/frame_a_pose.json",
        "pred_json": "/home/administrator/Downloads/data_association/perceptree_frame_a_b_output/frame_a_pred.json",
    },
    "frame_b": {
        "image": "output/images/frame_b.png",
        "pcd": "output/pointclouds/frame_b.pcd",
        "pose": "output/poses/frame_b_pose.json",
        "pred_json": "/home/administrator/Downloads/data_association/perceptree_frame_a_b_output/frame_b_pred.json"
    },
}

REGISTRY_PATH = "tree_registry_multiframe.json"
MATCH_DISTANCE_THRESHOLD = 0.75  # meters
MAX_PERP_DIST = 0.05

if os.path.exists(REGISTRY_PATH):
    os.remove(REGISTRY_PATH)
registry = {}
next_id_number = 1

# process each frame in order
for frame_name, paths in FRAMES.items():
    print(f"\n{'='*70}\nProcessing {frame_name}\n{'='*70}")

    image = cv2.imread(paths["image"])
    height, width = image.shape[:2]

    with open(paths["pred_json"], "r") as f:
        predictions = json.load(f)
    print(f"Loaded {len(predictions)} tree instances")

    cloud = o3d.io.read_point_cloud(paths["pcd"])
    lidar_points = np.asarray(cloud.points)

    with open(paths["pose"], "r") as f:
        pose_data = json.load(f)
    T_map_baselink = build_transform_matrix(
        pose_data["translation"], pose_data["quaternion_xyzw"]
    )
    print(f"Frame pose (map frame): {np.round(pose_data['translation'], 3)}")

    #per-instance raycasting
    frame_detections = []

    for instance in predictions:
        rle = instance["segmentation"]
        score = instance["score"]

        instance_mask = mask_util.decode(rle)
        tree_pixel_coords = np.argwhere(instance_mask > 0)

        matched_points_lidar = []

        for (v, u) in tree_pixel_coords:
            u, v = int(u), int(v)
            point, depth, perp_dist, idx = associate_pixel_to_lidar(
                u, v, lidar_points, max_perp_dist=MAX_PERP_DIST
            )
            if point is not None:
                matched_points_lidar.append(point)

        if len(matched_points_lidar) == 0:
            print(f"  Instance (score={score:.2f}): no LiDAR matches - skipping")
            continue

        matched_points_lidar = np.array(matched_points_lidar)
        centroid_lidar = matched_points_lidar.mean(axis=0)

        # transform this tree's centroid into WORLD frame using this
        # frame's own pose 
        centroid_world = lidar_point_to_world(centroid_lidar, T_map_baselink)

        print(f"  Instance (score={score:.2f}): {len(matched_points_lidar)} matches, "
              f"centroid_lidar={np.round(centroid_lidar, 3)}, "
              f"centroid_world={np.round(centroid_world, 3)}")

        frame_detections.append({
            "centroid_world": centroid_world,
            "score": score,
        })

    #match against registry
    print(f"\n ID assignment for {frame_name}")

    for det in frame_detections:
        centroid = det["centroid_world"]

        best_id = None
        best_dist = None

        for tree_id, entry in registry.items():
            known_pos = np.array(entry["position"])
            dist = np.linalg.norm(centroid - known_pos)

            if dist < MATCH_DISTANCE_THRESHOLD and (best_dist is None or dist < best_dist):
                best_id = tree_id
                best_dist = dist

        if best_id is not None:
            registry[best_id]["position"] = centroid.tolist()
            registry[best_id]["last_seen_count"] = registry[best_id].get("last_seen_count", 1) + 1
            registry[best_id]["seen_in_frames"] = registry[best_id].get("seen_in_frames", []) + [frame_name]
            print(f"    MATCHED existing tree: {best_id} (distance={best_dist:.3f}m)")
        else:
            new_id = f"tree_{next_id_number:04d}"
            registry[new_id] = {
                "position": centroid.tolist(),
                "last_seen_count": 1,
                "seen_in_frames": [frame_name],
            }
            print(f"    NEW tree assigned: {new_id} at {np.round(centroid, 3)}")
            next_id_number += 1

# which trees were seen in BOTH frames 
print(f"\n{'='*70}\nFINAL RESULT\n{'='*70}")

seen_in_both = {tid: e for tid, e in registry.items() if len(set(e["seen_in_frames"])) >= 2}
seen_in_one = {tid: e for tid, e in registry.items() if len(set(e["seen_in_frames"])) == 1}

print(f"Trees seen in BOTH frames (successful re-identification): {len(seen_in_both)}")
for tid, e in seen_in_both.items():
    print(f"  {tid}: position={np.round(e['position'], 3)}, frames={e['seen_in_frames']}")

print(f"\nTrees seen in only ONE frame: {len(seen_in_one)}")
for tid, e in seen_in_one.items():
    print(f"  {tid}: position={np.round(e['position'], 3)}, frame={e['seen_in_frames']}")

with open(REGISTRY_PATH, "w") as f:
    json.dump(registry, f, indent=2)
print(f"\nFull registry saved: {REGISTRY_PATH}")