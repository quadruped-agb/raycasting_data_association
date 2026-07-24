"""
global_map_glim.py

Builds a single global (map-frame) point cloud by fusing every LiDAR
frame listed in dataset_index_glim.json using its corresponding pose.

IMPORTANT: unlike the original global_map.py, the poses here come from
GLIM's own traj_lidar.txt (via resync_poses.py), interpolated to each
scan's timestamp. Those poses are already map -> lidar directly (GLIM
has no concept of a base_link frame at all - see config_sensors.json,
which only defines T_lidar_camera and T_lidar_imu). So there is no
separate lidar->baselink calibration step here: each scan's points go
straight from the LiDAR-local frame into the map frame using the
GLIM pose.

Input:
    output/dataset_index_glim_2.json
        each entry has: frame_id, lidar_file, image_file, pose {translation, quaternion}

Output:
    output/global_map_glim.ply            -> merged point cloud, in MAP frame, RGB colored by height (Z)
    output/global_map_glim_frame_ids.npy   -> (N,) int array, frame_id per point
    output/trajectory_glim.ply             -> trajectory as a colored point cloud
    output/trajectory_glim.obj             -> trajectory as a polyline
"""

import os
import json
import numpy as np
import open3d as o3d
import matplotlib.cm as cm
import matplotlib.colors as mcolors

# paths
DATASET_INDEX_PATH = "output/dataset_index_glim_2.json"
OUTPUT_MAP_PATH = "output/global_map_glim_2.ply"
OUTPUT_FRAMEIDS_PATH = "output/global_map_glim_frame_ids_2.npy"
OUTPUT_TRAJ_PCD_PATH = "output/trajectory_glim_2.ply"
OUTPUT_TRAJ_OBJ_PATH = "output/trajectory_glim_2.obj"

# colormap used to color points/trajectory by frame_id 
COLORMAP_NAME = "jet" 



# helpers:

# 4 numbers (x, y, z, w), and this formula produces the 3×3 rotation matrix that represents
# the exact same rotation, just in a different "shape"
def quaternion_to_matrix(x, y, z, w):
    """Standard quaternion (xyzw) to 3x3 rotation matrix."""
    # bec the rest of the code uses 3 by 3 rotation matrix
    R = np.array([
        [1 - 2*(y**2 + z**2), 2*(x*y - z*w),       2*(x*z + y*w)],
        [2*(x*y + z*w),       1 - 2*(x**2 + z**2), 2*(y*z - x*w)],
        [2*(x*z - y*w),       2*(y*z + x*w),       1 - 2*(x**2 + y**2)],
    ])
    return R

# robot's pose has translation and rotation, this converts them both into one 4 by 4 matrix
def build_transform_matrix(translation, quaternion_xyzw):
    """Builds a 4x4 transform from translation + quaternion."""
    tx, ty, tz = translation
    x, y, z, w = quaternion_xyzw
    T = np.eye(4)  # 4 by 4 identity matrix
    # top left 3 by 3 part filled with rotation matrix
    T[:3, :3] = quaternion_to_matrix(x, y, z, w)
    # fill last col's top 3 rows with the translation
    T[:3, 3] = [tx, ty, tz]
    return T


# takes points and converts them to the above 4 by 4 matrix
def transform_points(points_local, T):
    """transform of an (N,3) point array by a 4x4 matrix T."""
    # points local = Nx3 array so N numbers with x,y,z
    n = points_local.shape[0]  # this is N
    points_h = np.hstack([points_local, np.ones((n, 1))])  # adds 1 to the end of all points to make it :(N,4)
    # take h.T to convert to (4xN) for matrix multiplcaition
    # multiply by T: 4xN then  tranpose again to get Nx4 output (one row per point)
    points_transformed_h = (T @ points_h.T).T
    return points_transformed_h[:, :3]  # removes the last 1 from (x y z 1) and output is (x y z) points


def values_to_rgb(values, cmap_name=COLORMAP_NAME, vmin=None, vmax=None):
    """
    Maps an array of scalar values to (N,3) RGB colors in [0,1] using a
    matplotlib colormap. If vmin/vmax are not given, normalizes over the
    min/max of the values themselves.
    """
    values = np.asarray(values, dtype=np.float64)
    if vmin is None:
        vmin = values.min()
    if vmax is None:
        vmax = values.max()
    if vmax > vmin:
        norm = (values - vmin) / (vmax - vmin)
        norm = np.clip(norm, 0.0, 1.0)
    else:
        norm = np.zeros_like(values)
    cmap = cm.get_cmap(cmap_name)
    colors = cmap(norm)[:, :3]  # drop alpha channel
    return colors


def height_to_rgb(points_xyz, cmap_name=COLORMAP_NAME):
    """
    Classic lidar point cloud coloring: maps each point's Z (height)
    to a rainbow gradient, normalized over the min/max height in the
    cloud. Low points -> one end of the colormap, high points -> the
    other end.
    """
    z = points_xyz[:, 2]
    return values_to_rgb(z, cmap_name=cmap_name)


def write_trajectory_obj(positions, path):
    """
    Writes trajectory positions as a Wavefront .obj file: vertices (v)
    plus consecutive line segments (l). CloudCompare imports these
    line elements as a real polyline object, so you get a connected
    trajectory line rather than a scatter of points.
    """
    with open(path, "w") as f:
        f.write("# trajectory polyline, one vertex per frame pose\n")
        for p in positions:
            f.write(f"v {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
        # obj indices are 1-based; connect each consecutive pair of vertices
        n = len(positions)
        for i in range(1, n):
            f.write(f"l {i} {i + 1}\n")


# main loop
def main():

    with open(DATASET_INDEX_PATH, "r") as f:
        dataset = json.load(f)  # load as a lsit

    print(f"Loaded {len(dataset)} frames from {DATASET_INDEX_PATH}\n")
    # list of (Ni, 3) arrays, one per frame, in MAP frame
    # thsi has the transformed points in the map frame from every frame
    all_points_map = []

    # this has which point each frame came from
    all_frame_ids = []    # list of (Ni,) arrays, frame_id repeated per point

    # trajectory: one position per frame (in order), taken straight from the pose
    trajectory_positions = []
    trajectory_frame_ids = []

    for entry in dataset:
        # loading pcd, frame id and pose:
        frame_id = entry["frame_id"]
        lidar_file = entry["lidar_file"]
        pose = entry["pose"]

        # build this frame's map <- lidar transform from pose (4x4 matrix,
        # position + rotation). Unlike the base_link version, this pose
        # IS the map->lidar transform directly (that's what GLIM's
        # traj_lidar.txt gives us), so there's no separate calibration
        # step to compose in here.
        T_map_lidar = build_transform_matrix(
            pose["translation"],  # xyz
            pose["quaternion"]  # xyzw
        )

        # record trajectory point for this frame regardless of whether the
        # lidar scan itself had points, so the line doesn't have gaps
        trajectory_positions.append(np.array(pose["translation"], dtype=np.float64))
        trajectory_frame_ids.append(frame_id)

        # load this frame's raw local-frame LiDAR points
        cloud = o3d.io.read_point_cloud(lidar_file)
        points_lidar = np.asarray(cloud.points)   # (Ni, 3), LiDAR-local frame
        # if 0 points
        if points_lidar.shape[0] == 0:
            print(f"Frame {frame_id}: no points, skipping")
            continue

        # lidar local frame directly to map frame:
        points_map = transform_points(points_lidar, T_map_lidar)

        # add tranfomed points to list
        all_points_map.append(points_map)
        all_frame_ids.append(np.full(points_map.shape[0], frame_id, dtype=np.int32))

        if frame_id % 50 == 0:
            print(f"Processed frame {frame_id} ({points_map.shape[0]} points)")

    # concatenate everything into one global cloud
    # from the multiple arrays (one per frame) to a single array
    global_points = np.concatenate(all_points_map, axis=0)
    global_frame_ids = np.concatenate(all_frame_ids, axis=0)

    print(f"\nTotal points before downsampling: {global_points.shape[0]}")

    # color the global cloud by height (Z) - classic lidar point cloud
    # rainbow gradient look
    global_colors = height_to_rgb(global_points)

    # saving as an open3d pointcloud
    global_cloud = o3d.geometry.PointCloud()
    global_cloud.points = o3d.utility.Vector3dVector(global_points)
    global_cloud.colors = o3d.utility.Vector3dVector(global_colors)

    # if VOXEL_SIZE is not None:
    #     # voxel_down_sample does NOT preserve per-point frame_id
    #     # correspondence directly
    #     global_cloud = global_cloud.voxel_down_sample(voxel_size=VOXEL_SIZE)
    #     print(f"Total points after voxel downsampling ({VOXEL_SIZE} m): "
    #           f"{len(global_cloud.points)}")

    os.makedirs(os.path.dirname(OUTPUT_MAP_PATH), exist_ok=True)
    o3d.io.write_point_cloud(OUTPUT_MAP_PATH, global_cloud)
    np.save(OUTPUT_FRAMEIDS_PATH, global_frame_ids)

    # --- trajectory outputs ---
    trajectory_positions = np.array(trajectory_positions)
    trajectory_frame_ids = np.array(trajectory_frame_ids)
    # trajectory stays colored by sequence order (not height) so you can
    # see the direction/order of travel along the path
    trajectory_colors = values_to_rgb(trajectory_frame_ids)

    trajectory_cloud = o3d.geometry.PointCloud()
    trajectory_cloud.points = o3d.utility.Vector3dVector(trajectory_positions)
    trajectory_cloud.colors = o3d.utility.Vector3dVector(trajectory_colors)
    o3d.io.write_point_cloud(OUTPUT_TRAJ_PCD_PATH, trajectory_cloud)

    write_trajectory_obj(trajectory_positions, OUTPUT_TRAJ_OBJ_PATH)

    print("Global map built successfully.")
    print(f"Map file          : {OUTPUT_MAP_PATH}")
    print(f"Frame IDs file    : {OUTPUT_FRAMEIDS_PATH}")
    print(f"Trajectory (ply)  : {OUTPUT_TRAJ_PCD_PATH}")
    print(f"Trajectory (obj)  : {OUTPUT_TRAJ_OBJ_PATH}")
    print(f"Total points      : {len(global_cloud.points)}")
    print(f"Frames merged     : {len(dataset)}")


if __name__ == "__main__":
    main()