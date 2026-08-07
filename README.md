# raycasting_data_association

Pipeline for colorizing a LiDAR-built global map using camera imagery, by associating (raycasting) each camera pixel to the 3D LiDAR point it corresponds to.

## What this does

Given a rosbag containing synchronized LiDAR scans, camera images, and robot pose (TF), and a global map already built from that LiDAR data (via GLIM), this pipeline determines which 3D point in the global map each camera pixel "sees," and assigns that pixel's RGB color to the point -- producing a photorealistic colorized point cloud instead of a plain LiDAR-only or height-colored map.

## Pipeline order (shared across all branches)

1. `extract_dataset.py` -- reads the rosbag (`.mcap` format), extracting LiDAR scans, camera images, and robot pose (`/tf`, map -> base_link) with timestamps. Synchronizes each LiDAR frame with its nearest camera image and pose, saving the result as a dataset index JSON.
2. `resync_poses_2.py` -- aligns GLIM's estimated trajectory (`traj_lidar.txt`) to each scan's timestamp, producing `dataset_index_glim.json` (frame_id, lidar_file, image_file, pose). Uses linear and spherical (SLERP) interpolation.
3. `global_map_glim.py` -- fuses every LiDAR frame into a single global point cloud in the map frame, using GLIM's poses directly (GLIM has no base_link concept -- poses are already map->lidar). Colors the output by height (Z) for visualization. Produces `global_map_glim.ply` + a trajectory file.
4. Colorization -- the actual crop -> raycast -> colorize step. This is where the branches diverge (see below); each branch implements this step differently, trading off speed against implementation complexity.

## Branches

This repo has three branches, each a different implementation of step 4 above. They share the same input data and produce the same kind of output (a colorized `.ply` with the same point count as the input map), but differ in raycasting speed and map preprocessing:

* **`cpu`** -- navigate to this branch for the baseline implementation: raycasting via a CPU `scipy.spatial.cKDTree` neighbor search, no GPU acceleration, no map-level deduplication. Slowest, but has no PyTorch/CUDA dependency.
* **`gpu`** -- navigate to this branch for the GPU-accelerated version: raycasting via a PyTorch voxel-grid spatial hash instead of `cKDTree`, about ~5x faster than `cpu`. Still no map-level deduplication -- raycasts against the raw, un-voxelized map.
* **`voxelization`** -- navigate to this branch for the fastest and cleanest version: adds map-level voxelization (0.03m) on top of the `gpu` branch's raycasting approach, so raycasting competes among deduplicated points instead of near-duplicate clusters, then broadcasts each voxel's final color back to every original point sharing that voxel.

## Requirements (all branches)

* ROS 2 Humble + `ros-humble-rosbag2-storage-mcap` (for `.mcap` bags)
* Python 3.10, venv with `--system-site-packages`
* `open3d`, `opencv-python`, `numpy`, `matplotlib`
* `scipy` -- `cpu` branch only
* PyTorch (CUDA build compatible with your GPU -- note: PyTorch 2.13+ dropped support for Pascal-architecture GPUs like the GTX 1080; use 2.4.1/2.7.1 with CUDA 12.1/12.6 wheels if on that hardware) -- `gpu` and `voxelization` branches only
