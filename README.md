# raycasting_data_association (cpu branch)

Pipeline for colorizing a LiDAR-built global map using camera imagery, by associating (raycasting) each camera pixel to the 3D LiDAR point it corresponds to. This branch is the baseline CPU implementation -- no voxelization, no GPU acceleration.

## What this does

Given a rosbag containing synchronized LiDAR scans, camera images, and robot pose (TF), and a global map already built from that LiDAR data (via GLIM), this pipeline determines which 3D point in the global map each camera pixel "sees," and assigns that pixel's RGB color to the point -- producing a photorealistic colorized point cloud instead of a plain LiDAR-only or height-colored map.

## Pipeline order

1. `extract_dataset.py` -- reads the rosbag (`.mcap` format), extracting LiDAR scans, camera images, and robot pose (`/tf`, map -> base_link) with timestamps. Synchronizes each LiDAR frame with its nearest camera image and pose, saving the result as a dataset index JSON.
2. `resync_poses_2.py` -- aligns GLIM's estimated trajectory (`traj_lidar.txt`) to each scan's timestamp, producing `dataset_index_glim.json` (frame_id, lidar_file, image_file, pose). Uses linear and spherical (SLERP) interpolation.
3. `global_map_glim.py` -- fuses every LiDAR frame into a single global point cloud in the map frame, using GLIM's poses directly (GLIM has no base_link concept -- poses are already map->lidar). Colors the output by height (Z) for visualization. Produces `global_map_glim.ply` + a trajectory file.
4. `raycasting_glim.py` -- for each frame: crops the global map to that frame's camera FOV, does a fast vectorized forward projection to find candidate anchor pixels, then raycasts each candidate against a CPU `scipy.spatial.cKDTree` neighborhood search to find its true nearest matching 3D point (within a perpendicular-distance tolerance). Writes that pixel's real camera color into a shared color buffer indexed by the point's position in the global map. Points never seen by any camera stay grayscale.
   * No map-level deduplication -- raycasting runs directly against the raw (un-voxelized) cropped map, so tightly-clustered duplicate points compete individually for the same pixel.
   * Checkpointing: progress is saved periodically with atomic-safe writes and resume logic (`output/checkpoint/`).

## Requirements

* ROS 2 Humble + `ros-humble-rosbag2-storage-mcap` (for `.mcap` bags)
* Python 3.10, venv with `--system-site-packages`
* `open3d`, `opencv-python`, `numpy`, `scipy`, `matplotlib`

## Performance note

This is the slowest branch -- the per-anchor `cKDTree` query is the bottleneck. See the `gpu` branch for a ~5x faster spatial-hash approach, or the `voxelization` branch for the fastest + cleanest (deduplicated) output.

## Planned

* Persistent tree ID assignment: the goal is that once a tree has been observed in the GLIM map, revisiting it later should be recognized as the same tree rather than logged as new. Groundwork exists (`find_frame_pairs.py` detects candidate revisit pairs from the trajectory -- within 2.5m position, 15° yaw, 1-4s gap), but segmentation-driven ID assignment logic is not yet integrated into this pipeline.

## Notes

* `output/` and all generated `.ply` / `.npy` / `.json` files are gitignored -- run the pipeline above to regenerate them locally.
