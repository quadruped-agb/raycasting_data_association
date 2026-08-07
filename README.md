# raycasting_data_association

Pipeline for colorizing a LiDAR-built global map using camera imagery, by associating (raycasting) each camera pixel to the 3D LiDAR point it corresponds to.

## What this does

Given a rosbag containing synchronized LiDAR scans, camera images, and robot pose (TF), and a global map already built from that LiDAR data (via GLIM), this pipeline determines which 3D point in the global map each camera pixel "sees," and assigns that pixel's RGB color to the point — producing a photorealistic colorized point cloud instead of a plain LiDAR-only or height-colored map.

## Pipeline order

1. `extract_dataset.py` -- reads the rosbag (`.mcap` format), extracting LiDAR scans, camera images, and robot pose (`/tf`, map -> base_link) with timestamps. Synchronizes each LiDAR frame with its nearest camera image and pose, saving the result as a dataset index JSON.
2. GLIM trajectory resync -- aligns GLIM's estimated trajectory (`traj_lidar.txt`) to each scan's timestamp, producing `dataset_index_glim.json` (frame_id, lidar_file, image_file, pose). Uses linear and spherical (SLERP) interpolation.
3. `global_map_glim.py` -- fuses every LiDAR frame into a single global point cloud in the map frame, using GLIM's poses directly (GLIM has no base_link concept -- poses are already map->lidar). Colors the output by height (Z) for visualization. Produces `global_map_glim.ply` + a trajectory file.
4. `colorize_global_map_voxel_gpu.py` -- for each frame: crops the global map to that frame's camera FOV, raycasts each candidate pixel to find its nearest matching 3D point (within a perpendicular-distance tolerance), and writes that pixel's real camera color into the global map's color buffer. Points never seen by any camera stay grayscale.
   - **Voxelization:** the global map is voxelized at 0.03m to merge near-duplicate points before colorization.
   - **GPU acceleration:** neighbor search uses a PyTorch-based voxel-grid spatial hash (points bucketed into cells, each candidate searches its cell + 26 neighbors), giving roughly a 5x speedup.
   - **Checkpointing:** progress is saved every 20 frames with atomic-safe writes and resume logic (`output/checkpoint/`).

## Requirements

* ROS 2 Humble + `ros-humble-rosbag2-storage-mcap` (for `.mcap` bags)
* Python 3.10, venv with `--system-site-packages`
* `open3d`, `opencv-python`, `numpy`, `scipy`, `matplotlib`
* PyTorch (CUDA build compatible with your GPU -- note: PyTorch 2.13+ dropped support for Pascal-architecture GPUs like the GTX 1080; use 2.4.1/2.7.1 with CUDA 12.1/12.6 wheels if on that hardware)

## Planned

* **Persistent tree ID assignment:** the goal is that once a tree has been observed in the GLIM map, revisiting it later should be recognized as the same tree rather than logged as new. Groundwork exists (`find_frame_pairs.py` detects candidate revisit pairs from the trajectory -- within 2.5m position, 15° yaw, 1-4s gap), but segmentation-driven ID assignment logic is not yet integrated into this pipeline.

## Notes

* `output/` and all generated `.ply` / `.npy` / `.json` files are gitignored -- run the pipeline above to regenerate them locally.
