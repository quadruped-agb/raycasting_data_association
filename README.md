# raycasting_data_association (voxelization branch)

Pipeline for colorizing a LiDAR-built global map using camera imagery, by associating (raycasting) each camera pixel to the 3D LiDAR point it corresponds to. This branch adds map-level voxelization on top of the GPU branch, so raycasting operates on clean, deduplicated points and colors are broadcast back to every original point sharing a voxel.

## What this does

Given a rosbag containing synchronized LiDAR scans, camera images, and robot pose (TF), and a global map already built from that LiDAR data (via GLIM), this pipeline determines which 3D point in the global map each camera pixel "sees," and assigns that pixel's RGB color to the point -- producing a photorealistic colorized point cloud instead of a plain LiDAR-only or height-colored map.

## Files

* **`colorize_global_map_voxel_gpu.py`** -- the entire pipeline in one file. The GPU spatial-hash raycasting is embedded directly in this file (a separate copy from the `gpu` branch's `raycasting_gpu.py`, adapted to operate on voxel points and return `voxel_index` instead of `global_index`). Its own checkpoint directory (`output/checkpoint_voxel_gpu/`) keeps it fully isolated from any other branch's output.

## Pipeline order

1. `extract_dataset.py` -- reads the rosbag (`.mcap` format), extracting LiDAR scans, camera images, and robot pose (`/tf`, map -> base_link) with timestamps. Synchronizes each LiDAR frame with its nearest camera image and pose, saving the result as a dataset index JSON.
2. `resync_poses_2.py` -- aligns GLIM's estimated trajectory (`traj_lidar.txt`) to each scan's timestamp, producing `dataset_index_glim_2.json` (frame_id, lidar_file, image_file, pose). Uses linear and spherical (SLERP) interpolation.
3. `global_map_glim.py` -- fuses every LiDAR frame into a single global point cloud in the map frame, using GLIM's poses directly (GLIM has no base_link concept -- poses are already map->lidar). Colors the output by height (Z) for visualization. Produces `global_map_glim_2.ply` + a trajectory file.
4. `voxelization_raycast.py` -- runs in 5 stages:
   * **Voxelize (once, up front):** the global map is voxelized at 0.03m -- tightly-clustered near-duplicate points are merged into a single representative point (the average position), and every original point is mapped to its voxel for later broadcasting.
   * **Crop:** the voxelized map is cropped to each frame's camera FOV, in the LiDAR-local frame.
   * **Raycast (GPU):** the same voxel-grid spatial-hash raycasting approach as the `gpu` branch, but running on deduplicated voxel points instead of raw points -- so "closest point wins" is chosen among clean targets rather than a pile of near-duplicates.
   * **Colorize:** real camera RGB is sampled per matched pixel and written into a global *voxel*-level color buffer (blended across repeat observations within `COLOR_MATCH_THRESHOLD`, or rejected as a conflict if colors disagree too much).
   * **Broadcast:** each voxel's final color is copied onto every original raw point belonging to that voxel, so the output has the same point count as `global_map_glim_2.ply`, but much better coverage since a whole duplicate cluster now shares one voxel's color instead of only whichever single point happened to win the raycast.
   * **Checkpointing:** progress is saved every 20 frames with resumable state (`output/checkpoint_voxel_gpu/`), including a checkpoint-mismatch guard that would catch a stale checkpoint built against a different voxel count.

## Tuning notes

* `MAX_PERP_DIST = 0.15` and `RAYCAST_MARGIN = 1` -- selected as best-performing after testing different values.
* `COLOR_MATCH_THRESHOLD = 0.20` -- loosened from an earlier value of `0.12`, which was found to reject roughly 2x more legitimate repeat observations than it accepted.
* `VOXEL_SIZE = 0.03` -- map-level deduplication resolution (separate from `GPU_HASH_VOXEL_SIZE = 0.5`, which is just the spatial-hash bucket size used for the raycasting search, not the deduplication).

## Requirements

* ROS 2 Humble + `ros-humble-rosbag2-storage-mcap` (for `.mcap` bags)
* Python 3.10, venv with `--system-site-packages`
* `open3d`, `opencv-python`, `numpy`, `matplotlib`
* PyTorch (CUDA build compatible with your GPU -- note: PyTorch 2.13+ dropped support for Pascal-architecture GPUs like the GTX 1080; use 2.4.1/2.7.1 with CUDA 12.1/12.6 wheels if on that hardware). Falls back to CPU tensors automatically if CUDA isn't available, just without the GPU speedup.

## Performance note

Fastest and cleanest branch: voxelizing the map first means raycasting competes among deduplicated targets rather than a pile of near-duplicate points, and the final broadcast step means a whole duplicate cluster shares one color instead of only whichever single point happened to win.

## Planned

* Persistent tree ID assignment: the goal is that once a tree has been observed in the GLIM map, revisiting it later should be recognized as the same tree rather than logged as new. Groundwork exists (`find_frame_pairs.py` detects candidate revisit pairs from the trajectory -- within 2.5m position, 15° yaw, 1-4s gap), but segmentation-driven ID assignment logic is not yet integrated into this pipeline.

## Notes

* `output/` and all generated `.ply` / `.npy` / `.json` files are gitignored -- run the pipeline above to regenerate them locally.
