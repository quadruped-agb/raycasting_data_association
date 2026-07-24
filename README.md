# raycasting_data_association

Pipeline for colorizing a LiDAR-built global map using camera imagery,
by associating (raycasting) each camera pixel to the 3D LiDAR point it
corresponds to.

## What this does

Given a rosbag containing synchronized LiDAR scans, camera images, and
robot pose (TF), and a global map already built from that LiDAR data
(via GLIM), this pipeline determines which 3D point in the global map
each camera pixel "sees," and assigns that pixel's RGB color to the
point -- producing a photorealistic colorized point cloud instead of
a plain LiDAR-only or height-colored map.
Moreover, if a tree has already been seen once in the glim map, it further does id assignment to prevent that tree being marked as a new tree. Segmentation outputs have been incorporated for this aspect. 


## Pipeline order

1. **`extract_dataset.py`** -- reads the rosbag (`.mcap` format),
   extracting LiDAR scans, camera images, and robot pose (`/tf`,
   map -> base_link) with timestamps. Synchronizes each LiDAR frame
   with its nearest camera image and pose, saving the result as a
   dataset index JSON.

2. **GLIM trajectory resync** -- aligns GLIM's estimated trajectory
   (`traj_lidar.txt`) to each scan's timestamp, producing
   `dataset_index_glim.json` (frame_id, lidar_file, image_file, pose). Uses linear and spherical
   interpolation.

4. **`global_map_glim.py`** -- fuses every LiDAR frame into a single
   global point cloud in the map frame, using GLIM's poses directly
   (GLIM has no base_link concept -- poses are already map->lidar).
   Colors the output by height (Z) for visualization. Produces
   `global_map_glim.ply` + a trajectory file.

5. **`colorize_global_map.py`** -- for each frame: crops the global
   map to that frame's camera FOV, raycasts each candidate pixel to
   find its nearest matching 3D point (within a perpendicular-distance
   tolerance), and writes that pixel's real camera color into a
   shared color buffer indexed by the point's position in the global
   map. Points never seen by any camera stay grayscale. Supports
   checkpointing/resuming (`output/checkpoint/`) so long runs can
   survive interruption without losing progress or re-processing
   already-completed frames.

## Requirements

- ROS 2 Humble + `ros-humble-rosbag2-storage-mcap` (for `.mcap` bags)
- Python 3.10, venv with `--system-site-packages`
- `open3d`, `opencv-python`, `numpy`, `scipy`, `matplotlib`

## Notes

- `output/` and all generated `.ply` / `.npy` / `.json` files are
  gitignored -- run the pipeline above to regenerate them locally.

