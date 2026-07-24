"""
resync_poses.py
Replaces the poses in dataset_index.json (originally taken from
/tf map -> unitree_go2/base_link) with GLIM's own LiDAR trajectory
(traj_lidar.txt), interpolated to each LiDAR scan's exact timestamp.

/tf map->base_link turned out to come from a different, unrelated
pose source (not GLIM). traj_lidar.txt is GLIM's actual map<-lidar
trajectory, i.e. the poses GLIM itself used to build its reference map.
Using these instead means our fused map should match GLIM's map.

traj_lidar.txt format (TUM style, one pose per line):
    timestamp tx ty tz qx qy qz qw

 we interpolate: linear for translation, SLERP for rotation,
between the two bracketing keyframe poses. Frames whose lidar_time
falls entirely outside traj_lidar's covered range (the first few
seconds while GLIM is still initializing) are dropped, since there's
nothing to interpolate from

Output:
    dataset_index_glim_2.json  -> same structure as dataset_index.json,
                                 but pose = GLIM's interpolated
                                 map->lidar pose, for every frame
                                 covered by traj_lidar.txt's time range.
"""

import json
import numpy as np

DATASET_INDEX_PATH = "output/dataset_index.json"
TRAJ_LIDAR_PATH = "forest_49_map_new/map/traj_lidar.txt"
OUTPUT_PATH = "output/dataset_index_glim_2.json"

# if two consecutive traj_lidar.txt poses are farther apart in time
# than this, don't interpolate across that gap at all - drop any
# frames that fall inside it instead. Confirmed that
# at least one such gap (45s) involved ~66m of REAL robot motion, not
# a stationary pause - interpolating across it would have 
# compressed hundreds of distinct frames onto a near-static point.
MAX_GAP_SECONDS = 1.0

def slerp(q0, q1, t):
    """Spherical linear interpolation between two xyzw quaternions."""
    q0 = np.asarray(q0, dtype=np.float64)
    q1 = np.asarray(q1, dtype=np.float64)
    dot = np.dot(q0, q1)
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    dot = np.clip(dot, -1.0, 1.0)
    if dot > 0.9995:
        result = q0 + t * (q1 - q0)
        return result / np.linalg.norm(result)
    theta_0 = np.arccos(dot)
    theta = theta_0 * t
    q2 = q1 - q0 * dot
    q2 = q2 / np.linalg.norm(q2)
    return q0 * np.cos(theta) + q2 * np.sin(theta)


def interpolate_pose(poses, times, target_time, max_gap_seconds=1.0):
    """
    Linearly interpolates translation and SLERPs rotation between the
    two traj_lidar.txt poses bracketing target_time.
    Returns None if target_time is outside [times[0], times[-1]], OR
    if the two bracketing poses are more than max_gap_seconds apart.

     traj_lidar.txt can have large gaps (found ~66m of real accumulated motion
    over a 45s gap). Interpolating across a gap like that would
    compress every frame in between onto a near-static point,
    even though the robot actually traveled real distance. Only interpolate 
    across genuinely small (normal
    keyframe-to-keyframe) gaps; drop frames that fall in anything larger.
    """
    #check if timsetamp is before the 1st or after the last recorded glim pose
    if target_time < times[0] or target_time > times[-1]:
        return None
 #find the index where glim's pose is right after the target time
    idx = np.searchsorted(times, target_time)
    if idx == 0:
        return poses[0]
    if idx >= len(times):
        return poses[-1]

    #poses[idx-1] is the pose 
    #just before our target time, and poses[idx] is the pose just after it 
    t0, t1 = times[idx - 1], times[idx]
    #if its greater than defined gap, return none
    if (t1 - t0) > max_gap_seconds:
        return None
    p0, p1 = poses[idx - 1], poses[idx]

    #alpha = 0 if time at target and befre it is same
    alpha = 0.0 if t1 == t0 else (target_time - t0) / (t1 - t0)

    ##translation
    trans = [
        p0["translation"][i] + alpha * (p1["translation"][i] - p0["translation"][i])
        for i in range(3)
    ]
    #rotation
    quat = slerp(p0["quaternion"], p1["quaternion"], alpha)

    return {"translation": trans, "quaternion": quat.tolist()}

#lidar_traj.txt file reading:
def load_traj_lidar(path):
    """Parses a TUM-format trajectory file into a list of pose dicts."""
    poses = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            t = float(parts[0])
            #extract timestamp, quarternion and translation
            tx, ty, tz = (float(v) for v in parts[1:4])
            qx, qy, qz, qw = (float(v) for v in parts[4:8])
            poses.append({
                "timestamp": t,
                "translation": [tx, ty, tz],
                "quaternion": [qx, qy, qz, qw],
            })
    return poses


def main():
    with open(DATASET_INDEX_PATH, "r") as f:
        dataset = json.load(f)

    traj_poses = load_traj_lidar(TRAJ_LIDAR_PATH)
    traj_times = np.array([p["timestamp"] for p in traj_poses])

    print(f"Loaded {len(dataset)} frames from dataset_index.json")
    print(f"Loaded {len(traj_poses)} poses from traj_lidar.txt "
          f"(t = {traj_times.min():.3f} -> {traj_times.max():.3f} s)\n")

    # report any large gaps found in traj_lidar.txt up front, so it's
    # obvious in the output how many frames are about to be dropped
    # and why - not silently interpolated across
    gaps = np.diff(traj_times)
    large_gap_idxs = np.where(gaps > MAX_GAP_SECONDS)[0]
    if len(large_gap_idxs) > 0:
        print(f"Found {len(large_gap_idxs)} gap(s) in traj_lidar.txt larger "
              f"than {MAX_GAP_SECONDS}s - frames inside these will be DROPPED, "
              f"not interpolated:")
        for idx in large_gap_idxs:
            print(f"  gap: t={traj_times[idx]:.3f}s -> t={traj_times[idx+1]:.3f}s "
                  f"({gaps[idx]:.1f}s)")
        print()

    new_dataset = []
    dropped_out_of_range = 0
    dropped_large_gap = 0

#for evry frame in datset_index.json, take that frame's lidar time
    for entry in dataset:
        lidar_time = entry["lidar_time"]
#call interpolate pose function to get GLIM's pose at that exact time
        interp = interpolate_pose(traj_poses, traj_times, lidar_time,
                                   max_gap_seconds=MAX_GAP_SECONDS)
         #frame is dropped if the frame's timestamp falls 
         # before glim's earliest or latest pose
        if interp is None:
            if lidar_time < traj_times.min() or lidar_time > traj_times.max():
                dropped_out_of_range += 1
            else:
                dropped_large_gap += 1
            continue

 #otherwise frame is kept with new pose overwritten
        new_entry = dict(entry)
        new_entry["pose"] = interp
        new_dataset.append(new_entry)

    # re-number frame_id so it's contiguous after dropping frames
    for new_id, entry in enumerate(new_dataset):
        entry["frame_id"] = new_id

    with open(OUTPUT_PATH, "w") as f:
        json.dump(new_dataset, f, indent=4)


    print(f"Frames kept                    : {len(new_dataset)}")
    print(f"Dropped (outside GLIM range)   : {dropped_out_of_range}")
    print(f"Dropped (inside large gap >{MAX_GAP_SECONDS}s): {dropped_large_gap}")
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()