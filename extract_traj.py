import rclpy
from rclpy.serialization import deserialize_message
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from tf2_msgs.msg import TFMessage

rclpy.init()

from rosbag2_py import StorageOptions, ConverterOptions

storage_options = StorageOptions(
    uri="/home/administrator/Downloads/data_association/forest_dataset_49/forest_dataset_49_0-001.mcap",
    storage_id="mcap"
)

converter_options = ConverterOptions(
    input_serialization_format="cdr",
    output_serialization_format="cdr"
)

reader = SequentialReader()
reader.open(storage_options, converter_options)

trajectory = []  # list of (timestamp_sec, x, y, z, qx, qy, qz, qw)

count = 0

while reader.has_next():
    topic, data, timestamp = reader.read_next()

    if topic == "/tf":
        msg = deserialize_message(data, TFMessage)

        for transform in msg.transforms:
            if transform.header.frame_id == "map" and transform.child_frame_id == "unitree_go2/base_link":
                t = transform.transform.translation
                q = transform.transform.rotation

                # timestamp from the message header 
                # recv timestamp for matching against image/lidar frames later)
                stamp = transform.header.stamp
                stamp_sec = stamp.sec + stamp.nanosec * 1e-9

                trajectory.append((stamp_sec, t.x, t.y, t.z, q.x, q.y, q.z, q.w))
                count += 1

print(f"Collected {count} map->base_link poses")

# save full trajectory for plotting/frame selection
import csv
with open("trajectory.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["stamp_sec", "x", "y", "z", "qx", "qy", "qz", "qw"])
    writer.writerows(trajectory)

print("Saved: trajectory.csv")


import numpy as np
traj = np.array(trajectory)
xs, ys = traj[:, 1], traj[:, 2]

print(f"\nX range: {xs.min():.2f} to {xs.max():.2f} m")
print(f"Y range: {ys.min():.2f} to {ys.max():.2f} m")
print(f"Time span: {traj[0,0]:.1f}s to {traj[-1,0]:.1f}s ({traj[-1,0]-traj[0,0]:.1f}s total)")

# check for potential revisits: for each pose, find the CLOSEST-in-space
# pose that is FAR AWAY in time (i.e. the robot came back near here later,
# not just the previous timestep which is close)
print("\nSearching for potential revisits (same area, different time)...")

positions = traj[:, 1:3]  # x, y only
times = traj[:, 0]

MIN_TIME_GAP = 20.0   # seconds - must be wellseparated in time to count as a "revisit" not just adjacent frames
MAX_SPATIAL_DIST = 1.0  # meters, how close counts as "same area"

#Condition 1: time separation: time_gap >= 20.0 seconds
#Condition 2 :spatial proximity: distance < 1.0 meters, computed as straight-line Euclidean distance between the two (x, y, z) positions. 
# This defines how close counts as "the same spot"
revisit_candidates = []

for i in range(len(positions)):
    for j in range(i + 1, len(positions)):
        time_gap = times[j] - times[i]
        if time_gap < MIN_TIME_GAP:
            continue
        dist = np.linalg.norm(positions[j] - positions[i])
        if dist < MAX_SPATIAL_DIST:
            revisit_candidates.append((i, j, times[i], times[j], time_gap, dist))

print(f"Found {len(revisit_candidates)} candidate index pairs suggesting a revisit "
      f"(same spot within {MAX_SPATIAL_DIST}m, {MIN_TIME_GAP}s+ apart)")

if revisit_candidates:
    step = max(1, len(revisit_candidates) // 10)
    print("\nSample candidates (index_i, index_j, time_i, time_j, time_gap, spatial_dist):")
    for c in revisit_candidates[::step][:10]:
        print(f"  t={c[2]:.1f}s <-> t={c[3]:.1f}s | gap={c[4]:.1f}s | dist={c[5]:.2f}m")
else:
    print("\nNo revisits found with these thresholds - this bag may be a single "
          "continuous pass with no loop-back")

rclpy.shutdown()