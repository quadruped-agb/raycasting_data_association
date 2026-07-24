#Two nearby-in-time moments where the robot hasn't 
# moved/turned far, so the same trees plausibly stay in frame
import numpy as np
import csv

#load trajectory
data = []
with open("/home/administrator/Downloads/data_association/trajectory.csv", "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        data.append([
            float(row["stamp_sec"]), float(row["x"]), float(row["y"]), float(row["z"]),
            float(row["qx"]), float(row["qy"]), float(row["qz"]), float(row["qw"]),
        ])

traj = np.array(data)
times = traj[:, 0]
positions = traj[:, 1:4]
quats = traj[:, 4:8]  # qx, qy, qz, qw

def quat_to_yaw(qx, qy, qz, qw):
    """Extract yaw (rotation about Z) from a quaternion."""
    siny_cosp = 2 * (qw * qz + qx * qy)
    cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
    return np.arctan2(siny_cosp, cosy_cosp)


yaws = np.array([quat_to_yaw(*q) for q in quats])


# search for candidate pairs: close in position AND heading, but separated
# enough in time to give a different point (not just
# consecutive near-identical frames)


#Minimum 1 second AND maximum 4 seconds
#Within 2.5m 

MIN_TIME_GAP = 1.0     # seconds - want a real change, not near-duplicate frames
MAX_TIME_GAP = 4.0     # seconds - too much time = too much movement, trees drop out of FOV
MAX_POSITION_DIST = 2.5   # meters
MAX_YAW_DIFF_DEG = 15.0   # degrees - keep camera pointed roughly the same direction

max_yaw_diff_rad = np.radians(MAX_YAW_DIFF_DEG)

candidates = []

for i in range(len(traj)):
    for j in range(i + 1, len(traj)):
        time_gap = times[j] - times[i]

        if time_gap < MIN_TIME_GAP:
            continue
        if time_gap > MAX_TIME_GAP:
            break  # times are sorted, no point checking further j for this i

        pos_dist = np.linalg.norm(positions[j] - positions[i])
        if pos_dist > MAX_POSITION_DIST:
            continue

        yaw_diff = abs(np.arctan2(np.sin(yaws[j] - yaws[i]), np.cos(yaws[j] - yaws[i])))
        if yaw_diff > max_yaw_diff_rad:
            continue

        candidates.append({
            "t1": times[i], "t2": times[j],
            "time_gap": time_gap,
            "pos_dist": pos_dist,
            "yaw_diff_deg": np.degrees(yaw_diff),
        })

print(f"Found {len(candidates)} candidate pairs "
      f"(position <{MAX_POSITION_DIST}m, yaw <{MAX_YAW_DIFF_DEG} deg, "
      f"time gap {MIN_TIME_GAP}-{MAX_TIME_GAP}s)")

if not candidates:
    print("\nNo candidates found with these thresholds. Try loosening "
          "MAX_POSITION_DIST, MAX_YAW_DIFF_DEG, or MAX_TIME_GAP.")
else:
    # sort by time_gap descending: prefer the most time-separated (most
    # different point) among the valid candidates, for the
    # strongest test of ID persistence
    candidates.sort(key=lambda c: -c["time_gap"])

    print("\nTop candidates (largest time gap first, still within position/yaw limits):")
    print(f"{'t1 (s)':>8s} {'t2 (s)':>8s} {'gap (s)':>8s} {'dist (m)':>9s} {'yaw diff (deg)':>15s}")
    for c in candidates[:15]:
        print(f"{c['t1']:8.2f} {c['t2']:8.2f} {c['time_gap']:8.2f} "
              f"{c['pos_dist']:9.2f} {c['yaw_diff_deg']:15.2f}")

    best = candidates[0]
    print(f"\nBest candidate pair: t1={best['t1']:.2f}s, t2={best['t2']:.2f}s "
          f"(gap={best['time_gap']:.2f}s, dist={best['pos_dist']:.2f}m, "
          f"yaw_diff={best['yaw_diff_deg']:.1f} deg)")
