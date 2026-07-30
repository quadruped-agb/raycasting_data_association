"""
Converts GLIM's traj_lidar.txt pose output into the same trajectory.csv
format find_frame_pairs.py already expects

WHY THIS EXISTS:
The original trajectory.csv was built from /tf's map->base_link poses,
which were found to be 30+ meters off compared to GLIM's own output.
This script instead reads GLIM's traj_lidar.txt directly (already
correct map->lidar poses), so trajectory.csv reflects the real,
accurate trajectory.

INPUT FORMAT (traj_lidar.txt):
One pose per line, space-separated, no header:
  stamp_sec x y z qx qy qz qw

OUTPUT FORMAT (trajectory.csv):
  stamp_sec,x,y,z,qx,qy,qz,qw
"""

import csv

INPUT_PATH = "forest_49_map_new/map/traj_lidar.txt"
OUTPUT_PATH = "output/trajectory.csv"

rows = []
with open(INPUT_PATH, "r") as f:
    for line_num, line in enumerate(f, start=1):
        line = line.strip()
        if not line:
            continue  # skip blank lines, if any

        parts = line.split()
        if len(parts) != 8:
            print(f"WARNING: line {line_num} has {len(parts)} values, expected 8 -- skipping: {line}")
            continue

        stamp_sec, x, y, z, qx, qy, qz, qw = (float(p) for p in parts)
        rows.append([stamp_sec, x, y, z, qx, qy, qz, qw])

print(f"Read {len(rows)} poses from {INPUT_PATH}")

with open(OUTPUT_PATH, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["stamp_sec", "x", "y", "z", "qx", "qy", "qz", "qw"])
    writer.writerows(rows)

print(f"Saved: {OUTPUT_PATH}")

if rows:
    times = [r[0] for r in rows]
    print(f"Time span: {times[0]:.2f}s to {times[-1]:.2f}s "
          f"({times[-1] - times[0]:.2f}s total)")