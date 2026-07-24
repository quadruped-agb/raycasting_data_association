import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

data = []
with open("trajectory.csv", "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        data.append([
            float(row["stamp_sec"]), float(row["x"]), float(row["y"]), float(row["z"]),
            float(row["qx"]), float(row["qy"]), float(row["qz"]), float(row["qw"]),
        ])

traj = np.array(data)
times = traj[:, 0]
xs, ys = traj[:, 1], traj[:, 2]
quats = traj[:, 4:8]


def quat_to_yaw(qx, qy, qz, qw):
    siny_cosp = 2 * (qw * qz + qx * qy)
    cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
    return np.arctan2(siny_cosp, cosy_cosp)


yaws = np.array([quat_to_yaw(*q) for q in quats])

fig, ax = plt.subplots(figsize=(10, 8))

# path colored by time (dark = early, bright = late)
norm_times = (times - times.min()) / (times.max() - times.min() + 1e-9)
sc = ax.scatter(xs, ys, c=times, cmap="viridis", s=15, zorder=3)

# connect points with a thin line so the path shape is clear
ax.plot(xs, ys, color="gray", alpha=0.4, linewidth=1, zorder=1)

# heading arrows every Nth point, so we can see which way the camera
# was pointing at various points along the path
N = max(1, len(traj) // 40)  # roughly 40 arrows total regardless of point density
arrow_len = 0.6
for i in range(0, len(traj), N):
    dx = arrow_len * np.cos(yaws[i])
    dy = arrow_len * np.sin(yaws[i])
    ax.arrow(xs[i], ys[i], dx, dy, head_width=0.25, head_length=0.3,
              fc="red", ec="red", alpha=0.7, zorder=2)

# mark start and end clearly
ax.scatter([xs[0]], [ys[0]], color="lime", s=150, marker="^", edgecolor="black",
           zorder=4, label=f"Start (t={times[0]:.1f}s)")
ax.scatter([xs[-1]], [ys[-1]], color="red", s=150, marker="s", edgecolor="black",
           zorder=4, label=f"End (t={times[-1]:.1f}s)")

ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.set_title("Robot trajectory (map frame) - color = time, red arrows = heading (yaw)")
ax.set_aspect("equal", adjustable="box")
ax.legend(loc="best")
ax.grid(True, alpha=0.3)

cbar = fig.colorbar(sc, ax=ax)
cbar.set_label("Time (s)")

plt.tight_layout()
plt.savefig("trajectory_plot.png", dpi=150)
print("Saved: trajectory_plot.png")
plt.show()