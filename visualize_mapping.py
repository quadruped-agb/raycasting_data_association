import numpy as np
import cv2
import open3d as o3d

from raycasting import K, T_cam_lidar, R_cam_lidar, t_cam_lidar

image = cv2.imread("output/frame_100_roi_image.png")
height, width = image.shape[:2]

pixel_to_point = np.load("pixel_to_point.npy", allow_pickle=True).item()

cloud = o3d.io.read_point_cloud("output/frame_550_cropped_map.ply")
lidar_points = np.asarray(cloud.points)  # (N, 3), full raw scan, in LIDAR frame


# VISUALIZATION 1: coverage map
# Which pixels got a valid ray -> lidar match, drawn directly on the image.

coverage_img = image.copy()

for (u, v) in pixel_to_point.keys():
    cv2.circle(coverage_img, (u, v), radius=1, color=(0, 255, 0), thickness=-1)

cv2.imwrite("output/images/frame0_coverage_map.png", coverage_img)
print(f"Coverage map saved. Matched pixels drawn: {len(pixel_to_point)}")

# VISUALIZATION 2: reprojection overlay 
# Forward-project the FULL raw lidar scan (not just matched pixels) back
# into image space:
#   p_cam = R_cam_lidar^T @ (p_lidar - t_cam_lidar)      [invert cam->lidar]
#   pixel = K @ (p_cam / p_cam.z)                         [project]
# If extrinsics/intrinsics are correct, these dots should land on real
# tree/ground surfaces in the photo 
R_lidar_cam = R_cam_lidar.T  # inverse of a rotation matrix is its transpose

overlay_img = image.copy()

# depth range for color-mapping (near = one color, far = another)
depths_for_scaling = []

projected_count = 0
behind_camera_count = 0
outside_image_count = 0

projected_pixels = []  #(u, v, depth) for every point that successfully projects

for p_lidar in lidar_points[::10]:
    p_cam = R_lidar_cam @ (p_lidar - t_cam_lidar)

    if p_cam[2] <= 0.05:  # behind or at the camera (or too close to it)
        behind_camera_count += 1
        continue

    pixel_h = K @ (p_cam / p_cam[2])
    u, v = int(round(pixel_h[0])), int(round(pixel_h[1]))
    # v = height-v
    if not (0 <= u < width and 0 <= v < height):
        outside_image_count += 1
        continue

    projected_pixels.append((u, v, p_cam[2]))
    projected_count += 1

print(f"\nReprojection stats:")
print(f"  Total raw lidar points   : {len(lidar_points)}")
print(f"  Projected inside image   : {projected_count}")
print(f"  Behind camera (skipped)  : {behind_camera_count}")
print(f"  Outside image (skipped)  : {outside_image_count}")

if projected_pixels:
    depths = np.array([d for (_, _, d) in projected_pixels])
    min_depth, max_depth = depths.min(), depths.max()

    for (u, v, d) in projected_pixels:
        # normalize depth to 0-255 for a color-mapped dot (near=red, far=blue via JET)
        norm_d = np.clip((d - min_depth) / (max_depth - min_depth + 1e-6), 0, 1)
        color_val = np.uint8([[int((1 - norm_d) * 255)]])  # invert so near=high value
        color = cv2.applyColorMap(color_val, cv2.COLORMAP_JET)[0][0]
        color = (int(color[0]), int(color[1]), int(color[2]))

        cv2.circle(overlay_img, (u, v), radius=1, color=color, thickness=1)

    print(f"  Depth range              : {min_depth:.2f} m to {max_depth:.2f} m")

cv2.imwrite("output/frame100_roi_reprojection_overlay.png", overlay_img)
print("\nReprojection overlay saved.")
print("Check: do the colored dots land on real tree trunks/ground/foliage,")
print("or do they look shifted/rotated relative to what's actually there?")