import os
import numpy as np
import cv2
import rclpy
from rclpy.serialization import deserialize_message
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions

from sensor_msgs.msg import Image, PointCloud2
from sensor_msgs_py import point_cloud2
from tf2_msgs.msg import TFMessage
from numpy.lib.recfunctions import structured_to_unstructured

rclpy.init()

# target timestamps from find_frame_pairs.py's best candidate
TARGET_TIMES = {
    "frame_a": 37.21,
    "frame_b": 39.08,
}

IMAGE_TOPIC = "/unitree_go2/front_cam/color_image"
POINT_TOPIC = "/unitree_go2/lidar/point_cloud"

os.makedirs("output/images", exist_ok=True)
os.makedirs("output/pointclouds", exist_ok=True)
os.makedirs("output/poses", exist_ok=True)

storage_options = StorageOptions(uri=".", storage_id="sqlite3")
converter_options = ConverterOptions(
    input_serialization_format="cdr",
    output_serialization_format="cdr"
)

reader = SequentialReader()
reader.open(storage_options, converter_options)

# pass 1: find the actual closest message to each target time, per topic
# (need to scan the whole bag once per data type since messages aren't
# indexed by time
best_image_msg = {name: (None, float("inf")) for name in TARGET_TIMES}   # (msg_data, time_diff)
best_pc_msg = {name: (None, float("inf")) for name in TARGET_TIMES}
best_pose = {name: (None, float("inf")) for name in TARGET_TIMES}

count = 0
while reader.has_next():
    topic, data, bag_timestamp = reader.read_next()
    count += 1

    if topic == IMAGE_TOPIC:
        msg = deserialize_message(data, Image)
        msg_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        for name, target_t in TARGET_TIMES.items():
            diff = abs(msg_time - target_t)
            if diff < best_image_msg[name][1]:
                best_image_msg[name] = (msg, diff)

    elif topic == POINT_TOPIC:
        msg = deserialize_message(data, PointCloud2)
        msg_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        for name, target_t in TARGET_TIMES.items():
            diff = abs(msg_time - target_t)
            if diff < best_pc_msg[name][1]:
                best_pc_msg[name] = (msg, diff)

    elif topic == "/tf":
        msg = deserialize_message(data, TFMessage)
        for transform in msg.transforms:
            if transform.header.frame_id == "map" and transform.child_frame_id == "unitree_go2/base_link":
                msg_time = transform.header.stamp.sec + transform.header.stamp.nanosec * 1e-9

                for name, target_t in TARGET_TIMES.items():
                    diff = abs(msg_time - target_t)
                    if diff < best_pose[name][1]:
                        best_pose[name] = (transform, diff)

print(f"Scanned {count} total messages")


# pass 2: save everything found, reporting how close the match actually was
for name, target_t in TARGET_TIMES.items():
    print(f"\n--- {name} (target t={target_t}s) ---")

    # image
    img_msg, img_diff = best_image_msg[name]
    if img_msg is None:
        print("  No image found at all!")
    else:
        print(f"  Image match: diff={img_diff*1000:.1f} ms")
        img = np.frombuffer(img_msg.data, dtype=np.uint8)

        if img_msg.encoding == "rgb8":
            img = img.reshape(img_msg.height, img_msg.width, 3)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        elif img_msg.encoding == "bgr8":
            img = img.reshape(img_msg.height, img_msg.width, 3)
        else:
            print(f"  Unsupported encoding: {img_msg.encoding}, skipping image save")
            img = None

        if img is not None:
            path = f"output/images/{name}.png"
            cv2.imwrite(path, img)
            print(f"  Saved: {path}")

    # point cloud
    pc_msg, pc_diff = best_pc_msg[name]
    if pc_msg is None:
        print("  No point cloud found at all!")
    else:
        print(f"  Point cloud match: diff={pc_diff*1000:.1f} ms")
        pts_struct = point_cloud2.read_points(pc_msg, field_names=("x", "y", "z"), skip_nans=True)
        pts = structured_to_unstructured(pts_struct).astype(np.float32)

        import open3d as o3d
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
        path = f"output/pointclouds/{name}.pcd"
        o3d.io.write_point_cloud(path, cloud)
        print(f"  Saved: {path} ({pts.shape[0]} points)")

    # pose
    pose, pose_diff = best_pose[name]
    if pose is None:
        print("  No pose found at all!")
    else:
        print(f"  Pose match: diff={pose_diff*1000:.1f} ms")
        t = pose.transform.translation
        q = pose.transform.rotation
        pose_data = {
            "translation": [t.x, t.y, t.z],
            "quaternion_xyzw": [q.x, q.y, q.z, q.w],
        }
        import json
        path = f"output/poses/{name}_pose.json"
        with open(path, "w") as f:
            json.dump(pose_data, f, indent=2)
        print(f"  Saved: {path}")
        print(f"  Position: {np.round(pose_data['translation'], 3)}")

print("\nDone.")

rclpy.shutdown()