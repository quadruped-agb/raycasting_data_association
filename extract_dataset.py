import os
import json
import shutil
import cv2
import numpy as np
import open3d as o3d

import rclpy
from rclpy.serialization import deserialize_message
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions

from sensor_msgs.msg import Image, PointCloud2
from sensor_msgs_py import point_cloud2 
from tf2_msgs.msg import TFMessage
from numpy.lib.recfunctions import structured_to_unstructured

# PATHS 
BAG_PATH = "forest_dataset_49/forest_dataset_49_0.mcap"

IMAGE_TOPIC = "/unitree_go2/front_cam/color_image"
POINT_TOPIC = "/unitree_go2/lidar/point_cloud"

TF_PARENT = "map"
TF_CHILD = "unitree_go2/base_link"

# OUTPUT FOLDERS
RAW_IMAGE_DIR = "output/raw_images"
RAW_LIDAR_DIR = "output/raw_lidar"
# FRAME_DIR = "output/frames"

os.makedirs(RAW_IMAGE_DIR, exist_ok=True)
os.makedirs(RAW_LIDAR_DIR, exist_ok=True)
# os.makedirs(FRAME_DIR, exist_ok=True)

# ROS INITIALIZATION
rclpy.init()

storage_options = StorageOptions(
    uri=BAG_PATH,
    storage_id="mcap"
)

converter_options = ConverterOptions(
    input_serialization_format="cdr",
    output_serialization_format="cdr"
)

reader = SequentialReader()
reader.open(storage_options, converter_options) # open rosbag 

images = []
lidars = []
poses = []

# to count number of messages and verify later with .yaml file 
image_counter = 0
lidar_counter = 0
message_counter = 0

print("Reading rosbag...\n")

while reader.has_next():

    topic, data, bag_timestamp = reader.read_next()
    message_counter += 1

    # RGB IMAGE
    if topic == IMAGE_TOPIC:

        msg = deserialize_message(data, Image) # convert into ros image message from binary message 
        
        # extract timestamp
        timestamp = (
            msg.header.stamp.sec +
            msg.header.stamp.nanosec * 1e-9  # sec + nanosec + 10^-9 = seconds stored in float  
        )

        img = np.frombuffer(msg.data, dtype=np.uint8) # convert bytes into Numpy arrays 

        # allm possible image encodings so the code can be generic for any rosbag 
        if msg.encoding == "rgb8":

            img = img.reshape(msg.height, msg.width, 3)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # cnversion to openCY format 

        elif msg.encoding == "bgr8":

            img = img.reshape(msg.height, msg.width, 3)

        elif msg.encoding == "rgba8":

            img = img.reshape(msg.height, msg.width, 4)
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)

        elif msg.encoding == "bgra8":

            img = img.reshape(msg.height, msg.width, 4)
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        else:

            print(f"Unsupported image encoding: {msg.encoding}")
            continue

        filename = f"image_{image_counter:06d}.png"
        filepath = os.path.join(RAW_IMAGE_DIR, filename)

        cv2.imwrite(filepath, img)

        images.append({

            "timestamp": timestamp,
            "filename": filepath

        })

        image_counter += 1

    # LIDAR
    elif topic == POINT_TOPIC:

        msg = deserialize_message(data, PointCloud2)

        timestamp = (
            msg.header.stamp.sec +
            msg.header.stamp.nanosec * 1e-9
        )

        pts_struct = point_cloud2.read_points(  # extract coordinates 
            msg,
            field_names=("x", "y", "z"),
            skip_nans=True
        )

        pts = structured_to_unstructured(  #convert to a numpy array 
            pts_struct
        ).astype(np.float32)

        cloud = o3d.geometry.PointCloud()   #create point cloud using open3d
        cloud.points = o3d.utility.Vector3dVector(
            pts.astype(np.float64)
        )

        filename = f"lidar_{lidar_counter:06d}.pcd"   # save each as .pcd file
        filepath = os.path.join(RAW_LIDAR_DIR, filename)

        o3d.io.write_point_cloud(filepath, cloud)

        lidars.append({

            "timestamp": timestamp,
            "filename": filepath

        })

        lidar_counter += 1

    # ROBOT POSE
    elif topic == "/tf":

        msg = deserialize_message(data, TFMessage) # conversiin from binary to actual ROS TF message 

        for transform in msg.transforms:

            if (                 # since we are concerned with the robots pose, we check whether this transform descirbes robot's pose in the map 
                transform.header.frame_id == TF_PARENT and   
                transform.child_frame_id == TF_CHILD
            ):

                timestamp = (
                    transform.header.stamp.sec +
                    transform.header.stamp.nanosec * 1e-9
                )
                
                # extraction
                t = transform.transform.translation
                q = transform.transform.rotation

                # store 
                poses.append({

                    "timestamp": timestamp,

                    "translation": [
                        t.x,
                        t.y,
                        t.z
                    ],

                    "quaternion": [
                        q.x,
                        q.y,
                        q.z,
                        q.w
                    ]

                })

print("\nFinished reading rosbag.")
print("----------------------------------------")
print(f"Total ROS messages : {message_counter}")
print(f"Saved Images       : {len(images)}")
print(f"Saved LiDAR Frames : {len(lidars)}")
print(f"Stored Poses       : {len(poses)}")
print("----------------------------------------")

# PART 2
# Synchronize LiDAR, RGB and Pose
print("\nSynchronizing LiDAR, RGB and Pose...\n")

def find_closest(data_list, target_time):
    """
    Returns the element whose timestamp is closest to target_time.
    """

    return min(
        data_list,
        key=lambda x: abs(x["timestamp"] - target_time)
    )

dataset = []   # List of dictionaries 

image_errors = []   # stores difference in timestamps of image and lidar 
pose_errors = []    # stores difference in timestamps of pose and lidar 

for frame_id, lidar in enumerate(lidars):

    lidar_time = lidar["timestamp"]

    nearest_image = find_closest(images, lidar_time)
    nearest_pose = find_closest(poses, lidar_time)

    image_diff = abs(
        nearest_image["timestamp"] - lidar_time
    )

    pose_diff = abs(
        nearest_pose["timestamp"] - lidar_time
    )
  
    # conversion in ms 
    image_errors.append(image_diff * 1000) 
    pose_errors.append(pose_diff * 1000)

    dataset.append({

        "frame_id": frame_id,

        "lidar_time": lidar_time,

        "lidar_file": lidar["filename"],

        "image_file": nearest_image["filename"],

        "pose": {

            "translation":
                nearest_pose["translation"],

            "quaternion":
                nearest_pose["quaternion"]

        },

        "image_diff_ms":
            image_diff * 1000,

        "pose_diff_ms":
            pose_diff * 1000

    })

print("----------------------------------------")
print("Synchronization Complete")
print("----------------------------------------")
print(f"Frames synchronized : {len(dataset)}")

print()

print("Image timestamp error")

print(f"Mean : {np.mean(image_errors):.2f} ms")
print(f"Max  : {np.max(image_errors):.2f} ms")

print()

print("Pose timestamp error")

print(f"Mean : {np.mean(pose_errors):.2f} ms")
print(f"Max  : {np.max(pose_errors):.2f} ms")

print("----------------------------------------")

print("\nExample synchronized frame:\n")

example = dataset[0]

print(f"Frame ID     : {example['frame_id']}")
print(f"Image File   : {example['image_file']}")
print(f"LiDAR File   : {example['lidar_file']}")
print(f"Image Error  : {example['image_diff_ms']:.2f} ms")
print(f"Pose Error   : {example['pose_diff_ms']:.2f} ms")

# PART 3
# Save dataset index
print("\nSaving dataset index...\n")

dataset_json = []

for frame in dataset:

    dataset_json.append({

        "frame_id": frame["frame_id"],

        "lidar_time": frame["lidar_time"],

        "image_file": frame["image_file"],

        "lidar_file": frame["lidar_file"],

        "pose": frame["pose"],
        "image_diff_ms": frame["image_diff_ms"],

        "pose_diff_ms": frame["pose_diff_ms"]

    })

index_path = "output/dataset_index.json"

with open(index_path, "w") as f:

    json.dump(
        dataset_json,
        f,
        indent=4
    )

print("----------------------------------------")
print("Dataset index saved successfully.")
print("----------------------------------------")
print(f"Frames indexed : {len(dataset_json)}")
print(f"Index location : {index_path}")
print("----------------------------------------")

rclpy.shutdown()


