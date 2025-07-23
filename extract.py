import os
import argparse
import cv2
import numpy as np

# ROS libraries
import rospy
import rosbag
from sensor_msgs.msg import Image, PointCloud2
from cv_bridge import CvBridge
import ros_numpy # for point cloud conversion

# To synchronize topics by their timestamps
import message_filters

# --- CONFIGURATION ---
# TODO: Update the BAG_FILE_PATH to your file's location
BAG_FILE_PATH     = '/share1/car_bags/with_obstacles.bag' # <--- IMPORTANT: SET YOUR BAG FILE PATH
IMAGE_TOPIC       = '/zed_node/left/image_rect_color' # <--- Correct image topic
POINT_CLOUD_TOPIC = '/velodyne_points'                # <--- Correct point cloud topic
OUTPUT_DIR        = 'extracted_data'                  # <--- The directory where data will be saved
# --- END CONFIGURATION ---

# Global variables
bridge = CvBridge()
file_counter = 0

def create_directories():
    """Creates the output directories for images and point clouds."""
    global image_output_dir, pc_output_dir
    image_output_dir = os.path.join(OUTPUT_DIR, 'image_2')
    pc_output_dir = os.path.join(OUTPUT_DIR, 'velodyne')
    os.makedirs(image_output_dir, exist_ok=True)
    os.makedirs(pc_output_dir, exist_ok=True)
    print(f"Saving images to: {image_output_dir}")
    print(f"Saving point clouds to: {pc_output_dir}")

def synchronized_callback(image_msg, pc_msg):
    """
    This function is called for every pair of synchronized image and point cloud messages.
    """
    global file_counter

    # --- Process and Save Image ---
    try:
        # Convert ROS Image message to OpenCV format (BGR8)
        cv_image = bridge.imgmsg_to_cv2(image_msg, "bgr8")
    except Exception as e:
        print(f"Error converting image: {e}")
        return

    image_filename = os.path.join(image_output_dir, f"{file_counter:06d}.png")
    cv2.imwrite(image_filename, cv_image)

    # --- Process and Save Point Cloud ---
    # Convert PointCloud2 message to a NumPy structured array
    pc_data = ros_numpy.point_cloud2.pointcloud2_to_array(pc_msg)
    
    # Extract x, y, z, and intensity fields
    points = np.zeros((pc_data.shape[0], 4), dtype=np.float32)
    points[:, 0] = pc_data['x']
    points[:, 1] = pc_data['y']
    points[:, 2] = pc_data['z']
    if 'intensity' in pc_data.dtype.names:
        points[:, 3] = pc_data['intensity']

    # Create the .bin file
    pc_filename = os.path.join(pc_output_dir, f"{file_counter:06d}.bin")
    points.tofile(pc_filename)

    print(f"Saved frame {file_counter:06d}: {image_filename} | {pc_filename}")
    file_counter += 1


def main():
    """Main function to read the bag and process messages."""
    create_directories()
    
    try:
        bag = rosbag.Bag(BAG_FILE_PATH, 'r')
    except rosbag.bag.BagException as e:
        print(f"Error: Could not open bag file. Please check the path: {BAG_FILE_PATH}")
        return
        
    image_sub = message_filters.Subscriber(IMAGE_TOPIC, Image)
    pc_sub = message_filters.Subscriber(POINT_CLOUD_TOPIC, PointCloud2)
    
    # Synchronize the topics. 'slop' allows for a small time difference (in seconds).
    ts = message_filters.ApproximateTimeSynchronizer([image_sub, pc_sub], queue_size=10, slop=0.2)
    ts.registerCallback(synchronized_callback)
    
    print("Reading messages from the bag... This may take a while.")
    # This loop feeds the messages to the synchronizer
    for topic, msg, t in bag.read_messages(topics=[IMAGE_TOPIC, POINT_CLOUD_TOPIC]):
        if topic == IMAGE_TOPIC:
            image_sub.signalMessage(msg, msg.header.stamp)
        elif topic == POINT_CLOUD_TOPIC:
            pc_sub.signalMessage(msg, msg.header.stamp)
            
    bag.close()
    print(f"Finished processing. Extracted {file_counter} synchronized frames.")


if __name__ == '__main__':
    main()