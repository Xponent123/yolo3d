import rosbag
import yaml
import argparse
from pathlib import Path

def main(bag_path, output_path):
    """
    Reads the /tf_static topic from a ROS bag file and saves all transforms
    to a structured YAML file.
    """
    print(f"Reading from bag file: {bag_path}")
    
    transforms_list = []
    
    try:
        with rosbag.Bag(bag_path, 'r') as bag:
            # The /tf_static topic contains the permanent transforms
            for topic, msg, t in bag.read_messages(topics=['/tf_static']):
                for transform_stamped in msg.transforms:
                    # Create a dictionary for each transform
                    tf_data = {
                        'header': {
                            'stamp': {
                                'secs': transform_stamped.header.stamp.secs,
                                'nsecs': transform_stamped.header.stamp.nsecs
                            },
                            'frame_id': transform_stamped.header.frame_id
                        },
                        'child_frame_id': transform_stamped.child_frame_id,
                        'transform': {
                            'translation': {
                                'x': transform_stamped.transform.translation.x,
                                'y': transform_stamped.transform.translation.y,
                                'z': transform_stamped.transform.translation.z
                            },
                            'rotation': {
                                'x': transform_stamped.transform.rotation.x,
                                'y': transform_stamped.transform.rotation.y,
                                'z': transform_stamped.transform.rotation.z,
                                'w': transform_stamped.transform.rotation.w
                            }
                        }
                    }
                    transforms_list.append(tf_data)
                    print(f"  Found transform: {transform_stamped.header.frame_id} -> {transform_stamped.child_frame_id}")
    except Exception as e:
        print(f"An error occurred: {e}")
        return

    # Save all found transforms to the YAML file
    with open(output_path, 'w') as f:
        yaml.dump(transforms_list, f, default_flow_style=False, sort_keys=False)
        
    print(f"\nSuccessfully extracted {len(transforms_list)} static transforms to '{output_path}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract /tf_static data from a ROS bag file to a YAML file.")
    parser.add_argument('--bag', type=str, required=True, help="Path to the input ROS bag file.")
    parser.add_argument('--output', type=str, default="my_robot_transforms.yaml", help="Path to the output YAML file.")
    args = parser.parse_args()
    
    main(args.bag, args.output)