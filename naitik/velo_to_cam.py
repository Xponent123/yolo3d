import yaml
import numpy as np
from scipy.spatial.transform import Rotation as R

def get_transform_matrix(transform):
    """Convert a single transform (dict) into a 4x4 transformation matrix."""
    t = transform['transform']['translation']
    r = transform['transform']['rotation']
    
    # Convert quaternion to rotation matrix
    quat = [r['x'], r['y'], r['z'], r['w']]
    rot = R.from_quat(quat).as_matrix()
    
    trans = np.eye(4)
    trans[:3, :3] = rot
    trans[:3, 3] = [t['x'], t['y'], t['z']]
    return trans

def load_tr_velo_to_cam(yaml_path):
    """Load a YAML file with transforms and return the 4x4 Tr_velo_to_cam matrix."""
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)

    # Collect relevant transforms
    tf_dict = {}
    for tf in data:
        parent = tf['header']['frame_id']
        child = tf['child_frame_id']
        tf_dict[(parent, child)] = tf

    # Build the chain: velo_link → imu_link → camera_color_left
    T_velo_to_imu = get_transform_matrix(tf_dict[('imu_link', 'velo_link')])
    T_imu_to_cam = get_transform_matrix(tf_dict[('imu_link', 'camera_color_left')])

    # Tr_velo_to_cam = T_cam_from_imu @ T_imu_from_velo
    Tr_velo_to_cam = np.linalg.inv(T_imu_to_cam) @ T_velo_to_imu
    return Tr_velo_to_cam