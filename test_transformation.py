#!/usr/bin/env python3
"""
Test script to check transformation results without needing dependencies
"""

import numpy as np
import sys
sys.path.append('/home/jal.parikh/yolo3d/YOLO3D')

# Only import what we need for 3D detection
import cv2
import torch
from torchvision.models import resnet18
from script.Dataset import generate_bins, DetectedObject
from script import ClassAverages
from script.Model import ResNet18
from library.Math import calc_location

class Bbox:
    def __init__(self, box_2d, class_):
        self.box_2d = box_2d
        self.detected_class = class_

def parse_calib_file(calib_file):
    """Parse KITTI calibration file"""
    calib_data = {}
    
    with open(calib_file, 'r') as f:
        for line in f:
            line = line.strip()
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip()
                value = value.strip()
                
                # Convert string values to numpy arrays
                values = np.array([float(x) for x in value.split()])
                
                if key.startswith('P'):
                    # Projection matrix is 3x4
                    calib_data[key] = values.reshape(3, 4)
                elif key == 'R0_rect':
                    # Rectification matrix is 3x3
                    calib_data[key] = values.reshape(3, 3)
                elif key == 'Tr_velo_to_cam' or key == 'Tr_imu_to_velo':
                    # Transformation matrix is 3x4
                    calib_data[key] = values.reshape(3, 4)
    
    return calib_data

def create_3d_bounding_box_in_lidar_frame(location_cam, dimensions, orientation, calib_data):
    """Test the transformation with corrected rotation"""
    
    # Get transformation matrices
    R0_rect = calib_data['R0_rect']
    Tr_velo_to_cam = calib_data['Tr_velo_to_cam']
    
    # Create the full transformation matrix from rectified camera to LiDAR
    R0_rect_hom = np.eye(4)
    R0_rect_hom[:3, :3] = R0_rect
    Tr_velo_to_cam_hom = np.vstack([Tr_velo_to_cam, [0, 0, 0, 1]])
    Tr_cam_rect_to_velo = np.linalg.inv(R0_rect_hom @ Tr_velo_to_cam_hom)
    
    # Transform the center location from camera to LiDAR coordinates
    loc_cam_rect_hom = np.append(location_cam, 1)
    loc_velo_hom = Tr_cam_rect_to_velo @ loc_cam_rect_hom
    center_velo = loc_velo_hom[:3]
    
    # Height adjustment
    height = dimensions[0]
    center_velo[2] += height / 2.0
    
    # Dimension mapping: [width, length, height]
    extents = [dimensions[1], dimensions[2], dimensions[0]]
    
    # CORRECTED rotation: subtract 90 degrees instead of adding
    yaw_lidar = orientation - np.pi/2
    
    # Normalize angle to [-π, π]
    while yaw_lidar > np.pi:
        yaw_lidar -= 2 * np.pi
    while yaw_lidar < -np.pi:
        yaw_lidar += 2 * np.pi
    
    print(f"  -> Camera orient: {orientation:.3f} rad ({np.degrees(orientation):.1f}°)")
    print(f"  -> LiDAR yaw: {yaw_lidar:.3f} rad ({np.degrees(yaw_lidar):.1f}°)")
    print(f"  -> LiDAR location: {center_velo}")
    print(f"  -> Extents (w,l,h): {extents}")
    
    return center_velo, yaw_lidar, extents

def main():
    """Test the corrected transformation"""
    print("🔧 Testing corrected transformation logic...")
    
    # File paths
    calib_file = '/home/jal.parikh/yolo3d/YOLO3D/eval/camera_cal/calib_cam_to_cam.txt'
    
    # Load calibration
    print("🔄 Loading calibration...")
    calib_data = parse_calib_file(calib_file)
    
    # Test with some example values (typical car detection)
    print("\n📦 Testing transformation with sample detection:")
    location_cam = np.array([2.5, 1.5, 15.0])  # Camera coordinates
    dimensions = np.array([1.5, 1.6, 3.8])     # Height, width, length
    orientation = 0.5                           # Some rotation in radians
    
    center_velo, yaw_lidar, extents = create_3d_bounding_box_in_lidar_frame(
        location_cam, dimensions, orientation, calib_data
    )
    
    print("\n✅ Transformation test completed!")
    print(f"Rotation change: {np.degrees(orientation):.1f}° (camera) -> {np.degrees(yaw_lidar):.1f}° (LiDAR)")

if __name__ == "__main__":
    main()
