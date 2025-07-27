#!/usr/bin/env python3
"""
Test script to debug coordinate transformations without Open3D dependency
"""

import numpy as np
import cv2
import torch
import sys
import os

# Add the current directory to path for imports
sys.path.append('/home/jal.parikh/yolo3d/YOLO3D')

from lidar_3d_visualization import detect_2d_objects, detect_3d_objects, parse_calib_file, load_lidar_points

def analyze_transformations():
    """Analyze the coordinate transformations without 3D visualization"""
    
    # File paths
    bin_file = '/home/jal.parikh/yolo3d/YOLO3D/eval/velodyne/000002.bin'
    calib_file = '/home/jal.parikh/yolo3d/YOLO3D/eval/camera_cal/calib_cam_to_cam.txt'
    image_file = '/home/jal.parikh/yolo3d/YOLO3D/eval/image_2/000002.png'
    
    print("🔄 Loading LiDAR points...")
    lidar_points = load_lidar_points(bin_file)
    print(f"✅ Loaded {len(lidar_points)} LiDAR points")
    
    print("🔄 Parsing calibration file...")
    calib_data = parse_calib_file(calib_file)
    print("✅ Calibration data loaded")
    
    # Print LiDAR statistics
    print(f"\n📊 LiDAR Point Cloud Statistics:")
    print(f"   X (forward): {lidar_points[:, 0].min():.2f} to {lidar_points[:, 0].max():.2f}m")
    print(f"   Y (left):    {lidar_points[:, 1].min():.2f} to {lidar_points[:, 1].max():.2f}m")
    print(f"   Z (up):      {lidar_points[:, 2].min():.2f} to {lidar_points[:, 2].max():.2f}m")
    
    print("\n🔄 Running 2D object detection...")
    dets_2d = detect_2d_objects(image_file)
    print(f"✅ Found {len(dets_2d)} 2D detections")
    
    print("\n🔄 Running 3D object detection...")
    try:
        detected_boxes = detect_3d_objects(image_file, calib_file)
        print(f"✅ Successfully processed {len(detected_boxes)} 3D detections")
        
        print(f"\n📦 3D Detection Analysis:")
        for i, box in enumerate(detected_boxes):
            loc = box['location']
            dim = box['dimensions']
            orient = box['orient']
            print(f"   {i+1}. {box['class']}:")
            print(f"      Camera location: ({loc[0]:.2f}, {loc[1]:.2f}, {loc[2]:.2f})")
            print(f"      Dimensions (h,w,l): ({dim[0]:.2f}, {dim[1]:.2f}, {dim[2]:.2f})")
            print(f"      Orientation: {orient:.3f} rad ({np.degrees(orient):.1f}°)")
            
            # Transform to LiDAR coordinates manually
            R0_rect = calib_data['R0_rect']
            Tr_velo_to_cam = calib_data['Tr_velo_to_cam']
            
            R0_rect_hom = np.eye(4)
            R0_rect_hom[:3, :3] = R0_rect
            Tr_velo_to_cam_hom = np.vstack([Tr_velo_to_cam, [0, 0, 0, 1]])
            Tr_cam_rect_to_velo = np.linalg.inv(R0_rect_hom @ Tr_velo_to_cam_hom)
            
            loc_cam_rect_hom = np.append(loc, 1)
            loc_velo_hom = Tr_cam_rect_to_velo @ loc_cam_rect_hom
            center_velo = loc_velo_hom[:3]
            center_velo[2] += dim[0] / 2.0  # Add half height
            
            print(f"      LiDAR location: ({center_velo[0]:.2f}, {center_velo[1]:.2f}, {center_velo[2]:.2f})")
            
            if i >= 4:  # Only show first 5 detections
                break
                
    except Exception as e:
        print(f"⚠️  3D detection failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    analyze_transformations()
