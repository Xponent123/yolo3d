#!/usr/bin/env python3
"""
Simple tool to analyze the 3D detection output and identify issues
"""

import numpy as np

def analyze_detection_output():
    """Analyze the detection output from the console log"""
    
    print("=== ANALYSIS OF 3D DETECTION RESULTS ===\n")
    
    # Sample data from the console output
    detections = [
        {"box": 1, "cam_loc": [-23.03, -6.47, 70.74], "lidar_loc": [70.938, 22.969, 8.125], "cam_orient": 71.1, "lidar_yaw": -161.1},
        {"box": 2, "cam_loc": [-47.10, -6.78, 98.34], "lidar_loc": [98.523, 47.032, 8.965], "cam_orient": -131.0, "lidar_yaw": 41.0},
        {"box": 10, "cam_loc": [-2.87, -0.86, 14.99], "lidar_loc": [15.249, 2.856, 1.709], "cam_orient": -74.1, "lidar_yaw": -15.9},
        {"box": 11, "cam_loc": [-0.50, -0.17, 10.09], "lidar_loc": [10.36, 0.497, 0.912], "cam_orient": -84.1, "lidar_yaw": -5.9},
    ]
    
    print("Camera Coordinates Analysis:")
    print("X (right): negative values = left side of image")
    print("Y (down): negative values = above image center") 
    print("Z (forward): distance from camera")
    print()
    
    print("LiDAR Coordinates Analysis:")
    print("X (forward): distance ahead of vehicle")
    print("Y (left): positive = left side, negative = right side")
    print("Z (up): height above ground (should be ~-1.5 to 0 for cars)")
    print()
    
    for det in detections:
        print(f"Box {det['box']}:")
        print(f"  Camera: X={det['cam_loc'][0]:6.1f} (side), Y={det['cam_loc'][1]:6.1f} (height), Z={det['cam_loc'][2]:6.1f} (forward)")
        print(f"  LiDAR:  X={det['lidar_loc'][0]:6.1f} (forward), Y={det['lidar_loc'][1]:6.1f} (left), Z={det['lidar_loc'][2]:6.1f} (up)")
        print(f"  Orient: Camera={det['cam_orient']:6.1f}°, LiDAR={det['lidar_yaw']:6.1f}°")
        
        # Check for issues
        issues = []
        if det['lidar_loc'][2] > 3:  # Z too high
            issues.append(f"Z too high ({det['lidar_loc'][2]:.1f}m - cars should be near ground level)")
        if abs(det['lidar_loc'][1]) > 50:  # Too far to the side
            issues.append(f"Too far to side ({det['lidar_loc'][1]:.1f}m)")
        
        if issues:
            print(f"  ISSUES: {'; '.join(issues)}")
        print()
    
    print("=== IDENTIFIED PROBLEMS ===")
    print("1. Z-coordinates (height) are too high - cars floating above ground")
    print("2. This suggests the height adjustment calculation is incorrect")
    print("3. The ground level in KITTI LiDAR is typically around Z = -1.5 to -2.0")
    print()
    print("=== SUGGESTED FIXES ===")
    print("1. Recalculate the height adjustment - may need to subtract rather than add")
    print("2. Verify the KITTI coordinate system conventions")
    print("3. Check if the camera coordinates already include ground-level adjustment")

if __name__ == "__main__":
    analyze_detection_output()
