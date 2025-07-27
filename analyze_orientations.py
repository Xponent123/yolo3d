#!/usr/bin/env python3
"""
Analyze orientation angles to find the correct rotation conversion
"""

import numpy as np

def analyze_orientations():
    """Analyze the orientation patterns from the detection output"""
    
    # Data from the latest run
    detections = [
        {"box": 1, "cam_orient": 71.1, "lidar_yaw": -161.1},
        {"box": 2, "cam_orient": -131.0, "lidar_yaw": 41.0},
        {"box": 3, "cam_orient": -130.1, "lidar_yaw": 40.1},
        {"box": 4, "cam_orient": -124.8, "lidar_yaw": 34.8},
        {"box": 8, "cam_orient": 80.7, "lidar_yaw": -170.7},
        {"box": 9, "cam_orient": -131.8, "lidar_yaw": 41.8},
        {"box": 10, "cam_orient": -74.1, "lidar_yaw": -15.9},
        {"box": 11, "cam_orient": -84.1, "lidar_yaw": -5.9},
    ]
    
    print("=== ORIENTATION ANALYSIS ===")
    print("Current conversion: yaw_lidar = -ry - π/2")
    print()
    print("Box | Camera° | LiDAR°  | Expected° | Diff°")
    print("----|---------|---------|-----------|-------")
    
    for det in detections:
        cam_rad = np.radians(det['cam_orient'])
        current_lidar = det['lidar_yaw']
        
        # Test different conversion formulas
        expected1 = np.degrees(-cam_rad - np.pi/2)  # Current formula
        expected2 = np.degrees(-cam_rad + np.pi/2)  # Alternative 1
        expected3 = np.degrees(cam_rad + np.pi/2)   # Alternative 2
        expected4 = np.degrees(cam_rad - np.pi/2)   # Alternative 3
        
        # Normalize to [-180, 180]
        def normalize_angle(angle):
            while angle > 180:
                angle -= 360
            while angle < -180:
                angle += 360
            return angle
        
        expected1 = normalize_angle(expected1)
        expected2 = normalize_angle(expected2)
        expected3 = normalize_angle(expected3)
        expected4 = normalize_angle(expected4)
        
        diff1 = abs(current_lidar - expected1)
        diff2 = abs(current_lidar - expected2)
        diff3 = abs(current_lidar - expected3)
        diff4 = abs(current_lidar - expected4)
        
        print(f"{det['box']:3d} | {det['cam_orient']:7.1f} | {current_lidar:7.1f} | {expected1:7.1f}   | {diff1:5.1f}")
    
    print()
    print("=== TESTING ALTERNATIVE FORMULAS ===")
    
    for i, det in enumerate(detections[:3]):  # Test first 3 detections
        cam_rad = np.radians(det['cam_orient'])
        current_lidar = det['lidar_yaw']
        
        print(f"\nBox {det['box']} (Camera: {det['cam_orient']:.1f}°, Current LiDAR: {current_lidar:.1f}°):")
        
        formulas = [
            ("Current: -ry - π/2", -cam_rad - np.pi/2),
            ("Alt 1: -ry + π/2", -cam_rad + np.pi/2),
            ("Alt 2: ry + π/2", cam_rad + np.pi/2),
            ("Alt 3: ry - π/2", cam_rad - np.pi/2),
            ("Alt 4: -ry", -cam_rad),
            ("Alt 5: ry", cam_rad),
        ]
        
        for name, result_rad in formulas:
            result_deg = np.degrees(result_rad)
            while result_deg > 180:
                result_deg -= 360
            while result_deg < -180:
                result_deg += 360
            
            diff = abs(current_lidar - result_deg)
            print(f"  {name:15s}: {result_deg:7.1f}° (diff: {diff:5.1f}°)")

if __name__ == "__main__":
    analyze_orientations()
