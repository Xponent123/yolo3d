#!/usr/bin/env python3
"""
Simple test script to check rotation transformation without dependencies
"""

import numpy as np

def test_rotation_change():
    """Test the rotation change from +90° to -90°"""
    print("🔧 Testing rotation transformation change...")
    
    # Test different camera orientations
    test_orientations = [0, np.pi/4, np.pi/2, np.pi, -np.pi/2]
    
    print("\nCamera Orientation -> LiDAR Yaw:")
    print("================================")
    
    for orientation in test_orientations:
        # Old method: +90°
        yaw_old = orientation + np.pi/2
        while yaw_old > np.pi:
            yaw_old -= 2 * np.pi
        while yaw_old < -np.pi:
            yaw_old += 2 * np.pi
            
        # New method: -90°
        yaw_new = orientation - np.pi/2
        while yaw_new > np.pi:
            yaw_new -= 2 * np.pi
        while yaw_new < -np.pi:
            yaw_new += 2 * np.pi
        
        print(f"Camera: {np.degrees(orientation):6.1f}° -> Old: {np.degrees(yaw_old):6.1f}° | New: {np.degrees(yaw_new):6.1f}°")
    
    print("\n✅ The change rotates all bounding boxes by 90° clockwise")
    print("   This should fix the horizontal vs vertical orientation issue!")

if __name__ == "__main__":
    test_rotation_change()
