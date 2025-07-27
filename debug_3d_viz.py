#!/usr/bin/env python3
"""
Debug script to analyze 3D visualization issues
"""

import numpy as np
import open3d as o3d
import cv2

def debug_coordinate_systems():
    """Debug coordinate system transformations"""
    
    # Define calibration matrices from KITTI dataset
    R0_rect = np.array([
        [9.999239e-01, 9.837760e-03, -7.445048e-03],
        [-9.869795e-03, 9.999421e-01, -4.278459e-03],
        [7.402527e-03, 4.351614e-03, 9.999631e-01]
    ])
    
    Tr_velo_to_cam = np.array([
        [7.533745e-03, -9.999714e-01, -6.166020e-04, -4.069766e-03],
        [1.480249e-02, 7.280733e-04, -9.998902e-01, -7.631618e-02],
        [9.998621e-01, 7.523790e-03, 1.480755e-02, -2.717806e-01]
    ])
    
    print("=== COORDINATE SYSTEM ANALYSIS ===")
    print("\nKITTI Coordinate Systems:")
    print("- Camera: X=right, Y=down, Z=forward")
    print("- LiDAR: X=forward, Y=left, Z=up")
    print("- Rectified camera: aligned with camera but corrected")
    
    # Create transformation matrices
    R0_rect_hom = np.eye(4)
    R0_rect_hom[:3, :3] = R0_rect
    Tr_velo_to_cam_hom = np.vstack([Tr_velo_to_cam, [0, 0, 0, 1]])
    
    print(f"\nR0_rect matrix:\n{R0_rect}")
    print(f"\nTr_velo_to_cam matrix:\n{Tr_velo_to_cam}")
    
    # Calculate the complete transformation
    # Full transformation: LiDAR -> Camera -> Rectified Camera
    Tr_velo_to_cam_rect = R0_rect_hom @ Tr_velo_to_cam_hom
    Tr_cam_rect_to_velo = np.linalg.inv(Tr_velo_to_cam_rect)
    
    print(f"\nTr_velo_to_cam_rect (LiDAR to rectified camera):\n{Tr_velo_to_cam_rect}")
    print(f"\nTr_cam_rect_to_velo (rectified camera to LiDAR):\n{Tr_cam_rect_to_velo}")
    
    # Test transformation with some points
    # Test point in camera coordinates (2m forward, 0.5m right, 0m vertical)
    test_point_cam = np.array([0.5, 0.0, 2.0, 1.0])  # [x_right, y_down, z_forward, 1]
    test_point_lidar = Tr_cam_rect_to_velo @ test_point_cam
    
    print(f"\nTest transformation:")
    print(f"Camera point [x=0.5(right), y=0.0(center), z=2.0(forward)]: {test_point_cam[:3]}")
    print(f"LiDAR point [x(forward), y(left), z(up)]: {test_point_lidar[:3]}")
    
    return Tr_cam_rect_to_velo

def debug_orientation_transformation():
    """Debug orientation angle transformations"""
    
    print("\n=== ORIENTATION ANALYSIS ===")
    
    # In camera coordinates, rotation around Y-axis (yaw)
    # Positive rotation = counterclockwise when looking down Y-axis
    test_angles_cam = [0, np.pi/4, np.pi/2, np.pi, -np.pi/2]
    
    for angle_cam in test_angles_cam:
        # In camera frame: rotation around Y-axis (down)
        # In LiDAR frame: need to convert to rotation around Z-axis (up)
        
        # The transformation relationship:
        # Camera Y-axis (down) corresponds to LiDAR -Z-axis (down)
        # So rotation around camera Y should be rotation around LiDAR -Z
        # But we want rotation around LiDAR +Z, so we negate
        angle_lidar = -angle_cam
        
        print(f"Camera angle (around Y): {np.degrees(angle_cam):6.1f}° -> LiDAR angle (around Z): {np.degrees(angle_lidar):6.1f}°")

def visualize_debug_scene():
    """Create a debug visualization with known objects"""
    
    print("\n=== CREATING DEBUG VISUALIZATION ===")
    
    # Load actual LiDAR data
    lidar_file = '/home/jal.parikh/yolo3d/YOLO3D/eval/velodyne/000002.bin'
    lidar_points = np.fromfile(lidar_file, dtype=np.float32).reshape(-1, 4)
    
    print(f"Loaded {len(lidar_points)} LiDAR points")
    print(f"LiDAR point range:")
    print(f"  X (forward): {lidar_points[:, 0].min():.2f} to {lidar_points[:, 0].max():.2f}")
    print(f"  Y (left):    {lidar_points[:, 1].min():.2f} to {lidar_points[:, 1].max():.2f}")
    print(f"  Z (up):      {lidar_points[:, 2].min():.2f} to {lidar_points[:, 2].max():.2f}")
    
    # Create point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(lidar_points[:, :3])
    
    # Color points by height for better visualization
    points_np = np.asarray(pcd.points)
    colors = np.zeros((len(points_np), 3))
    
    # Color coding: Blue (low) to Red (high)
    z_min, z_max = points_np[:, 2].min(), points_np[:, 2].max()
    z_normalized = (points_np[:, 2] - z_min) / (z_max - z_min)
    colors[:, 0] = z_normalized  # Red component
    colors[:, 2] = 1 - z_normalized  # Blue component
    
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # Create coordinate frame
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=3.0, origin=[0, 0, 0])
    
    # Add some test boxes at known locations in LiDAR coordinates
    geometries = [pcd, coord_frame]
    
    # Test box 1: 5m forward, 2m left, ground level
    center1 = [5.0, 2.0, -1.5]
    extents1 = [4.0, 2.0, 1.5]  # length, width, height
    R1 = np.eye(3)  # No rotation
    
    obb1 = o3d.geometry.OrientedBoundingBox(center1, R1, extents1)
    obb1.color = (0, 1, 0)  # Green
    geometries.append(obb1)
    
    # Test box 2: 10m forward, 0m left, rotated 45 degrees
    center2 = [10.0, 0.0, -1.5]
    extents2 = [4.0, 2.0, 1.5]
    angle2 = np.pi/4  # 45 degrees around Z-axis
    R2 = np.array([[np.cos(angle2), -np.sin(angle2), 0],
                   [np.sin(angle2),  np.cos(angle2), 0],
                   [0,               0,              1]])
    
    obb2 = o3d.geometry.OrientedBoundingBox(center2, R2, extents2)
    obb2.color = (0, 0, 1)  # Blue
    geometries.append(obb2)
    
    print(f"Added test boxes:")
    print(f"  Box 1 (Green): center={center1}, no rotation")
    print(f"  Box 2 (Blue):  center={center2}, 45° rotation")
    
    # Launch visualization
    o3d.visualization.draw_geometries(geometries, 
                                    window_name="Debug 3D Scene",
                                    width=1200, height=800)

def main():
    """Main debug function"""
    print("YOLO3D Debug Script")
    print("==================")
    
    # Debug coordinate systems
    Tr_cam_rect_to_velo = debug_coordinate_systems()
    
    # Debug orientation
    debug_orientation_transformation()
    
    # Show debug visualization
    visualize_debug_scene()
    
    print("\n=== COMMON ISSUES TO CHECK ===")
    print("1. Coordinate system mismatch (Camera vs LiDAR)")
    print("2. Incorrect orientation angle conversion")
    print("3. Wrong dimension ordering (h,w,l vs l,w,h)")
    print("4. Transformation matrix errors")
    print("5. Units mismatch")

if __name__ == "__main__":
    main()
