#!/usr/bin/env python3
"""
Enhanced debug script to validate 3D transformations
"""

import numpy as np
import open3d as o3d

def validate_transformations():
    """Validate the coordinate transformations with known test cases"""
    
    print("=== TRANSFORMATION VALIDATION ===")
    
    # KITTI calibration matrices
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
    
    # Build transformation matrices
    R0_rect_hom = np.eye(4)
    R0_rect_hom[:3, :3] = R0_rect
    Tr_velo_to_cam_hom = np.vstack([Tr_velo_to_cam, [0, 0, 0, 1]])
    Tr_cam_rect_to_velo = np.linalg.inv(R0_rect_hom @ Tr_velo_to_cam_hom)
    
    # Test cases: Objects in front of the camera
    test_cases = [
        {
            'name': 'Car 5m forward, centered',
            'camera_pos': [0.0, 0.0, 5.0],  # [right, down, forward]
            'expected_lidar': [5.0, 0.0, -1.7],  # [forward, left, up] - approximate ground level
        },
        {
            'name': 'Car 10m forward, 2m right',
            'camera_pos': [2.0, 0.0, 10.0],
            'expected_lidar': [10.0, -2.0, -1.7],
        }
    ]
    
    for test in test_cases:
        cam_pos = np.array(test['camera_pos'] + [1.0])  # Homogeneous coordinates
        lidar_pos = Tr_cam_rect_to_velo @ cam_pos
        
        print(f"\n{test['name']}:")
        print(f"  Camera: {test['camera_pos']}")
        print(f"  LiDAR:  {lidar_pos[:3]}")
        print(f"  Expected: {test['expected_lidar']}")
        
        # Check if transformation is reasonable
        error = np.linalg.norm(lidar_pos[:3] - np.array(test['expected_lidar']))
        print(f"  Error magnitude: {error:.2f}m")

def test_rotation_conversion():
    """Test rotation angle conversions"""
    
    print("\n=== ROTATION CONVERSION VALIDATION ===")
    
    test_angles = [0, 45, 90, 135, 180, -45, -90]
    
    for angle_deg in test_angles:
        angle_cam = np.radians(angle_deg)
        
        # Original (incorrect) conversion
        angle_lidar_old = -angle_cam
        
        # New (corrected) conversion - accounting for 90-degree coordinate system offset
        angle_lidar_new = angle_cam + np.pi/2
        
        print(f"Camera: {angle_deg:4.0f}° | Old LiDAR: {np.degrees(angle_lidar_old):6.1f}° | New LiDAR: {np.degrees(angle_lidar_new):6.1f}°")

def create_test_visualization():
    """Create a test visualization with both old and new methods"""
    
    print("\n=== CREATING COMPARISON VISUALIZATION ===")
    
    # Load real LiDAR data
    lidar_file = '/home/jal.parikh/yolo3d/YOLO3D/eval/velodyne/000002.bin'
    lidar_points = np.fromfile(lidar_file, dtype=np.float32).reshape(-1, 4)
    
    # Create point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(lidar_points[:, :3])
    
    # Color by height
    points_np = np.asarray(pcd.points)
    colors = np.zeros((len(points_np), 3))
    z_min, z_max = points_np[:, 2].min(), points_np[:, 2].max()
    z_normalized = (points_np[:, 2] - z_min) / (z_max - z_min)
    colors[:, 0] = z_normalized
    colors[:, 2] = 1 - z_normalized
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    geometries = [pcd]
    
    # Add coordinate frame
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5.0, origin=[0, 0, 0])
    geometries.append(coord_frame)
    
    # Simulate detected boxes at reasonable locations
    test_boxes = [
        {
            'location': [0.0, -1.7, 5.0],  # Camera coordinates: centered, ground level, 5m forward
            'dimensions': [1.5, 1.8, 4.2],  # h, w, l (typical car dimensions)
            'orient': 0.0,  # Facing forward
            'color': (0, 1, 0),  # Green for test box 1
            'label': 'Test Car 1 (0°)'
        },
        {
            'location': [2.0, -1.7, 8.0],  # 2m right, 8m forward
            'dimensions': [1.5, 1.8, 4.2],
            'orient': np.pi/4,  # 45 degrees
            'color': (0, 0, 1),  # Blue for test box 2
            'label': 'Test Car 2 (45°)'
        },
        {
            'location': [-1.5, -1.7, 10.0],  # 1.5m left, 10m forward
            'dimensions': [1.5, 1.8, 4.2],
            'orient': -np.pi/6,  # -30 degrees
            'color': (1, 1, 0),  # Yellow for test box 3
            'label': 'Test Car 3 (-30°)'
        }
    ]
    
    # Transformation matrix
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
    R0_rect_hom = np.eye(4)
    R0_rect_hom[:3, :3] = R0_rect
    Tr_velo_to_cam_hom = np.vstack([Tr_velo_to_cam, [0, 0, 0, 1]])
    Tr_cam_rect_to_velo = np.linalg.inv(R0_rect_hom @ Tr_velo_to_cam_hom)
    
    for i, box in enumerate(test_boxes):
        loc_cam_rect = box['location']
        dim = box['dimensions']
        ry = box['orient']
        
        print(f"\n{box['label']}:")
        print(f"  Camera location: {loc_cam_rect}")
        print(f"  Camera orientation: {np.degrees(ry):.1f}°")
        
        # Transform to LiDAR coordinates
        loc_cam_rect_hom = np.append(loc_cam_rect, 1)
        loc_velo_hom = Tr_cam_rect_to_velo @ loc_cam_rect_hom
        center_velo = loc_velo_hom[:3]
        
        # Adjust center for ground plane (move up by half height)
        height = dim[0]
        center_velo[2] += height / 2.0
        
        # Correct rotation conversion
        yaw_lidar = ry + np.pi/2
        
        # Create rotation matrix
        R_velo = np.array([[np.cos(yaw_lidar), -np.sin(yaw_lidar), 0],
                           [np.sin(yaw_lidar),  np.cos(yaw_lidar), 0],
                           [0,                  0,                 1]])
        
        # Open3D extents: [length, width, height]
        extents = [dim[2], dim[1], dim[0]]
        
        print(f"  LiDAR location: {center_velo}")
        print(f"  LiDAR orientation: {np.degrees(yaw_lidar):.1f}°")
        print(f"  Extents (l,w,h): {extents}")
        
        # Create bounding box
        obb = o3d.geometry.OrientedBoundingBox(center_velo, R_velo, extents)
        obb.color = box['color']
        geometries.append(obb)
    
    print(f"\nVisualization contains:")
    print(f"- Point cloud colored by height (blue=low, red=high)")
    print(f"- Coordinate frame (X=red/forward, Y=green/left, Z=blue/up)")
    print(f"- {len(test_boxes)} test bounding boxes with different orientations")
    
    # Launch visualization
    o3d.visualization.draw_geometries(geometries, 
                                    window_name="Validation Test - 3D Transformations",
                                    width=1200, height=800)

def main():
    print("3D Transformation Validation Script")
    print("==================================")
    
    validate_transformations()
    test_rotation_conversion()
    create_test_visualization()

if __name__ == "__main__":
    main()
