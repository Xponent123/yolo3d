#!/usr/bin/env python3
"""
Test script to verify calibration file reading and coordinate transformations
"""

import numpy as np
from library.BirdEyeView import create_camera_matrices_from_file, BirdEyeView

def test_calibration_reading():
    """Test reading calibration from actual KITTI file"""
    calib_file = 'eval/calib/000002.txt'
    
    print("Reading calibration from:", calib_file)
    K, P_rect, R_rect, Tr_velo_to_cam = create_camera_matrices_from_file(calib_file)
    
    print("\nCalibration Matrices:")
    print("K (Camera intrinsic):")
    print(K)
    print(f"Shape: {K.shape}")
    
    print("\nP_rect (Projection matrix):")
    print(P_rect)
    print(f"Shape: {P_rect.shape}")
    
    print("\nR_rect (Rectification matrix):")
    print(R_rect)
    print(f"Shape: {R_rect.shape}")
    
    print("\nTr_velo_to_cam (Velodyne to Camera):")
    print(Tr_velo_to_cam)
    print(f"Shape: {Tr_velo_to_cam.shape}")
    
    return K, P_rect, R_rect, Tr_velo_to_cam

def test_coordinate_transformations():
    """Test coordinate transformations with BEV"""
    K, P_rect, R_rect, Tr_velo_to_cam = test_calibration_reading()
    
    # Create BEV with actual calibration
    bev = BirdEyeView(camera_matrix=K, projection_matrix=P_rect, 
                      rectification_matrix=R_rect, velo_to_cam_matrix=Tr_velo_to_cam,
                      bev_width=800, bev_height=800, x_range=(-60, 60), z_range=(0, 120))
    
    # Test with some sample camera coordinates
    test_points_cam = np.array([
        [-17.17, 0.28, 20.34],  # cyclist
        [-2.54, -0.13, 16.71],  # car
        [0.39, 0.54, 11.28]     # car
    ])
    
    print("\n" + "="*60)
    print("COORDINATE TRANSFORMATION TEST")
    print("="*60)
    
    for i, point in enumerate(test_points_cam):
        print(f"\nTest Point {i+1}: {point}")
        
        # Transform to velodyne coordinates
        velo_coords = bev.camera_to_velodyne_coordinates(point.reshape(1, -1))
        print(f"  Velodyne coords: {velo_coords[0]}")
        
        # Get BEV pixel coordinates using camera coordinates
        bev_pixel_cam = bev.world_to_bev_pixel(point.reshape(1, -1), use_velodyne_coords=False)
        print(f"  BEV pixel (camera): {bev_pixel_cam[0]}")
        
        # Get BEV pixel coordinates using velodyne coordinates
        bev_pixel_velo = bev.world_to_bev_pixel(point.reshape(1, -1), use_velodyne_coords=True)
        print(f"  BEV pixel (velodyne): {bev_pixel_velo[0]}")

def test_inverse_transformation():
    """Test if camera to velodyne transformation is working correctly"""
    print("\n" + "="*60)
    print("TESTING VELODYNE ↔ CAMERA TRANSFORMATION")
    print("="*60)
    
    K, P_rect, R_rect, Tr_velo_to_cam = test_calibration_reading()
    
    # Compute inverse transformation
    cam_to_velo = np.linalg.inv(Tr_velo_to_cam)
    
    print("\nTr_velo_to_cam @ cam_to_velo (should be identity):")
    identity_check = Tr_velo_to_cam @ cam_to_velo
    print(identity_check)
    print(f"Is close to identity: {np.allclose(identity_check, np.eye(4))}")
    
    # Test transformation consistency
    test_point_cam = np.array([1.0, 2.0, 10.0, 1.0])  # homogeneous
    
    # Transform cam -> velo -> cam
    velo_point = cam_to_velo @ test_point_cam
    recovered_cam = Tr_velo_to_cam @ velo_point
    
    print(f"\nOriginal camera point: {test_point_cam[:3]}")
    print(f"Velodyne point: {velo_point[:3]}")
    print(f"Recovered camera point: {recovered_cam[:3]}")
    print(f"Transformation consistent: {np.allclose(test_point_cam, recovered_cam)}")

if __name__ == "__main__":
    test_calibration_reading()
    test_coordinate_transformations()
    test_inverse_transformation()
