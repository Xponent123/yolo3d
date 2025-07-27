#!/usr/bin/env python3
"""
Simple LiDAR back-projection without matplotlib display
"""

import numpy as np
import cv2
import matplotlib.cm as cm

def load_lidar_points(bin_file):
    """Load LiDAR points from binary file"""
    points = np.fromfile(bin_file, dtype=np.float32).reshape(-1, 4)
    return points[:, :3]  # Return only x, y, z (ignore intensity)

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

def project_lidar_to_camera(lidar_points, calib_data, camera_id=2):
    """Project LiDAR points to camera image coordinates"""
    
    # Get calibration matrices
    P = calib_data[f'P{camera_id}']  # Projection matrix
    R0_rect = calib_data['R0_rect']  # Rectification matrix
    Tr_velo_to_cam = calib_data['Tr_velo_to_cam']  # LiDAR to camera transformation
    
    # Convert to homogeneous coordinates
    lidar_homo = np.hstack([lidar_points, np.ones((lidar_points.shape[0], 1))])
    
    # Transform from LiDAR to camera coordinates
    # Add homogeneous row to transformation matrix
    Tr_velo_to_cam_homo = np.vstack([Tr_velo_to_cam, [0, 0, 0, 1]])
    cam_points_homo = Tr_velo_to_cam_homo @ lidar_homo.T
    cam_points = cam_points_homo[:3, :].T
    
    # Apply rectification
    R0_rect_homo = np.eye(4)
    R0_rect_homo[:3, :3] = R0_rect
    rect_points_homo = R0_rect_homo @ np.vstack([cam_points.T, np.ones(cam_points.shape[0])])
    rect_points = rect_points_homo[:3, :].T
    
    # Filter out points behind the camera
    front_mask = rect_points[:, 2] > 0
    rect_points_front = rect_points[front_mask]
    
    if len(rect_points_front) == 0:
        return np.array([]), np.array([]), np.array([])
    
    # Project to image coordinates
    rect_points_homo = np.hstack([rect_points_front, np.ones((rect_points_front.shape[0], 1))])
    image_points_homo = P @ rect_points_homo.T
    
    # Convert from homogeneous coordinates
    image_points = image_points_homo[:2, :] / image_points_homo[2, :]
    image_points = image_points.T
    
    # Get depths for coloring
    depths = rect_points_front[:, 2]
    
    return image_points, depths, front_mask

def create_lidar_image(image_points, depths, image_size=(1242, 375), max_distance=80):
    """Create an image with projected LiDAR points"""
    
    width, height = image_size
    image = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Filter points within image bounds
    valid_mask = (
        (image_points[:, 0] >= 0) & 
        (image_points[:, 0] < width) & 
        (image_points[:, 1] >= 0) & 
        (image_points[:, 1] < height)
    )
    
    valid_points = image_points[valid_mask]
    valid_depths = depths[valid_mask]
    
    if len(valid_points) == 0:
        print("No valid points to project!")
        return image
    
    # Normalize depths for coloring
    normalized_depths = np.clip(valid_depths / max_distance, 0, 1)
    
    # Use a colormap (jet colormap: close = red, far = blue)
    colors = cm.jet(1 - normalized_depths)
    colors_rgb = (colors[:, :3] * 255).astype(np.uint8)
    
    # Draw points on image
    for i, (point, color) in enumerate(zip(valid_points, colors_rgb)):
        x, y = int(point[0]), int(point[1])
        # Draw a small circle for each point
        cv2.circle(image, (x, y), 2, color.tolist(), -1)
    
    print(f"Drew {len(valid_points)} valid points on image")
    return image

def add_colorbar_and_info(image, max_distance=80, num_points=0):
    """Add colorbar and information to the image"""
    height, width = image.shape[:2]
    
    # Create colorbar
    colorbar_width = 30
    colorbar_height = height // 2
    colorbar_x = width - colorbar_width - 10
    colorbar_y = 10
    
    # Create depth gradient
    depths_gradient = np.linspace(0, max_distance, colorbar_height)
    normalized_gradient = depths_gradient / max_distance
    colors_gradient = cm.jet(1 - normalized_gradient)
    
    # Draw colorbar
    for i, color in enumerate(colors_gradient):
        color_rgb = (np.array(color[:3]) * 255).astype(np.uint8)
        cv2.rectangle(image, 
                     (colorbar_x, colorbar_y + i), 
                     (colorbar_x + colorbar_width, colorbar_y + i + 1), 
                     color_rgb.tolist(), -1)
    
    # Add text labels for colorbar
    cv2.putText(image, '0m', (colorbar_x + colorbar_width + 5, colorbar_y + 15), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.putText(image, f'{max_distance}m', 
                (colorbar_x + colorbar_width + 5, colorbar_y + colorbar_height), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    # Add title and info
    cv2.putText(image, 'LiDAR Back-Projection to Camera View', (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(image, f'Scene: 000002 | Points: {num_points} | Max depth: {max_distance}m', 
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(image, 'Colors: Red=Close, Blue=Far', (10, height - 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return image

def main():
    """Generate LiDAR back-projection image"""
    
    # File paths
    bin_file = '/home/jal.parikh/yolo3d/YOLO3D/eval/velodyne/000002.bin'
    calib_file = '/home/jal.parikh/yolo3d/YOLO3D/eval/calib/000002.txt'
    output_file = '/home/jal.parikh/yolo3d/YOLO3D/lidar_backproject_simple.png'
    
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
    
    print("\n🔄 Projecting LiDAR points to camera...")
    image_points, depths, front_mask = project_lidar_to_camera(lidar_points, calib_data, camera_id=2)
    
    if len(image_points) > 0:
        print(f"✅ Successfully projected {len(image_points)} points")
        print(f"📏 Depth range: {depths.min():.2f} to {depths.max():.2f} meters")
        
        print("🔄 Creating projection image...")
        image = create_lidar_image(image_points, depths, max_distance=80)
        
        # Add colorbar and information
        image = add_colorbar_and_info(image, max_distance=80, num_points=len(image_points))
        
        # Save the image
        cv2.imwrite(output_file, image)
        print(f"✅ Saved projection image: {output_file}")
        
        # Print image statistics
        in_bounds = np.sum(
            (image_points[:, 0] >= 0) & 
            (image_points[:, 0] < 1242) & 
            (image_points[:, 1] >= 0) & 
            (image_points[:, 1] < 375)
        )
        print(f"📈 Points within image bounds: {in_bounds}/{len(image_points)} ({100*in_bounds/len(image_points):.1f}%)")
        
    else:
        print("❌ No points could be projected to the image!")

if __name__ == "__main__":
    main()
