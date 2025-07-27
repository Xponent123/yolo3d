#!/usr/bin/env python3
"""
Advanced LiDAR back-projection with optional camera image overlay
"""

import numpy as np
import cv2
import matplotlib.cm as cm
import os

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
                
                values = np.array([float(x) for x in value.split()])
                
                if key.startswith('P'):
                    calib_data[key] = values.reshape(3, 4)
                elif key == 'R0_rect':
                    calib_data[key] = values.reshape(3, 3)
                elif key == 'Tr_velo_to_cam' or key == 'Tr_imu_to_velo':
                    calib_data[key] = values.reshape(3, 4)
    
    return calib_data

def project_lidar_to_camera(lidar_points, calib_data, camera_id=2):
    """Project LiDAR points to camera image coordinates"""
    
    P = calib_data[f'P{camera_id}']
    R0_rect = calib_data['R0_rect']
    Tr_velo_to_cam = calib_data['Tr_velo_to_cam']
    
    # Convert to homogeneous coordinates
    lidar_homo = np.hstack([lidar_points, np.ones((lidar_points.shape[0], 1))])
    
    # Transform: LiDAR -> Camera -> Rectified -> Image
    Tr_velo_to_cam_homo = np.vstack([Tr_velo_to_cam, [0, 0, 0, 1]])
    cam_points_homo = Tr_velo_to_cam_homo @ lidar_homo.T
    cam_points = cam_points_homo[:3, :].T
    
    # Apply rectification
    R0_rect_homo = np.eye(4)
    R0_rect_homo[:3, :3] = R0_rect
    rect_points_homo = R0_rect_homo @ np.vstack([cam_points.T, np.ones(cam_points.shape[0])])
    rect_points = rect_points_homo[:3, :].T
    
    # Filter points in front of camera
    front_mask = rect_points[:, 2] > 0
    rect_points_front = rect_points[front_mask]
    
    if len(rect_points_front) == 0:
        return np.array([]), np.array([]), np.array([])
    
    # Project to image
    rect_points_homo = np.hstack([rect_points_front, np.ones((rect_points_front.shape[0], 1))])
    image_points_homo = P @ rect_points_homo.T
    image_points = image_points_homo[:2, :] / image_points_homo[2, :]
    image_points = image_points.T
    
    depths = rect_points_front[:, 2]
    
    return image_points, depths, front_mask

def create_projection_overlay(image_points, depths, background_image=None, 
                            image_size=(1242, 375), max_distance=80, point_size=2):
    """Create projection image with optional background"""
    
    width, height = image_size
    
    if background_image is not None:
        # Use provided background image
        image = background_image.copy()
        if image.shape[:2] != (height, width):
            image = cv2.resize(image, (width, height))
    else:
        # Create black background
        image = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Filter valid points
    valid_mask = (
        (image_points[:, 0] >= 0) & 
        (image_points[:, 0] < width) & 
        (image_points[:, 1] >= 0) & 
        (image_points[:, 1] < height)
    )
    
    valid_points = image_points[valid_mask]
    valid_depths = depths[valid_mask]
    
    if len(valid_points) == 0:
        return image, 0
    
    # Color by depth
    normalized_depths = np.clip(valid_depths / max_distance, 0, 1)
    colors = cm.jet(1 - normalized_depths)  # Close=red, far=blue
    colors_rgb = (colors[:, :3] * 255).astype(np.uint8)
    
    # Draw points
    for point, color in zip(valid_points, colors_rgb):
        x, y = int(point[0]), int(point[1])
        cv2.circle(image, (x, y), point_size, color.tolist(), -1)
    
    return image, len(valid_points)

def add_visualization_elements(image, num_points, max_distance=80, scene_name="000002"):
    """Add colorbar, legend, and information"""
    height, width = image.shape[:2]
    
    # Colorbar
    colorbar_width = 25
    colorbar_height = height // 3
    colorbar_x = width - colorbar_width - 15
    colorbar_y = 20
    
    # Draw colorbar
    for i in range(colorbar_height):
        depth_ratio = i / colorbar_height
        color = cm.jet(depth_ratio)
        color_rgb = (np.array(color[:3]) * 255).astype(np.uint8)
        cv2.rectangle(image, 
                     (colorbar_x, colorbar_y + i), 
                     (colorbar_x + colorbar_width, colorbar_y + i + 1), 
                     color_rgb.tolist(), -1)
    
    # Colorbar labels
    cv2.putText(image, f'{max_distance}m', 
                (colorbar_x + colorbar_width + 5, colorbar_y + 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.putText(image, '0m', 
                (colorbar_x + colorbar_width + 5, colorbar_y + colorbar_height), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    # Title and info
    cv2.putText(image, f'LiDAR Projection - Scene {scene_name}', (15, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(image, f'Points: {num_points:,} | Range: 0-{max_distance}m', (15, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
    
    # Legend
    legend_y = height - 80
    cv2.putText(image, 'Color Scale:', (15, legend_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(image, 'Red = Close, Blue = Far', (15, legend_y + 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Add colored dots as examples
    cv2.circle(image, (25, legend_y + 35), 4, (0, 0, 255), -1)  # Red dot
    cv2.putText(image, 'Close', (35, legend_y + 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    cv2.circle(image, (100, legend_y + 35), 4, (255, 0, 0), -1)  # Blue dot
    cv2.putText(image, 'Far', (110, legend_y + 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    return image

def main():
    """Generate enhanced LiDAR back-projection"""
    
    # Configuration
    bin_file = '/home/jal.parikh/yolo3d/YOLO3D/eval/velodyne/000002.bin'
    calib_file = '/home/jal.parikh/yolo3d/YOLO3D/eval/calib/000002.txt'
    camera_image_file = '/home/jal.parikh/yolo3d/YOLO3D/eval/image_2/000002.png'  # Optional
    
    scene_name = "000002"
    max_distance = 80
    point_size = 2
    
    print(f"🚗 KITTI LiDAR Back-Projection Generator")
    print(f"📁 Scene: {scene_name}")
    print("=" * 50)
    
    # Load data
    print("🔄 Loading LiDAR points...")
    lidar_points = load_lidar_points(bin_file)
    print(f"✅ Loaded {len(lidar_points):,} LiDAR points")
    
    print("🔄 Loading calibration data...")
    calib_data = parse_calib_file(calib_file)
    print("✅ Calibration loaded")
    
    # Check for camera image
    background_image = None
    if os.path.exists(camera_image_file):
        print("🔄 Loading camera image...")
        background_image = cv2.imread(camera_image_file)
        print("✅ Camera image loaded")
    else:
        print("ℹ️  No camera image found, using black background")
    
    # Print statistics
    print(f"\n📊 Point Cloud Info:")
    print(f"   Forward range: {lidar_points[:, 0].min():.1f} to {lidar_points[:, 0].max():.1f}m")
    print(f"   Side range:    {lidar_points[:, 1].min():.1f} to {lidar_points[:, 1].max():.1f}m")
    print(f"   Height range:  {lidar_points[:, 2].min():.1f} to {lidar_points[:, 2].max():.1f}m")
    
    # Project points
    print("\n🔄 Projecting to camera view...")
    image_points, depths, front_mask = project_lidar_to_camera(lidar_points, calib_data, camera_id=2)
    
    if len(image_points) > 0:
        print(f"✅ Projected {len(image_points):,} points")
        print(f"📏 Depth range: {depths.min():.1f} to {depths.max():.1f}m")
        
        # Create visualization
        print("🔄 Creating visualization...")
        
        # Version 1: Black background
        image_black, num_points = create_projection_overlay(
            image_points, depths, None, max_distance=max_distance, point_size=point_size
        )
        image_black = add_visualization_elements(image_black, num_points, max_distance, scene_name)
        
        output_black = f'/home/jal.parikh/yolo3d/YOLO3D/lidar_projection_{scene_name}_black.png'
        cv2.imwrite(output_black, image_black)
        print(f"✅ Saved: {output_black}")
        
        # Version 2: Camera overlay (if available)
        if background_image is not None:
            image_overlay, _ = create_projection_overlay(
                image_points, depths, background_image, max_distance=max_distance, point_size=point_size
            )
            image_overlay = add_visualization_elements(image_overlay, num_points, max_distance, scene_name)
            
            output_overlay = f'/home/jal.parikh/yolo3d/YOLO3D/lidar_projection_{scene_name}_overlay.png'
            cv2.imwrite(output_overlay, image_overlay)
            print(f"✅ Saved: {output_overlay}")
        
        # Statistics
        in_bounds_x = np.sum((image_points[:, 0] >= 0) & (image_points[:, 0] < 1242))
        in_bounds_y = np.sum((image_points[:, 1] >= 0) & (image_points[:, 1] < 375))
        in_bounds_both = np.sum(
            (image_points[:, 0] >= 0) & (image_points[:, 0] < 1242) &
            (image_points[:, 1] >= 0) & (image_points[:, 1] < 375)
        )
        
        print(f"\n📈 Projection Statistics:")
        print(f"   Points in X bounds: {in_bounds_x:,}/{len(image_points):,} ({100*in_bounds_x/len(image_points):.1f}%)")
        print(f"   Points in Y bounds: {in_bounds_y:,}/{len(image_points):,} ({100*in_bounds_y/len(image_points):.1f}%)")
        print(f"   Points in image:    {in_bounds_both:,}/{len(image_points):,} ({100*in_bounds_both/len(image_points):.1f}%)")
        
        print(f"\n🎉 Successfully generated LiDAR back-projection for scene {scene_name}!")
        
    else:
        print("❌ No points could be projected!")

if __name__ == "__main__":
    main()
