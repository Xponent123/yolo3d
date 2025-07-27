#!/usr/bin/env python3
"""
Simple 3D LiDAR Point Cloud Visualization
Creates interactive 3D visualization of LiDAR point cloud using Open3D
"""

import numpy as np
import open3d as o3d
import os

def load_lidar_points(bin_file):
    """Load LiDAR points from binary file"""
    points = np.fromfile(bin_file, dtype=np.float32).reshape(-1, 4)
    return points  # Return x, y, z, intensity

def create_3d_visualization(lidar_points):
    """
    Create interactive 3D visualization of LiDAR point cloud using Open3D
    """
    print(f"\n🎯 Creating 3D visualization with {len(lidar_points)} LiDAR points...")
    
    # Create Open3D PointCloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(lidar_points[:, :3])
    
    # Color the point cloud by height for better visualization
    points_np = np.asarray(pcd.points)
    colors = np.zeros((len(points_np), 3))
    
    # Method 1: Color by height (Z-coordinate)
    z_min, z_max = points_np[:, 2].min(), points_np[:, 2].max()
    z_normalized = (points_np[:, 2] - z_min) / (z_max - z_min)
    colors[:, 0] = z_normalized  # Red component increases with height
    colors[:, 2] = 1 - z_normalized  # Blue component decreases with height
    colors[:, 1] = 0.3  # Add some green for better visibility
    
    # Alternative: Color by intensity
    # intensity_normalized = (lidar_points[:, 3] - lidar_points[:, 3].min()) / (lidar_points[:, 3].max() - lidar_points[:, 3].min())
    # colors[:, :] = intensity_normalized.reshape(-1, 1)  # Grayscale based on intensity
    
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # Add a coordinate frame for reference (X-red, Y-green, Z-blue)
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5.0, origin=[0, 0, 0])
    
    # Create ground plane for reference
    ground_mesh = o3d.geometry.TriangleMesh.create_box(width=100, height=0.1, depth=100)
    ground_mesh.translate([-50, -50, -2])  # Position at approximate ground level
    ground_mesh.paint_uniform_color([0.5, 0.5, 0.5])  # Gray color
    
    geometries = [pcd, coord_frame, ground_mesh]
    
    print(f"\n🚀 Launching 3D LiDAR Point Cloud Visualizer...")
    print("📝 Controls:")
    print("   - Mouse: Rotate view")
    print("   - Scroll: Zoom in/out")
    print("   - Ctrl + Mouse: Pan")
    print("   - Press 'Q' to quit")
    print("   - Press 'H' for more controls")
    print("   - Press 'P' to print current view parameters")
    print("   - Press 'S' to save screenshot")
    
    # Set up visualization parameters for better viewing
    vis = o3d.visualization.Visualizer()
    vis.create_window(
        window_name="LiDAR 3D Point Cloud Visualization", 
        width=1400, 
        height=900,
        left=50,
        top=50
    )
    
    # Add geometries
    for geom in geometries:
        vis.add_geometry(geom)
    
    # Set camera position for better initial view
    view_control = vis.get_view_control()
    view_control.set_front([0.5, 0.5, -0.5])  # Looking from front-top
    view_control.set_up([0, 0, 1])  # Z is up
    view_control.set_lookat([0, 0, 0])  # Look at origin
    view_control.set_zoom(0.3)
    
    # Run the visualization
    vis.run()
    vis.destroy_window()

def main():
    """Main function to run 3D LiDAR point cloud visualization"""
    
    # File path
    bin_file = '/home/jal.parikh/yolo3d/YOLO3D/eval/velodyne/000002.bin'
    
    print("🚀 Starting 3D LiDAR Point Cloud Visualization...")
    
    # Check if file exists
    if not os.path.exists(bin_file):
        print(f"❌ LiDAR file not found: {bin_file}")
        return
    
    print("🔄 Loading LiDAR points...")
    lidar_points = load_lidar_points(bin_file)
    print(f"✅ Loaded {len(lidar_points)} LiDAR points")
    
    # Print LiDAR statistics
    print(f"\n📊 LiDAR Point Cloud Statistics:")
    print(f"   X (forward):  {lidar_points[:, 0].min():.2f} to {lidar_points[:, 0].max():.2f} meters")
    print(f"   Y (left):     {lidar_points[:, 1].min():.2f} to {lidar_points[:, 1].max():.2f} meters")
    print(f"   Z (up):       {lidar_points[:, 2].min():.2f} to {lidar_points[:, 2].max():.2f} meters")
    print(f"   Intensity:    {lidar_points[:, 3].min():.2f} to {lidar_points[:, 3].max():.2f}")
    print(f"   Point range:  {np.linalg.norm(lidar_points[:, :3], axis=1).max():.2f} meters")
    
    # Create and show 3D visualization
    create_3d_visualization(lidar_points)
    
    print(f"\n🎉 3D LiDAR visualization completed!")

if __name__ == "__main__":
    main()
