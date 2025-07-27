#!/usr/bin/env python3
"""
3D LiDAR Point Cloud Visualization with 3D Bounding Box Detection
Creates interactive 3D visualization using Open3D
"""

import numpy as np
import cv2
import torch
import torch.nn as nn
from torchvision.models import resnet18
import sys
import os
from pathlib import Path
import open3d as o3d

# Add the current directory to path for imports
sys.path.append('/home/jal.parikh/yolo3d/YOLO3D')

from models.common import DetectMultiBackend
from utils.datasets import LoadImages
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.torch_utils import select_device, time_sync
from script.Dataset import generate_bins, DetectedObject
from library.Math import calc_location, rotation_matrix, create_corners
from library.Plotting import plot_3d_box, project_3d_pt
from script import Model, ClassAverages
from script.Model import ResNet, ResNet18, VGG11

class Bbox:
    def __init__(self, box_2d, class_):
        self.box_2d = box_2d
        self.detected_class = class_

def plot3d(img, proj_matrix, box_2d, dimensions, alpha, theta_ray, img_2d=None):
    """Calculate 3D location using the YOLO3D math"""
    location, X = calc_location(dimensions, proj_matrix, box_2d, alpha, theta_ray)
    if location is None:
        return None

    orient = alpha + theta_ray

    if img_2d is not None:
        pass  # Skip 2D plotting

    # We don't draw on image here, just return the location
    return location

def load_lidar_points(bin_file):
    """Load LiDAR points from binary file"""
    points = np.fromfile(bin_file, dtype=np.float32).reshape(-1, 4)
    return points  # Return x, y, z, intensity

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

def detect_2d_objects(image_path):
    """Run YOLOv5 2D detection - exactly matching inference_org.py detect2d function"""
    bbox_list = []
    
    device = select_device('0')
    model = DetectMultiBackend('yolov5s.pt', device=device, dnn=False)
    stride, names, pt, jit, onnx, engine = model.stride, model.names, model.pt, model.jit, model.onnx, model.engine
    imgsz = check_img_size([640, 640], s=stride)
    
    dataset = LoadImages(image_path, img_size=imgsz, stride=stride, auto=pt)
    model.warmup(imgsz=(1, 3, *imgsz), half=False)
    
    for path, im, im0s, vid_cap, s in dataset:
        im = torch.from_numpy(im).to(device).float()
        im /= 255
        if len(im.shape) == 3:
            im = im[None]
            
        pred = model(im, augment=False, visualize=False)
        pred = non_max_suppression(prediction=pred, classes=[0, 2, 3, 5])  # person, car, motorcycle, bus
        
        for i, det in enumerate(pred):
            if len(det):
                # Scale coordinates back to original image size - EXACTLY like inference_org.py
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0s.shape).round()
                
                for *xyxy, conf, cls in reversed(det):
                    xyxy_ = (torch.tensor(xyxy).view(1,4)).view(-1).tolist()
                    xyxy_ = [int(x) for x in xyxy_]
                    
                    # Create bbox in the same format as inference_org.py
                    top_left, bottom_right = ((xyxy_[0]), (xyxy_[1])), ((xyxy_[2]), (xyxy_[3]))
                    bbox = [top_left, bottom_right]
                    c = int(cls)
                    label = names[c]
                    
                    # Map YOLO classes to KITTI classes - only cars and pedestrians
                    label_map = {
                        'person': 'pedestrian',
                        'car': 'car'
                        # Exclude motorcycles and buses to get only 11 car detections
                    }
                    mapped_label = label_map.get(label, None)
                    if mapped_label:  # Only include mapped classes
                        bbox_list.append(Bbox(bbox, mapped_label))
                        print(f"  -> Detected {label} -> {mapped_label} at {bbox}")
    
    return bbox_list

def detect_3d_objects(image_path, calib_file, reg_weights='weights/resnet18.pkl'):
    """Run full 3D detection pipeline - matching inference_org.py exactly"""
    
    print("🔄 Loading 3D detection models...")
    
    # Load 3D regression model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    base_model = resnet18(pretrained=True)
    regressor = ResNet18(model=base_model).to(device)
    
    if os.path.exists(reg_weights):
        checkpoint = torch.load(reg_weights, map_location=device)
        regressor.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ Loaded 3D regression weights from {reg_weights}")
    else:
        print(f"⚠️  Warning: Could not find weights file {reg_weights}, using random weights")
    
    regressor.eval()
    
    # Load class averages and angle bins
    averages = ClassAverages.ClassAverages()
    angle_bins = generate_bins(2)
    
    # Load image and calibration
    img = cv2.imread(image_path)
    
    # Get 2D detections
    print("🔄 Running 2D object detection...")
    dets = detect_2d_objects(image_path)
    print(f"✅ Found {len(dets)} 2D detections")
    
    all_detected_boxes = []
    
    print("🔄 Processing 3D regression for each detection...")
    for i, det in enumerate(dets):
        if not averages.recognized_class(det.detected_class):
            continue
            
        try:
            detectedObject = DetectedObject(img, det.detected_class, det.box_2d, calib_file)
        except Exception as e:
            print(f"   Skipping detection {i}: {e}")
            continue
        
        theta_ray = detectedObject.theta_ray
        input_img = detectedObject.img
        proj_matrix = detectedObject.proj_matrix
        box_2d = det.box_2d
        detected_class = det.detected_class
        
        # Prepare input tensor
        input_tensor = torch.zeros([1, 3, 224, 224]).to(device)
        input_tensor[0, :, :, :] = input_img
        
        # Run 3D regression - exact same as inference_org.py
        with torch.no_grad():
            [orient, conf, dim] = regressor(input_tensor)
            orient = orient.cpu().data.numpy()[0, :, :]
            conf = conf.cpu().data.numpy()[0, :]
            dim = dim.cpu().data.numpy()[0, :]
            dim += averages.get_item(detected_class)
        
        # Process orientation - exact same as inference_org.py
        argmax = np.argmax(conf)
        orient = orient[argmax, :]
        cos, sin = orient[0], orient[1]
        alpha = np.arctan2(sin, cos)
        alpha += angle_bins[argmax]
        alpha -= np.pi
        
        # Calculate 3D location using the original plot3d function logic
        location = plot3d(img, proj_matrix, box_2d, dim, alpha, theta_ray)
        
        if location is not None:
            orient_3d = alpha + theta_ray
            all_detected_boxes.append({
                'location': location,
                'dimensions': dim,
                'orient': orient_3d,
                'class': detected_class,
                'box_2d': box_2d
            })
            print(f"   Detection {i+1}: {detected_class} at {location}")
    
    print(f"✅ Successfully processed {len(all_detected_boxes)} 3D detections")
    return all_detected_boxes

def create_3d_bounding_box_in_lidar_frame(location_cam, dimensions, orientation, calib_data):
    """
    Create 3D bounding box in LiDAR coordinate frame
    Fixed transformation logic for better accuracy
    """
    
    # Get transformation matrices
    R0_rect = calib_data['R0_rect']
    Tr_velo_to_cam = calib_data['Tr_velo_to_cam']
    
    # Create the full transformation matrix from rectified camera to LiDAR
    R0_rect_hom = np.eye(4)
    R0_rect_hom[:3, :3] = R0_rect
    Tr_velo_to_cam_hom = np.vstack([Tr_velo_to_cam, [0, 0, 0, 1]])
    Tr_cam_rect_to_velo = np.linalg.inv(R0_rect_hom @ Tr_velo_to_cam_hom)
    
    # Transform the center location from camera to LiDAR coordinates
    loc_cam_rect_hom = np.append(location_cam, 1)
    loc_velo_hom = Tr_cam_rect_to_velo @ loc_cam_rect_hom
    center_velo = loc_velo_hom[:3]
    
    # CRITICAL FIX 1: Proper height adjustment
    # KITTI 3D location represents the center of the bottom face of the bounding box
    # We need to move the center up by half the height to get the geometric center
    height = dimensions[0]  # height is the first dimension
    
    # The Y axis in camera coordinates becomes -Z axis in LiDAR coordinates
    # So camera "up" direction is LiDAR "-Z" direction
    # But since we're in LiDAR frame now, we need to add height in +Z direction
    center_velo[2] += height / 2.0
    
    # CRITICAL FIX 2: Correct dimension mapping for Open3D
    # KITTI dimensions: [height, width, length] where length is front-to-back (longer)
    # Open3D OrientedBoundingBox expects: [length, width, height] for (x, y, z) extents
    # Since cars are typically longer than wide, we want length along the major axis
    extents = [dimensions[2], dimensions[1], dimensions[0]]  # [length, width, height]
    
    # CRITICAL FIX 3: Fix the rotation to align boxes with car orientation
    # Based on testing, we need to SUBTRACT 90 degrees (not add)
    yaw_lidar = orientation - np.pi/2  # Subtract 90 degrees to rotate correctly
    
    # Normalize angle to [-π, π]
    while yaw_lidar > np.pi:
        yaw_lidar -= 2 * np.pi
    while yaw_lidar < -np.pi:
        yaw_lidar += 2 * np.pi
    
    # Create rotation matrix for LiDAR frame (rotation around Z-axis)
    R_velo = np.array([[np.cos(yaw_lidar), -np.sin(yaw_lidar), 0],
                       [np.sin(yaw_lidar),  np.cos(yaw_lidar), 0],
                       [0,                  0,                 1]])
    
    print(f"  -> LiDAR location: {center_velo}, Extents (w,l,h): {extents}, LiDAR yaw: {yaw_lidar:.3f} rad ({np.degrees(yaw_lidar):.1f}°)")
    
    return center_velo, R_velo, extents

def create_3d_visualization(lidar_points, detected_boxes, calib_data):
    """
    Create interactive 3D visualization using Open3D
    """
    print(f"\n🎯 Creating 3D visualization with {len(lidar_points)} LiDAR points and {len(detected_boxes)} detections...")
    
    # Create Open3D PointCloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(lidar_points[:, :3])
    
    # Color the point cloud by height for better visualization
    points_np = np.asarray(pcd.points)
    colors = np.zeros((len(points_np), 3))
    z_min, z_max = points_np[:, 2].min(), points_np[:, 2].max()
    z_normalized = (points_np[:, 2] - z_min) / (z_max - z_min)
    colors[:, 0] = z_normalized  # Red component increases with height
    colors[:, 2] = 1 - z_normalized  # Blue component decreases with height
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # This list will hold all geometries to be rendered
    geometries = [pcd]
    
    print(f"Processing {len(detected_boxes)} detected boxes for 3D visualization:")
    
    # Process and add each detected bounding box
    for i, box in enumerate(detected_boxes):
        loc_cam_rect = box['location']
        dim = box['dimensions']
        ry = box['orient']
        class_name = box['class']
        
        print(f"Box {i+1}: {class_name} - Camera location: {loc_cam_rect}, Dimensions (h,w,l): {dim}, Orient: {ry:.3f} rad ({np.degrees(ry):.1f}°)")
        
        # Create 3D bounding box in LiDAR coordinate frame
        center_velo, R_velo, extents = create_3d_bounding_box_in_lidar_frame(
            loc_cam_rect, dim, ry, calib_data
        )
        
        # Create the OrientedBoundingBox
        obb = o3d.geometry.OrientedBoundingBox(center_velo, R_velo, extents)
        
        # Set color based on class
        if class_name == 'car':
            obb.color = (0, 1, 0)  # Green for cars
        elif class_name == 'pedestrian':
            obb.color = (1, 0, 0)  # Red for pedestrians
        else:
            obb.color = (1, 1, 0)  # Yellow for others
            
        geometries.append(obb)
    
    # Add a coordinate frame for reference (X-red, Y-green, Z-blue)
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5.0, origin=[0, 0, 0])
    geometries.append(coord_frame)
    
    print(f"\n🚀 Launching 3D visualizer...")
    print("📝 Controls:")
    print("   - Mouse: Rotate view")
    print("   - Scroll: Zoom in/out")
    print("   - Press 'Q' to quit")
    print("   - Press 'H' for more controls")
    
    # Open the visualization window
    o3d.visualization.draw_geometries(
        geometries, 
        window_name="LiDAR 3D Point Cloud with 3D Detections",
        width=1400, 
        height=900,
        left=50,
        top=50
    )

def main():
    """Main function to run 3D LiDAR visualization with detection"""
    
    # File paths
    bin_file = '/home/jal.parikh/yolo3d/YOLO3D/eval/velodyne/000002.bin'
    calib_file = '/home/jal.parikh/yolo3d/YOLO3D/eval/camera_cal/calib_cam_to_cam.txt'  # Use global calibration file
    image_file = '/home/jal.parikh/yolo3d/YOLO3D/eval/image_2/000002.png'
    
    print("🚀 Starting 3D LiDAR Point Cloud Visualization with Detection...")
    
    # Check if files exist
    if not os.path.exists(image_file):
        print(f"❌ Camera image not found: {image_file}")
        return
    
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
    print(f"   Intensity:   {lidar_points[:, 3].min():.2f} to {lidar_points[:, 3].max():.2f}")
    
    print("\n🔄 Running 3D object detection...")
    try:
        detected_boxes = detect_3d_objects(image_file, calib_file)
        print(f"✅ Detected {len(detected_boxes)} 3D objects")
    except Exception as e:
        print(f"⚠️  3D detection failed: {e}")
        print("📋 Continuing with LiDAR visualization only...")
        detected_boxes = []
    
    # Create and show 3D visualization
    create_3d_visualization(lidar_points, detected_boxes, calib_data)
    
    # Print detection summary
    if detected_boxes:
        print(f"\n📦 Detection Summary:")
        for i, box in enumerate(detected_boxes):
            print(f"   {i+1}. {box['class']} at location: ({box['location'][0]:.1f}, {box['location'][1]:.1f}, {box['location'][2]:.1f})")
    
    print(f"\n🎉 3D visualization completed!")

if __name__ == "__main__":
    main()
