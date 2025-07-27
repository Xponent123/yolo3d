#!/usr/bin/env python3
"""
LiDAR back-projection with 3D bounding box detection
Combines LiDAR point cloud visualization with YOLO3D object detection
"""

import numpy as np
import cv2
import matplotlib.cm as cm
import torch
import torch.nn as nn
from torchvision.models import resnet18
import sys
import os
from pathlib import Path

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

def plot3d(img, proj_matrix, box_2d, dimensions, alpha, theta_ray, img_2d=None):
    """Calculate 3D location using the YOLO3D math"""
    location, X = calc_location(dimensions, proj_matrix, box_2d, alpha, theta_ray)
    if location is None:
        return None

    orient = alpha + theta_ray

    if img_2d is not None:
        # plot_2d_box(img_2d, box_2d)  # Skip 2D plotting for now
        pass

    plot_3d_box(img, proj_matrix, orient, dimensions, location)
    return location

class Bbox:
    def __init__(self, box_2d, class_):
        self.box_2d = box_2d
        self.detected_class = class_

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

def project_3d_box_to_image(box_3d, calib_data, camera_id=2):
    """Project 3D bounding box corners to image coordinates"""
    
    P = calib_data[f'P{camera_id}']
    R0_rect = calib_data['R0_rect']
    Tr_velo_to_cam = calib_data['Tr_velo_to_cam']
    
    # Convert 3D box corners to homogeneous coordinates
    corners_homo = np.hstack([box_3d, np.ones((box_3d.shape[0], 1))])
    
    # Transform through the pipeline: LiDAR -> Camera -> Rectified -> Image
    Tr_velo_to_cam_homo = np.vstack([Tr_velo_to_cam, [0, 0, 0, 1]])
    cam_corners_homo = Tr_velo_to_cam_homo @ corners_homo.T
    cam_corners = cam_corners_homo[:3, :].T
    
    # Apply rectification
    R0_rect_homo = np.eye(4)
    R0_rect_homo[:3, :3] = R0_rect
    rect_corners_homo = R0_rect_homo @ np.vstack([cam_corners.T, np.ones(cam_corners.shape[0])])
    rect_corners = rect_corners_homo[:3, :].T
    
    # Filter out corners behind camera
    front_mask = rect_corners[:, 2] > 0
    if not np.any(front_mask):
        return None
    
    # Project to image
    rect_corners_homo = np.hstack([rect_corners, np.ones((rect_corners.shape[0], 1))])
    image_corners_homo = P @ rect_corners_homo.T
    image_corners = image_corners_homo[:2, :] / image_corners_homo[2, :]
    
    return image_corners.T

def create_3d_box_corners(location, dimensions, rotation_y):
    """Create 8 corners of 3D bounding box in LiDAR coordinates"""
    
    # dimensions: [height, width, length] in camera frame
    h, w, l = dimensions
    
    # Create corners in local object coordinate system (centered at origin)
    corners = np.array([
        [-l/2, -w/2, -h/2],  # 0: bottom-back-left
        [ l/2, -w/2, -h/2],  # 1: bottom-front-left  
        [ l/2,  w/2, -h/2],  # 2: bottom-front-right
        [-l/2,  w/2, -h/2],  # 3: bottom-back-right
        [-l/2, -w/2,  h/2],  # 4: top-back-left
        [ l/2, -w/2,  h/2],  # 5: top-front-left
        [ l/2,  w/2,  h/2],  # 6: top-front-right
        [-l/2,  w/2,  h/2],  # 7: top-back-right
    ])
    
    # Rotation matrix around Z-axis (yaw in LiDAR frame)
    cos_r, sin_r = np.cos(rotation_y), np.sin(rotation_y)
    R = np.array([
        [cos_r, -sin_r, 0],
        [sin_r,  cos_r, 0],
        [0,      0,     1]
    ])
    
    # Rotate corners
    corners_rotated = corners @ R.T
    
    # Translate to world position
    corners_world = corners_rotated + location
    
    return corners_world

def detect_2d_objects(image_path):
    """Run YOLOv5 2D detection"""
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
        pred = non_max_suppression(prediction=pred, classes=[0, 2, 3, 5, 7])  # person, car, motorcycle, bus, truck
        
        for i, det in enumerate(pred):
            if len(det):
                for *xyxy, conf, cls in reversed(det):
                    xyxy_ = [int(x) for x in (torch.tensor(xyxy).view(1,4)).view(-1).tolist()]
                    top_left, bottom_right = (xyxy_[0], xyxy_[1]), (xyxy_[2], xyxy_[3])
                    bbox = [top_left, bottom_right]
                    c = int(cls)
                    label = names[c]
                    label_map = {
                        'person': 'pedestrian',
                        'car': 'car',
                        'bus': 'car',
                        'truck': 'car'
                    }
                    mapped_label = label_map.get(label, label.lower())
                    if mapped_label and conf > 0.5:  # Confidence threshold
                        bbox_list.append(Bbox(bbox, mapped_label))
    
    return bbox_list

def detect_3d_objects(image_path, calib_file, reg_weights='weights/resnet18.pkl'):
    """Run full 3D detection pipeline using the same logic as inference_org.py"""
    
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
    result_img = img.copy()  # Create a copy for drawing 3D boxes
    
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
        
        # Run 3D regression
        with torch.no_grad():
            [orient, conf, dim] = regressor(input_tensor)
            orient = orient.cpu().data.numpy()[0, :, :]
            conf = conf.cpu().data.numpy()[0, :]
            dim = dim.cpu().data.numpy()[0, :]
            dim += averages.get_item(detected_class)
        
        # Process orientation
        argmax = np.argmax(conf)
        orient = orient[argmax, :]
        cos, sin = orient[0], orient[1]
        alpha = np.arctan2(sin, cos)
        alpha += angle_bins[argmax]
        alpha -= np.pi
        
        # Calculate 3D location and draw box directly on result image
        # This uses the EXACT same approach as inference_org.py
        location = plot3d(result_img, proj_matrix, box_2d, dim, alpha, theta_ray)
        
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
    return all_detected_boxes, result_img

def draw_lidar_points_on_image(image, image_points, depths, max_distance=80):
    """Draw LiDAR points on camera image"""
    
    image_copy = image.copy()
    height, width = image.shape[:2]
    
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
        return image_copy
    
    # Normalize depths for coloring
    normalized_depths = np.clip(valid_depths / max_distance, 0, 1)
    
    # Use jet colormap: close = red, far = blue
    colors = cm.jet(1 - normalized_depths)
    colors_rgb = (colors[:, :3] * 255).astype(np.uint8)
    
    # Draw points
    for point, color in zip(valid_points, colors_rgb):
        x, y = int(point[0]), int(point[1])
        cv2.circle(image_copy, (x, y), 1, color.tolist(), -1)
    
    return image_copy

def draw_3d_boxes_on_image(image, detected_boxes, calib_data):
    """Draw 3D bounding boxes projected onto image"""
    
    image_copy = image.copy()
    
    for i, box in enumerate(detected_boxes):
        location = box['location']
        dimensions = box['dimensions']
        rotation_y = box['orient']
        class_name = box['class']
        
        # Transform location from camera coordinates to LiDAR coordinates for box creation
        # We need to work in camera coordinates for the box projection
        
        # Create 3D box corners in camera coordinate system
        h, w, l = dimensions
        
        # Create corners in camera frame (KITTI convention)
        corners_3d_cam = np.array([
            [l/2,  h/2,  w/2],    # front-top-right
            [l/2,  h/2, -w/2],    # front-top-left
            [l/2, -h/2, -w/2],    # front-bottom-left
            [l/2, -h/2,  w/2],    # front-bottom-right
            [-l/2, h/2,  w/2],    # back-top-right
            [-l/2, h/2, -w/2],    # back-top-left
            [-l/2,-h/2, -w/2],    # back-bottom-left
            [-l/2,-h/2,  w/2]     # back-bottom-right
        ])
        
        # Rotation matrix around Y-axis (camera frame)
        cos_r, sin_r = np.cos(rotation_y), np.sin(rotation_y)
        R_y = np.array([
            [ cos_r, 0, sin_r],
            [ 0,     1, 0    ],
            [-sin_r, 0, cos_r]
        ])
        
        # Rotate and translate corners
        corners_3d_cam = corners_3d_cam @ R_y.T + location
        
        # Project corners to image
        corners_homo = np.hstack([corners_3d_cam, np.ones((corners_3d_cam.shape[0], 1))])
        
        # Get projection matrix and rectification
        P = calib_data['P2']
        R0_rect = calib_data['R0_rect']
        
        # Apply rectification
        R0_rect_homo = np.eye(4)
        R0_rect_homo[:3, :3] = R0_rect
        rect_corners_homo = R0_rect_homo @ corners_homo.T
        rect_corners = rect_corners_homo[:3, :].T
        
        # Filter corners in front of camera
        front_mask = rect_corners[:, 2] > 0
        if not np.any(front_mask):
            continue
            
        # Project to image
        rect_corners_homo_proj = np.hstack([rect_corners, np.ones((rect_corners.shape[0], 1))])
        image_corners_homo = P @ rect_corners_homo_proj.T
        image_corners = (image_corners_homo[:2, :] / image_corners_homo[2, :]).T
        
        # Convert to integer coordinates
        image_corners = image_corners.astype(int)
        
        # Define box edges (connections between corners)
        edges = [
            [0, 1], [1, 2], [2, 3], [3, 0],  # front face
            [4, 5], [5, 6], [6, 7], [7, 4],  # back face
            [0, 4], [1, 5], [2, 6], [3, 7]   # connecting edges
        ]
        
        # Draw edges
        color = (0, 255, 0) if class_name == 'Car' else (255, 0, 0)  # Green for cars, red for pedestrians
        thickness = 2
        
        for edge in edges:
            if front_mask[edge[0]] and front_mask[edge[1]]:
                pt1 = tuple(image_corners[edge[0]])
                pt2 = tuple(image_corners[edge[1]])
                cv2.line(image_copy, pt1, pt2, color, thickness)
        
        # Draw class label
        if front_mask[0]:  # If front corner is visible
            label_pos = tuple(image_corners[0])
            cv2.putText(image_copy, class_name, label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    return image_copy

def create_detection_image(base_image_with_3d_boxes, lidar_image_points, lidar_depths, detected_boxes):
    """Create comprehensive detection visualization by overlaying LiDAR points on image that already has 3D boxes"""
    
    # Start with the image that already has 3D boxes drawn
    result_image = base_image_with_3d_boxes.copy()
    
    # Draw LiDAR points on top
    print("🔄 Drawing LiDAR points...")
    result_image = draw_lidar_points_on_image(result_image, lidar_image_points, lidar_depths)
    
    # Add title and info
    cv2.putText(result_image, 'LiDAR + 3D Detection Overlay', (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    cv2.putText(result_image, f'Detections: {len(detected_boxes)} | LiDAR points: {len(lidar_image_points)}', 
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return result_image

def main():
    """Main function to run LiDAR back-projection with 3D detection"""
    
    # File paths
    bin_file = '/home/jal.parikh/yolo3d/YOLO3D/eval/velodyne/000002.bin'
    calib_file = '/home/jal.parikh/yolo3d/YOLO3D/eval/calib/000002.txt'
    image_file = '/home/jal.parikh/yolo3d/YOLO3D/eval/image_2/000002.png'
    output_file = '/home/jal.parikh/yolo3d/YOLO3D/lidar_detection_overlay.png'
    
    print("🚀 Starting LiDAR back-projection with 3D detection...")
    
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
    
    print("🔄 Projecting LiDAR points to camera...")
    image_points, depths, front_mask = project_lidar_to_camera(lidar_points, calib_data, camera_id=2)
    print(f"✅ Successfully projected {len(image_points)} LiDAR points")
    
    print("🔄 Running 3D object detection...")
    try:
        detected_boxes, result_image_with_3d_boxes = detect_3d_objects(image_file, calib_file)
        print(f"✅ Detected {len(detected_boxes)} 3D objects")
    except Exception as e:
        print(f"⚠️  3D detection failed: {e}")
        print("📋 Continuing with LiDAR visualization only...")
        detected_boxes = []
        result_image_with_3d_boxes = cv2.imread(image_file)
    
    print("🔄 Creating final visualization...")
    final_image = create_detection_image(result_image_with_3d_boxes, image_points, depths, detected_boxes)
    
    # Save result
    cv2.imwrite(output_file, final_image)
    print(f"✅ Saved detection overlay: {output_file}")
    
    # Print statistics
    in_bounds = np.sum(
        (image_points[:, 0] >= 0) & 
        (image_points[:, 0] < final_image.shape[1]) & 
        (image_points[:, 1] >= 0) & 
        (image_points[:, 1] < final_image.shape[0])
    )
    print(f"📈 LiDAR points in image: {in_bounds}/{len(image_points)} ({100*in_bounds/len(image_points):.1f}%)")
    print(f"📦 3D objects detected: {len(detected_boxes)}")
    print(f"🎉 Successfully generated LiDAR + 3D detection visualization!")

if __name__ == "__main__":
    main()
