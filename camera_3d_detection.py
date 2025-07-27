#!/usr/bin/env python3
"""
Camera-focused 3D detection visualization
Shows detected 3D bounding boxes projected onto the camera image
"""

import numpy as np
import cv2
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
        pass  # Skip 2D plotting for now

    plot_3d_box(img, proj_matrix, orient, dimensions, location)
    return location

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
                        'person': 'Pedestrian',
                        'car': 'Car',
                        'bus': 'Car',
                        'truck': 'Car'
                    }
                    mapped_label = label_map.get(label)
                    if mapped_label and conf > 0.5:  # Confidence threshold
                        bbox_list.append(Bbox(bbox, mapped_label))
    
    return bbox_list

def detect_3d_objects(image_path, calib_file, reg_weights='weights/resnet18.pkl'):
    """Run full 3D detection pipeline"""
    
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
    result_image = img.copy()
    
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
        
        # Calculate 3D location and draw 3D box
        location = plot3d(result_image, proj_matrix, box_2d, dim, alpha, theta_ray)
        
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
    
    # Add title and info to the result image
    cv2.putText(result_image, '3D Object Detection Results', (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    cv2.putText(result_image, f'Detected: {len(all_detected_boxes)} objects', 
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return all_detected_boxes, result_image

def main():
    """Main function to run 3D detection visualization"""
    
    # File paths
    calib_file = '/home/jal.parikh/yolo3d/YOLO3D/eval/calib/000002.txt'
    image_file = '/home/jal.parikh/yolo3d/YOLO3D/eval/image_2/000002.png'
    output_file = '/home/jal.parikh/yolo3d/YOLO3D/camera_3d_detection.png'
    
    print("🚀 Starting 3D object detection on camera image...")
    
    # Check if files exist
    if not os.path.exists(image_file):
        print(f"❌ Camera image not found: {image_file}")
        return
    
    print("🔄 Running 3D object detection...")
    try:
        detected_boxes, result_image = detect_3d_objects(image_file, calib_file)
        print(f"✅ Detected {len(detected_boxes)} 3D objects")
    except Exception as e:
        print(f"❌ 3D detection failed: {e}")
        return
    
    # Save result
    cv2.imwrite(output_file, result_image)
    print(f"✅ Saved 3D detection result: {output_file}")
    
    # Print detection summary
    print(f"\n📦 Detection Summary:")
    for i, box in enumerate(detected_boxes):
        print(f"   {i+1}. {box['class']} at location: ({box['location'][0]:.1f}, {box['location'][1]:.1f}, {box['location'][2]:.1f})")
    
    print(f"\n🎉 Successfully generated 3D detection visualization!")

if __name__ == "__main__":
    main()
