#!/usr/bin/env python3

import torch
import numpy as np
import cv2
from pathlib import Path
import sys
import os
import warnings

# Add the current directory to path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from script.Model import ResNet18
from script import ClassAverages
from script.Dataset import generate_bins, DetectedObject
from torchvision.models import resnet18, ResNet18_Weights
from library.Math import *
from library.Plotting import *

# Disable warnings
warnings.filterwarnings("ignore", category=FutureWarning)

def test_3d_detection():
    print("Loading model and weights...")
    
    # Load model
    base_model = resnet18(weights=ResNet18_Weights.DEFAULT)
    regressor = ResNet18(model=base_model).cuda()
    
    # Load weights
    checkpoint = torch.load('weights/resnet18.pkl', map_location='cpu')
    regressor.load_state_dict(checkpoint['model_state_dict'])
    regressor.eval()
    
    # Load class averages
    averages = ClassAverages.ClassAverages()
    angle_bins = generate_bins(2)
    
    print("Loading image...")
    img_path = "eval/image_2/000002.png"
    calib_path = "eval/camera_cal/calib_cam_to_cam.txt"
    img = cv2.imread(img_path)
    
    # Manual bounding box for testing (you can replace this with actual YOLO detections)
    # Let's use a car bounding box from your detected coordinates
    # From your output: "Box 14: car - Width: 269, Height: 119"
    # Let's estimate coordinates for a car
    box_2d = [(500, 200), (769, 319)]  # [top_left, bottom_right]
    detected_class = "car"
    
    print(f"Testing 3D detection for class: {detected_class}")
    print(f"Bounding box: {box_2d}")
    
    # Draw 2D box for reference
    plot_2d_box(img, box_2d)
    
    if not averages.recognized_class(detected_class):
        print(f"Class {detected_class} not recognized!")
        return
    
    try:
        print("Creating DetectedObject...")
        detectedObject = DetectedObject(img, detected_class, box_2d, calib_path)
        
        theta_ray = detectedObject.theta_ray
        input_img = detectedObject.img
        proj_matrix = detectedObject.proj_matrix
        
        print("Running 3D regression...")
        input_tensor = torch.zeros([1, 3, 224, 224]).cuda()
        input_tensor[0, :, :, :] = input_img
        
        [orient, conf, dim] = regressor(input_tensor)
        orient = orient.cpu().data.numpy()[0, :, :]
        conf = conf.cpu().data.numpy()[0, :]
        dim = dim.cpu().data.numpy()[0, :]
        
        dim += averages.get_item(detected_class)
        
        argmax = np.argmax(conf)
        orient = orient[argmax, :]
        cos = orient[0]
        sin = orient[1]
        alpha = np.arctan2(sin, cos)
        alpha += angle_bins[argmax]
        alpha -= np.pi
        
        print(f"Predicted dimensions: {dim}")
        print(f"Predicted orientation: {alpha}")
        
        # Plot 3D box
        location, X = calc_location(dim, proj_matrix, box_2d, alpha, theta_ray)
        orient = alpha + theta_ray
        
        print(f"Calculated location: {location}")
        
        plot_3d_box(img, proj_matrix, orient, dim, location)
        
        print("Saving result...")
        # Save result
        cv2.imwrite('test_3d_output.png', img)
        print("Result saved as test_3d_output.png")
        print("3D bounding box successfully generated!")
        
    except Exception as e:
        print(f"Error in 3D detection: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_3d_detection()
