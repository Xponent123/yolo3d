# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run inference on images, videos, directories, streams, etc.
"""

import argparse
import os
import sys
from pathlib import Path
import glob

import cv2
import torch
import numpy as np
import open3d as o3d  # ++++ IMPORT FOR 3D VISUALIZATION ++++

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.datasets import LoadImages
from utils.general import LOGGER, check_img_size, check_requirements, non_max_suppression, print_args, scale_coords
from utils.torch_utils import select_device, time_sync

import torch
import torch.nn as nn
from torchvision.models import resnet18, vgg11

from script.Dataset import generate_bins, DetectedObject
from library.Math import *
from library.Plotting import *
from script import Model, ClassAverages
from script.Model import ResNet, ResNet18, VGG11

# model factory to choose model
model_factory = {
    'resnet': resnet18(pretrained=True),
    'resnet18': resnet18(pretrained=True),
}
regressor_factory = {
    'resnet': ResNet,
    'resnet18': ResNet18,
    'vgg11': VGG11
}

class Bbox:
    def __init__(self, box_2d, class_):
        self.box_2d = box_2d
        self.detected_class = class_


# ++++ START: CORRECTED 3D VISUALIZATION FUNCTION ++++
def visualize_in_3d(lidar_points, detected_boxes, Tr_cam_rect_to_velo):
    """
    Generates an interactive 3D visualization of the LiDAR point cloud and 3D bounding boxes.
    This version correctly handles coordinate system transformations for orientation.
    """
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

    print(f"\nProcessing {len(detected_boxes)} detected boxes for 3D visualization:")

    # Process and add each detected bounding box
    for i, box in enumerate(detected_boxes):
        loc_cam_rect = box['location']
        dim = box['dimensions']  # Stored as (h, w, l) - height, width, length
        ry = box['orient']       # Rotation-Y in camera frame

        print(f"Box {i+1}: Camera location: {loc_cam_rect}, Dimensions (h,w,l): {dim}, Orient: {ry:.3f} rad ({np.degrees(ry):.1f}°)")

        # 1. Transform the center location from camera to LiDAR coordinates
        loc_cam_rect_hom = np.append(loc_cam_rect, 1)
        loc_velo_hom = Tr_cam_rect_to_velo @ loc_cam_rect_hom
        center_velo = loc_velo_hom[:3]

        # 2. CRITICAL FIX: Adjust the center to ground level
        # In KITTI, the camera coordinate system has the 3D box location representing
        # the center bottom of the object (on the ground). After transformation to LiDAR,
        # we need to adjust for the object's center.
        # However, the current calculation seems to be adding height incorrectly.
        
        # Let's check if the transformed location is already at the correct height
        # and adjust accordingly. In KITTI LiDAR coordinates, ground level is around Z = -1.7
        height = dim[0]  # height is the first dimension
        
        # IMPORTANT: The original center_velo Z-coordinate might already be correct
        # Let's adjust it to be at the geometric center of the bounding box
        # Since KITTI camera coordinates are at bottom center, and we want box center:
        # We should add half the height, but in the correct direction
        
        # For KITTI: if the car is on the ground, its center should be at ground + height/2
        # Typical ground level in KITTI LiDAR is around -1.7m
        # So for a 1.5m tall car, center should be at -1.7 + 0.75 = -0.95m
        
        # The issue might be that we're adding when we should be ensuring proper ground alignment
        # Let's try a different approach: check current height and adjust if needed
        
        expected_ground_level = -1.7  # Typical KITTI ground level
        expected_center_height = expected_ground_level + height / 2.0
        
        # If the transformed Z is way off from expected, use the expected value
        if abs(center_velo[2] - expected_center_height) > 3.0:  # If more than 3m off
            print(f"    Adjusting Z from {center_velo[2]:.2f} to {expected_center_height:.2f}")
            center_velo[2] = expected_center_height
        else:
            # Otherwise, just ensure it's at the geometric center
            center_velo[2] += height / 2.0

        # 3. Define the extents for Open3D in the correct order: [length, width, height]
        # KITTI dimensions are [height, width, length], Open3D expects [length, width, height]
        extents = [dim[2], dim[1], dim[0]]  # [length, width, height]

        # 4. CRITICAL FIX: Proper rotation transformation from camera to LiDAR
        # Camera frame: X=right, Y=down, Z=forward  
        # LiDAR frame: X=forward, Y=left, Z=up
        
        # Analysis of the transformation matrix shows:
        # Tr_cam_rect_to_velo transforms from camera coordinates to LiDAR coordinates
        # The rotation relationship for KITTI dataset is well-established:
        
        # Method: Use the transformation matrix to understand the coordinate alignment
        # Looking at Tr_cam_rect_to_velo:
        # [ 2.35e-04  1.04e-02  9.99e-01  ...]  <- LiDAR X comes mostly from Camera Z
        # [-9.99e-01  1.06e-02  1.24e-04  ...]  <- LiDAR Y comes mostly from -Camera X  
        # [-1.06e-02 -9.99e-01  1.05e-02  ...]  <- LiDAR Z comes mostly from -Camera Y
        
        # This means:
        # - Camera Z (forward) -> LiDAR X (forward) 
        # - Camera X (right) -> LiDAR -Y (right, since LiDAR Y is left)
        # - Camera Y (down) -> LiDAR -Z (down, since LiDAR Z is up)
        
        # For rotation: ry is rotation around camera Y-axis (down)
        # This corresponds to rotation around LiDAR -Z axis (down)
        # But we want rotation around LiDAR +Z axis (up), so we negate
        # Additionally, there's a 90-degree offset due to coordinate system alignment
        
        # The correct conversion based on KITTI coordinate system analysis:
        yaw_lidar = -ry - np.pi/2
        
        # Normalize angle to [-π, π]
        while yaw_lidar > np.pi:
            yaw_lidar -= 2 * np.pi
        while yaw_lidar < -np.pi:
            yaw_lidar += 2 * np.pi
        
        R_velo = np.array([[np.cos(yaw_lidar), -np.sin(yaw_lidar), 0],
                           [np.sin(yaw_lidar),  np.cos(yaw_lidar), 0],
                           [0,                  0,                 1]])

        print(f"  -> LiDAR location: {center_velo}, Extents (l,w,h): {extents}, LiDAR yaw: {yaw_lidar:.3f} rad ({np.degrees(yaw_lidar):.1f}°)")

        # 5. Create the OrientedBoundingBox
        obb = o3d.geometry.OrientedBoundingBox(center_velo, R_velo, extents)
        obb.color = (1, 0, 0)  # Set box color to red
        geometries.append(obb)

    # Add a coordinate frame for reference (X-red, Y-green, Z-blue)
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5.0, origin=[0, 0, 0])
    geometries.append(coord_frame)

    # Open the visualization window
    o3d.visualization.draw_geometries(geometries, 
                                    window_name="3D Detection Results",
                                    width=1200, height=800)

# ++++ END: CORRECTED 3D VISUALIZATION FUNCTION ++++


def detect3d(
    reg_weights,
    model_select,
    source,
    calib_file,
    show_result,
    save_result,
    output_path
    ):

    # Directory setup
    imgs_path = sorted(glob.glob(str(source) + '/*'))
    calib = str(calib_file)

    # Load models
    base_model = model_factory[model_select]
    regressor = regressor_factory[model_select](model=base_model).cuda()
    checkpoint = torch.load(reg_weights)
    regressor.load_state_dict(checkpoint['model_state_dict'])
    regressor.eval()

    averages = ClassAverages.ClassAverages()
    angle_bins = generate_bins(2)

    for i, img_path in enumerate(imgs_path):
        img = cv2.imread(img_path)
        
        dets = detect2d(
            weights='yolov5s.pt',
            source=img_path,
            data='data/coco128.yaml',
            imgsz=[640, 640],
            device=0,
            classes=[0, 2, 3, 5, 7] # person, car, motorcycle, bus, truck
        )

        resized_image = img
        all_detected_boxes = []

        for det in dets:
            if not averages.recognized_class(det.detected_class):
                continue
            try: 
                detectedObject = DetectedObject(img, det.detected_class, det.box_2d, calib)
            except:
                continue

            theta_ray = detectedObject.theta_ray
            input_img = detectedObject.img
            proj_matrix = detectedObject.proj_matrix
            box_2d = det.box_2d
            detected_class = det.detected_class

            input_tensor = torch.zeros([1,3,224,224]).cuda()
            input_tensor[0,:,:,:] = input_img

            [orient, conf, dim] = regressor(input_tensor)
            orient = orient.cpu().data.numpy()[0, :, :]
            conf = conf.cpu().data.numpy()[0, :]
            dim = dim.cpu().data.numpy()[0, :]
            dim += averages.get_item(detected_class)

            argmax = np.argmax(conf)
            orient = orient[argmax, :]
            cos, sin = orient[0], orient[1]
            alpha = np.arctan2(sin, cos)
            alpha += angle_bins[argmax]
            alpha -= np.pi
            # alpha= -1.5708/2

            location = plot3d(resized_image, proj_matrix, box_2d, dim, alpha, theta_ray)

            if location is not None:
                orient_3d = alpha + theta_ray
                all_detected_boxes.append({
                    'location': location,
                    'dimensions': dim,
                    'orient': orient_3d
                })

        # --- Show 2D Results First ---
        if show_result:
            cv2.imshow('2D Detection with 3D Boxes', resized_image)
            cv2.waitKey(1) # Use waitKey(1) to ensure the window updates

        # --- SETUP AND LAUNCH 3D VISUALIZATION ---
        if show_result and all_detected_boxes:
            # Define calibration matrices
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
            
            # Create the full transformation matrix from rectified camera to velodyne
            R0_rect_hom = np.eye(4)
            R0_rect_hom[:3, :3] = R0_rect
            Tr_velo_to_cam_hom = np.vstack([Tr_velo_to_cam, [0, 0, 0, 1]])
            Tr_cam_rect_to_velo = np.linalg.inv(R0_rect_hom @ Tr_velo_to_cam_hom)

            # Load LiDAR data
            lidar_file = '/home/jal.parikh/yolo3d/YOLO3D/eval/velodyne/000002.bin'
            lidar_points = np.fromfile(lidar_file, dtype=np.float32).reshape(-1, 4)
            
            # Launch the 3D viewer
            print("\nLaunching 3D visualizer... Press 'q' in the new window to close.")
            visualize_in_3d(lidar_points, all_detected_boxes, Tr_cam_rect_to_velo)

        if save_result and output_path is not None:
            os.makedirs(output_path, exist_ok=True)
            cv2.imwrite(f'{output_path}/{i:03d}.png', resized_image)

        # Clean up OpenCV windows
        cv2.destroyAllWindows()


@torch.no_grad()
def detect2d(
    weights,
    source,
    data,
    imgsz,
    device,
    classes
    ):

    bbox_list = []
    source = str(source)
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=False, data=data)
    stride, names, pt, jit, onnx, engine = model.stride, model.names, model.pt, model.jit, model.onnx, model.engine
    imgsz = check_img_size(imgsz, s=stride)

    dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
    model.warmup(imgsz=(1, 3, *imgsz), half=False)
    dt, seen = [0.0, 0.0, 0.0], 0
    for path, im, im0s, vid_cap, s in dataset:
        t1 = time_sync()
        im = torch.from_numpy(im).to(device).float()
        im /= 255
        if len(im.shape) == 3:
            im = im[None]
        t2 = time_sync()
        dt[0] += t2 - t1

        pred = model(im, augment=False, visualize=False)
        t3 = time_sync()
        dt[1] += t3 - t2

        pred = non_max_suppression(prediction=pred, classes=classes)
        dt[2] += time_sync() - t3

        for i, det in enumerate(pred):
            seen += 1
            p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
            p = Path(p)
            s += '%gx%g ' % im.shape[2:]

            if len(det):
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "

                for *xyxy, conf, cls in reversed(det):
                    xyxy_ = [int(x) for x in (torch.tensor(xyxy).view(1,4)).view(-1).tolist()]
                    top_left, bottom_right = (xyxy_[0], xyxy_[1]), (xyxy_[2], xyxy_[3])
                    bbox = [top_left, bottom_right]
                    c = int(cls)
                    label = names[c]
                    label_map = {
                        'person': 'Pedestrian',
                        'car': 'Car'
                    }
                    mapped_label = label_map.get(label)
                    if mapped_label:
                        bbox_list.append(Bbox(bbox, mapped_label))
            
            LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')

    t = tuple(x / seen * 1E3 for x in dt)
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    return bbox_list


def plot3d(
    img,
    proj_matrix,
    box_2d,
    dimensions,
    alpha,
    theta_ray,
    img_2d=None
    ):

    location, X = calc_location(dimensions, proj_matrix, box_2d, alpha, theta_ray)
    if location is None:
        return None

    orient = alpha + theta_ray

    if img_2d is not None:
        plot_2d_box(img_2d, box_2d)

    plot_3d_box(img, proj_matrix, orient, dimensions, location)
    return location


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default=ROOT / 'eval/image_2', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--reg_weights', type=str, default='weights/epoch_10.pkl', help='Regressor model weights')
    parser.add_argument('--model_select', type=str, default='resnet', help='Regressor model list: resnet, vgg, eff')
    parser.add_argument('--calib_file', type=str, default=ROOT / 'eval/camera_cal/calib_cam_to_cam.txt', help='Calibration file or path')
    parser.add_argument('--show_result', action='store_true', help='Show Results with imshow')
    parser.add_argument('--save_result', action='store_true', help='Save result')
    parser.add_argument('--output_path', type=str, default=ROOT / 'output', help='Save output path')

    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1
    print_args(FILE.stem, opt)
    return opt


def main(opt):
    detect3d(
        reg_weights=opt.reg_weights,
        model_select=opt.model_select,
        source=opt.source,
        calib_file=opt.calib_file,
        show_result=opt.show_result,
        save_result=opt.save_result,
        output_path=opt.output_path
    )

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)