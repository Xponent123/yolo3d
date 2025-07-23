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
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import numpy as np

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
    # 'vgg11': vgg11(pretrained=True)
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

def load_calibration_matrices(calib_file):
    """
    Load and process calibration matrices for proper coordinate transformations
    Returns camera-to-LiDAR transformation matrix
    """
    try:
        with open(calib_file, 'r') as f:
            lines = f.readlines()
        
        P2 = None
        R0_rect = None
        Tr_velo_to_cam = None
        
        for line in lines:
            if line.startswith('P2:'):
                P2 = np.array([float(x) for x in line.split()[1:]]).reshape(3, 4)
            elif line.startswith('R0_rect:'):
                R0_rect = np.array([float(x) for x in line.split()[1:]]).reshape(3, 3)
            elif line.startswith('Tr_velo_to_cam:'):
                Tr_velo_to_cam = np.array([float(x) for x in line.split()[1:]]).reshape(3, 4)
        
        if P2 is None or R0_rect is None or Tr_velo_to_cam is None:
            raise ValueError("Missing calibration parameters")
        
        # Convert Tr_velo_to_cam to 4x4 homogeneous matrix
        Tr_velo_to_cam_4x4 = np.vstack([Tr_velo_to_cam, [0, 0, 0, 1]])
        
        # Convert R0_rect to 4x4 homogeneous matrix
        R0_rect_4x4 = np.eye(4)
        R0_rect_4x4[:3, :3] = R0_rect
        
        # Camera-to-LiDAR transformation (inverse of LiDAR-to-camera)
        # First invert the rectification, then invert the velo-to-cam transform
        Tr_cam_to_velo = np.linalg.inv(Tr_velo_to_cam_4x4) @ np.linalg.inv(R0_rect_4x4)
        
        print(f"Loaded calibration matrices:")
        print(f"P2 (camera projection):\n{P2}")
        print(f"R0_rect (rectification):\n{R0_rect}")
        print(f"Tr_velo_to_cam (LiDAR->Camera):\n{Tr_velo_to_cam}")
        print(f"Tr_cam_to_velo (Camera->LiDAR):\n{Tr_cam_to_velo}")
        
        return P2, R0_rect, Tr_velo_to_cam, Tr_cam_to_velo
        
    except Exception as e:
        print(f"Error loading calibration file: {e}")
        # Return default matrices if loading fails
        P2 = np.array([[7.215377e+02, 0, 6.095593e+02, 4.485728e+01],
                       [0, 7.215377e+02, 1.728540e+02, 2.163791e-01],
                       [0, 0, 1, 2.745884e-03]])
        R0_rect = np.eye(3)
        Tr_velo_to_cam = np.array([[7.533745e-03, -9.999714e-01, -6.166020e-04, -4.069766e-03],
                                   [1.480249e-02, 7.280733e-04, -9.998902e-01, -7.631618e-02],
                                   [9.998621e-01, 7.523790e-03, 1.480755e-02, -2.717806e-01]])
        
        Tr_velo_to_cam_4x4 = np.vstack([Tr_velo_to_cam, [0, 0, 0, 1]])
        R0_rect_4x4 = np.eye(4)
        R0_rect_4x4[:3, :3] = R0_rect
        Tr_cam_to_velo = np.linalg.inv(Tr_velo_to_cam_4x4) @ (R0_rect_4x4)
        print("check 115")
        return P2, R0_rect, Tr_velo_to_cam, Tr_cam_to_velo

def transform_to_lidar_coords(camera_coords, Tr_cam_to_velo):
    """
    Transform 3D coordinates from camera coordinate system to LiDAR coordinate system
    Camera coords: X=right, Y=down, Z=forward
    LiDAR coords: X=forward, Y=left, Z=up
    """
    # Convert to homogeneous coordinates
    if camera_coords.shape[-1] == 3:
        camera_coords_hom = np.hstack([camera_coords, np.ones((camera_coords.shape[0], 1))])
    else:
        camera_coords_hom = camera_coords
    
    # Apply transformation
    lidar_coords_hom = (Tr_cam_to_velo @ camera_coords_hom.T).T
    print("tranformation applied line 132")
    
    # Return 3D coordinates
    return lidar_coords_hom[:, :3]

def create_bev_plot(detections, Tr_cam_to_velo, img_width=800, img_height=600, range_x=120, range_z=120):
    """
    Create simple bird's eye view with yellow boxes on black background
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    
    # Set black background
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')
    
    # Set up coordinate system without grid
    ax.set_xlim(-range_x, range_x)
    ax.set_ylim(-5, range_z)
    ax.set_aspect('equal')
    
    # Remove all axes, ticks, and labels
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    # Add red dot as ego vehicle
    ego_dot = plt.Circle((0, 0), 2, color='red', alpha=1.0)
    ax.add_patch(ego_dot)
    
    print(f"BEV: Processing {len(detections)} car detections")
    
    # Plot detected cars as yellow boxes
    for idx, detection in enumerate(detections):
        try:
            # Camera coordinates
            x_cam, y_cam, z_cam = detection['location']
            w, h, l = detection['dimensions']
            class_name = detection['class']
            
            # Transform to LiDAR coordinates
            camera_point = np.array([[x_cam, y_cam, z_cam]])
            lidar_point = transform_to_lidar_coords(camera_point, Tr_cam_to_velo)
            x_lidar, y_lidar, z_lidar = lidar_point[0]
            
            # Filter objects outside reasonable range
            if x_lidar < 0 or x_lidar > range_z or abs(y_lidar) > range_x/2:
                continue
                
            # Create yellow rectangle for car
            rect = Rectangle((x_lidar - l/2, y_lidar - w/2), l, w, 
                            linewidth=2, 
                            edgecolor='yellow',
                            facecolor='none',
                            alpha=1.0)
            ax.add_patch(rect)
                   
        except Exception as e:
            continue
    
    # Convert matplotlib figure to OpenCV image
    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    bev_img = cv2.cvtColor(buf, cv2.COLOR_RGB2BGR)
    plt.close(fig)
    
    return bev_img

def detect3d(
    reg_weights,
    model_select,
    source,
    calib_file,
    show_result,
    save_result,
    output_path,
    show_bev=False,
    save_bev=False
    ):

    # Directory
    imgs_path = sorted(glob.glob(str(source) + '/*'))
    calib = str(calib_file)

    # load model
    base_model = model_factory[model_select]
    regressor = regressor_factory[model_select](model=base_model).cuda()

    # load weight
    checkpoint = torch.load(reg_weights)
    regressor.load_state_dict(checkpoint['model_state_dict'])
    regressor.eval()

    averages = ClassAverages.ClassAverages()
    angle_bins = generate_bins(2)

    # Load calibration matrices with proper transformations
    P2, R0_rect, Tr_velo_to_cam, Tr_cam_to_velo = load_calibration_matrices(calib)

    # loop images
    for i, img_path in enumerate(imgs_path):
        # read image
        img = cv2.imread(img_path)
        
        # Run detection 2d - only detect cars (class 2)
        dets = detect2d(
            weights='yolov5s.pt',
            source=img_path,
            data='data/coco128.yaml',
            imgsz=[640, 640],
            device=0,
            classes=[2]  # Only cars
        )

        print(f"Total 2D car detections: {len(dets)}")
        
        resized_image = img
        detections_3d = []  # Store 3D detections for BEV

        for det_idx, det in enumerate(dets):
            # Skip if not a car
            if det.detected_class != 'car':
                continue
                
            print(f"Processing car detection {det_idx}: class={det.detected_class}")
            
            # Try to process all car detections
            try: 
                detectedObject = DetectedObject(img, det.detected_class, det.box_2d, calib)
            except Exception as e:
                print(f"  Failed to create DetectedObject: {e}")
                continue

            # Only do 3D regression if class is recognized by averages
            if averages.recognized_class(det.detected_class):
                theta_ray = detectedObject.theta_ray
                input_img = detectedObject.img
                proj_matrix = detectedObject.proj_matrix
                box_2d = det.box_2d
                detected_class = det.detected_class

                input_tensor = torch.zeros([1,3,224,224]).cuda()
                input_tensor[0,:,:,:] = input_img

                # predict orient, conf, and dim
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

                # plot 3d detection
                location = plot3d(resized_image, proj_matrix, box_2d, dim, alpha, theta_ray)
                
                # Store detection for BEV
                detections_3d.append({
                    'location': location,
                    'dimensions': dim,
                    'class': detected_class
                })
                print(f"  Added 3D car detection: loc={location}, dim={dim}")

        print(f"Total 3D car detections for BEV: {len(detections_3d)}")

        # Create and show/save bird's eye view
        if show_bev or save_bev:
            bev_img = create_bev_plot(detections_3d, Tr_cam_to_velo, range_x=120, range_z=120)
            
            if show_bev:
                # Resize images to similar heights for better visualization
                height = min(resized_image.shape[0], bev_img.shape[0])
                resized_camera = cv2.resize(resized_image, (int(resized_image.shape[1] * height / resized_image.shape[0]), height))
                resized_bev = cv2.resize(bev_img, (int(bev_img.shape[1] * height / bev_img.shape[0]), height))
                
                # Create combined view
                combined_img = np.hstack([resized_camera, resized_bev])
                cv2.imshow('3D Detection + Bird\'s Eye View', combined_img)
                cv2.waitKey(0)
            
            if save_bev and output_path is not None:
                try:
                    os.makedirs(f'{output_path}/bev', exist_ok=True)
                except:
                    pass
                cv2.imwrite(f'{output_path}/bev/{i:03d}_bev.png', bev_img)

        if show_result:
            cv2.imshow('3d detection', resized_image)
            cv2.waitKey(0)

        if save_result and output_path is not None:
            try:
                os.makedirs(output_path, exist_ok=True)
            except:
                pass
            cv2.imwrite(f'{output_path}/{i:03d}.png', resized_image)

@torch.no_grad()
def detect2d(
    weights,
    source,
    data,
    imgsz,
    device,
    classes
    ):

    # array for boundingbox detection
    bbox_list = []

    # Directories
    source = str(source)

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=False, data=data)
    stride, names, pt, jit, onnx, engine = model.stride, model.names, model.pt, model.jit, model.onnx, model.engine
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)

    # Run inference
    model.warmup(imgsz=(1, 3, *imgsz), half=False)  # warmup
    dt, seen = [0.0, 0.0, 0.0], 0
    for path, im, im0s, vid_cap, s in dataset:
        t1 = time_sync()
        im = torch.from_numpy(im).to(device)
        im = im.float()
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        pred = model(im, augment=False, visualize=False)
        t3 = time_sync()
        dt[1] += t3 - t2

        # NMS
        pred = non_max_suppression(prediction=pred, classes=classes)
        dt[2] += time_sync() - t3

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            s += '%gx%g ' % im.shape[2:]  # print string

            if len(det):
                # Rescale boxes from img_size to im0 size
                # det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    xyxy_ = (torch.tensor(xyxy).view(1,4)).view(-1).tolist()
                    xyxy_ = [int(x) for x in xyxy_]
                    # print(f"xyxy_ {xyxy_}")
                    # top_left, bottom_right = ((int)(xyxy_[0]*(1241/640)), (int)(xyxy_[1]*(376/224))), ((int)(xyxy_[2]*(1241/640)), (int)(xyxy_[3]*(376/224)))
                    top_left, bottom_right = ((xyxy_[0]), (xyxy_[1])), ((xyxy_[2]), (xyxy_[3]))
                    bbox = [top_left, bottom_right]
                    c = int(cls)  # integer class
                    label = names[c]
                    label_map = {
                        'person': 'pedestrian',
                        'car': 'car'
                    }
                    mapped_label = label_map.get(label, label.lower())
                    bbox_list.append(Bbox(bbox, mapped_label))

                    # bbox_list.append(Bbox(bbox, label))
                    
            # print(f"{bbox_list[0].box_2d}")
            # print(f"{bbox_list[1].box_2d}")

            
            # Print time (inference-only)
            LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')

    # Print results
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
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

    # the math! returns X, the corners used for constraint
    location, X = calc_location(dimensions, proj_matrix, box_2d, alpha, theta_ray)

    orient = alpha + theta_ray

    if img_2d is not None:
        plot_2d_box(img_2d, box_2d)

    plot_3d_box(img, proj_matrix, orient, dimensions, location) # 3d boxes

    return location

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default=ROOT / 'eval/image_2', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--classes', default=[2], nargs='+', type=int, help='filter by class: only cars (class 2)')
    parser.add_argument('--reg_weights', type=str, default='weights/epoch_10.pkl', help='Regressor model weights')
    parser.add_argument('--model_select', type=str, default='resnet', help='Regressor model list: resnet, vgg, eff')
    parser.add_argument('--calib_file', type=str, default=ROOT / 'eval/camera_cal/calib_cam_to_cam.txt', help='Calibration file or path')
    parser.add_argument('--show_result', action='store_true', help='Show Results with imshow')
    parser.add_argument('--save_result', action='store_true', help='Save result')
    parser.add_argument('--output_path', type=str, default=ROOT / 'output', help='Save output path')
    parser.add_argument('--show_bev', action='store_true', help='Show bird\'s eye view')
    parser.add_argument('--save_bev', action='store_true', help='Save bird\'s eye view images')

    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
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
        output_path=opt.output_path,
        show_bev=opt.show_bev,
        save_bev=opt.save_bev
    )

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)