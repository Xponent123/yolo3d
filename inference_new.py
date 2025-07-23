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

# Add these imports
import numpy as np
import open3d as o3d

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

# from script.Dataset import generate_bins, DetectedObject
# from library.Math import *
# from library.Plotting import *
# from script import Model, ClassAverages
# from script.Model import ResNet, ResNet18, VGG11


# ##################################################################
# # The following imports and functions are placeholders for       #
# # the user's custom modules. For a self-contained example,       #
# # I'm adding simple placeholder classes/functions here.          #
# ##################################################################

class Model(nn.Module):
    def __init__(self, model):
        super(Model, self).__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)

class ResNet(Model):
    def __init__(self, model):
        super(ResNet, self).__init__(model)
        # Modify the model for regression
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 3) # Example for dim

class ResNet18(ResNet):
    def __init__(self, model):
        super(ResNet18, self).__init__(model)


class VGG11(Model):
    pass

class ClassAverages:
    def __init__(self):
        self.averages = {'car': [1.52, 1.6, 3.88]}

    def recognized_class(self, class_):
        return class_ in self.averages

    def get_item(self, class_):
        return self.averages.get(class_, [0, 0, 0])

def generate_bins(num_bins):
    return np.linspace(-np.pi, np.pi, num_bins + 1)

def calc_location(dimensions, proj_matrix, box_2d, alpha, theta_ray):
    # This is a simplified placeholder.
    # In a real scenario, this would involve more complex geometric calculations.
    loc = [0, 0, 20] # Default location
    return loc, []


def plot_3d_box(img, proj_matrix, orient, dimensions, location):
    # This function is assumed to draw on the 2D image, which is not the focus of the point cloud plotting.
    pass

class DetectedObject:
    def __init__(self, img, detected_class, box_2d, calib):
        # Placeholder initialization
        self.img = torch.randn(3, 224, 224)
        self.theta_ray = 0
        self.proj_matrix = np.random.rand(3, 4)


# ##################################################################
# # End of placeholder section                                     #
# ##################################################################


# model factory to choose model
model_factory = {
    'resnet': resnet18(pretrained=False), # Set to False for placeholder
    'resnet18': resnet18(pretrained=False),
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

def parse_calib_file(filepath):
    """Reads a calibration file and returns the projection matrix (P2) and Tr_velo_to_cam."""
    with open(filepath) as f:
        lines = f.readlines()

    P2 = np.array([float(x) for x in lines[2].strip().split(' ')[1:]], dtype=np.float32).reshape(3, 4)
    Tr_velo_to_cam = np.array([float(x) for x in lines[5].strip().split(' ')[1:]], dtype=np.float32).reshape(3, 4)
    R0_rect = np.array([float(x) for x in lines[4].strip().split(' ')[1:]], dtype=np.float32).reshape(3, 3)

    return {'P2': P2, 'R0_rect': R0_rect, 'Tr_velo_to_cam': Tr_velo_to_cam}

def load_point_cloud(bin_filepath):
    """Loads a point cloud from a .bin file."""
    points = np.fromfile(bin_filepath, dtype=np.float32).reshape(-1, 4)
    return points

def project_velo_to_cam(velo_points, calib):
    """Projects velodyne points to camera coordinates."""
    points_3d = velo_points[:, :3]
    points_3d_hom = np.hstack((points_3d, np.ones((points_3d.shape[0], 1), dtype=np.float32)))

    # Apply Tr_velo_to_cam
    points_in_cam_ref = (calib['Tr_velo_to_cam'] @ points_3d_hom.T).T

    # Apply R0_rect
    r0_hom = np.eye(4)
    r0_hom[:3,:3] = calib['R0_rect']
    points_rect = (r0_hom @ np.hstack((points_in_cam_ref, np.ones((points_in_cam_ref.shape[0],1)))).T).T

    return points_rect[:,:3]

def plot_point_cloud(points, detections):
    """
    Plots the point cloud and the 3D bounding boxes.
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)

    for det in detections:
        center = det['location']
        dimensions = det['dimensions']
        yaw = det['orient']

        # Create a bounding box
        # Note: Open3D's rotation is R, while the KITTI yaw is around the Y-axis.
        # We need to construct the rotation matrix from the yaw.
        R = np.array([[np.cos(yaw), 0, np.sin(yaw)],
                      [0, 1, 0],
                      [-np.sin(yaw), 0, np.cos(yaw)]])
        bbox = o3d.geometry.OrientedBoundingBox(center, R, dimensions)
        bbox.color = (1, 0, 0) # Red color for the bounding box
        vis.add_geometry(bbox)

    # Set camera view
    ctr = vis.get_view_control()
    ctr.set_front([0, -1, -0.1])
    ctr.set_lookat([0, 0, 20])
    ctr.set_up([0, -1, 0])
    ctr.set_zoom(0.1)

    vis.run()
    vis.destroy_window()

def detect3d(
    reg_weights,
    model_select,
    source,
    calib_file,
    bin_file, # Add bin_file path argument
    show_result,
    save_result,
    output_path
    ):

    # Directory
    imgs_path = sorted(glob.glob(str(source) + '/*'))
    calib_data = parse_calib_file(calib_file)


    # load model
    base_model = model_factory[model_select]
    regressor = regressor_factory[model_select](model=base_model)
    # If you have a GPU, you can move the model to GPU
    # regressor = regressor.cuda()
    # checkpoint = torch.load(reg_weights)
    # regressor.load_state_dict(checkpoint['model_state_dict'])
    # regressor.eval()


    averages = ClassAverages()
    angle_bins = generate_bins(2)

    all_detections_for_point_cloud = []

    # loop images
    for i, img_path in enumerate(imgs_path):
        # read image
        img = cv2.imread(img_path)

        # Run detection 2d
        dets = detect2d(
            weights='yolov5s.pt',
            source=img_path,
            data='data/coco128.yaml',
            imgsz=[640, 640],
            device='cpu', # or '0' for GPU
            classes=[2] # Class '2' is 'car' in COCO
        )
        resized_image = img

        for det in dets:
            if not averages.recognized_class(det.detected_class):
                continue
            try:
                # Use a simple placeholder for DetectedObject
                detectedObject = DetectedObject(img, det.detected_class, det.box_2d, calib_data)
            except:
                continue

            theta_ray = detectedObject.theta_ray
            input_img = detectedObject.img
            proj_matrix = calib_data['P2']
            box_2d = det.box_2d
            detected_class = det.detected_class

            # Dummy values for demonstration
            orient = np.array([[np.cos(1.5), np.sin(1.5)]])
            conf = np.array([1.0])
            dim = np.array([1.5, 1.6, 3.9])

            dim += averages.get_item(detected_class)

            argmax = np.argmax(conf)
            orient = orient[argmax, :]
            cos_val = orient[0]
            sin_val = orient[1]
            alpha = np.arctan2(sin_val, cos_val)
            alpha += angle_bins[argmax]
            alpha -= np.pi/2 # Adjust for KITTI coordinate system

            location, _ = calc_location(dim, proj_matrix, box_2d, alpha, theta_ray)

            # Store detection for point cloud visualization
            all_detections_for_point_cloud.append({
                'location': location,
                'dimensions': dim,
                'orient': alpha + theta_ray
            })

            # plot 3d detection on 2D image
            plot3d(resized_image, proj_matrix, box_2d, dim, alpha, theta_ray)

        if show_result:
            cv2.imshow('3d detection', resized_image)
            cv2.waitKey(0)

        if save_result and output_path is not None:
            try:
                os.makedirs(output_path, exist_ok=True)
            except:
                pass
            cv2.imwrite(f'{output_path}/{i:03d}.png', resized_image)

    # Load and plot point cloud
    if bin_file:
        velo_points = load_point_cloud(bin_file)
        # For visualization, we transform points to the camera coordinate system
        points_in_cam_coord = project_velo_to_cam(velo_points, calib_data)
        plot_point_cloud(points_in_cam_coord, all_detections_for_point_cloud)


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
    model.warmup(imgsz=(1, 3, *imgsz))  # warmup
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


        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            s += '%gx%g ' % im.shape[2:]  # print string

            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    xyxy_ = (torch.tensor(xyxy).view(1,4)).view(-1).tolist()
                    xyxy_ = [int(x) for x in xyxy_]
                    top_left, bottom_right = (xyxy_[0], xyxy_[1]), (xyxy_[2], xyxy_[3])
                    bbox = [top_left, bottom_right]
                    c = int(cls)  # integer class
                    label = names[c]
                    bbox_list.append(Bbox(bbox, "car")) # Forcing car for this example

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
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--classes', default=[2], nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--reg_weights', type=str, default='weights/epoch_10.pkl', help='Regressor model weights')
    parser.add_argument('--model_select', type=str, default='resnet18', help='Regressor model list: resnet, vgg, eff')
    parser.add_argument('--calib_file', type=str, default=ROOT / 'eval/camera_cal/calib_cam_to_cam.txt', help='Calibration file or path')
    parser.add_argument('--bin_file', type=str, default=ROOT / 'eval/velodyne/000002.bin', help='Path to the .bin point cloud file')
    parser.add_argument('--show_result', action='store_true', help='Show Results with imshow')
    parser.add_argument('--save_result', action='store_true', help='Save result')
    parser.add_argument('--output_path', type=str, default=ROOT / 'output', help='Save output pat')

    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(FILE.stem, opt)
    return opt

def main(opt):
    # For demonstration, creating a dummy calibration file with the provided values
    calib_content = """P0: 7.215377000000e+02 0.000000000000e+00 6.095593000000e+02 0.000000000000e+00 0.000000000000e+00 7.215377000000e+02 1.728540000000e+02 0.000000000000e+00 0.000000000000e+00 0.000000000000e+00 1.000000000000e+00 0.000000000000e+00
P1: 7.215377000000e+02 0.000000000000e+00 6.095593000000e+02 -3.875744000000e+02 0.000000000000e+00 7.215377000000e+02 1.728540000000e+02 0.000000000000e+00 0.000000000000e+00 0.000000000000e+00 1.000000000000e+00 0.000000000000e+00
P2: 7.215377000000e+02 0.000000000000e+00 6.095593000000e+02 4.485728000000e+01 0.000000000000e+00 7.215377000000e+02 1.728540000000e+02 2.163791000000e-01 0.000000000000e+00 0.000000000000e+00 1.000000000000e+00 2.745884000000e-03
P3: 7.215377000000e+02 0.000000000000e+00 6.095593000000e+02 -3.395242000000e+02 0.000000000000e+00 7.215377000000e+02 1.728540000000e+02 2.199936000000e+00 0.000000000000e+00 0.000000000000e+00 1.000000000000e+00 2.729905000000e-03
R0_rect: 9.999239000000e-01 9.837760000000e-03 -7.445048000000e-03 -9.869795000000e-03 9.999421000000e-01 -4.278459000000e-03 7.402527000000e-03 4.351614000000e-03 9.999631000000e-01
Tr_velo_to_cam: 7.533745000000e-03 -9.999714000000e-01 -6.166020000000e-04 -4.069766000000e-03 1.480249000000e-02 7.280733000000e-04 -9.998902000000e-01 -7.631618000000e-02 9.998621000000e-01 7.523790000000e-03 1.480755000000e-02 -2.717806000000e-01
Tr_imu_to_velo: 9.999976000000e-01 7.553071000000e-04 -2.035826000000e-03 -8.086759000000e-01 -7.854027000000e-04 9.998898000000e-01 -1.482298000000e-02 3.195559000000e-01 2.024406000000e-03 1.482454000000e-02 9.998881000000e-01 -7.997231000000e-01
"""
    calib_dir = ROOT / 'eval/camera_cal'
    os.makedirs(calib_dir, exist_ok=True)
    with open(opt.calib_file, 'w') as f:
        f.write(calib_content)

    detect3d(
        reg_weights=opt.reg_weights,
        model_select=opt.model_select,
        source=opt.source,
        calib_file=opt.calib_file,
        bin_file=opt.bin_file,
        show_result=opt.show_result,
        save_result=opt.save_result,
        output_path=opt.output_path
    )

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)