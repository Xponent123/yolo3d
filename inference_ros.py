#!/usr/bin/env python3

import os
import cv2
import rospy
import torch
import numpy as np

from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from pathlib import Path

from models.common import DetectMultiBackend
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.torch_utils import select_device
from torchvision.models import resnet18, vgg11

import time

from script.Dataset import generate_bins, DetectedObject
from script import ClassAverages
from script.Model import ResNet18, VGG11
from library.Math import calc_location
from library.Plotting import plot_3d_box

ROOT = Path(__file__).resolve().parent

class Bbox:
    def __init__(self, box_2d, cls):
        self.box_2d = box_2d
        self.detected_class = cls

def detect2d(yolo_model, image, imgsz, class_filter):
    """Run YOLOv5 2D detection and return list of Bbox."""
    start = time.time()
    img = cv2.resize(image, tuple(imgsz))
    img0 = img.copy()
    tensor = torch.from_numpy(img).to(yolo_model.device).permute(2, 0, 1).float() / 255.0
    tensor = tensor.unsqueeze(0)

    yolo_model.warmup(imgsz=(1, 3, *imgsz), half=False)
    pred = yolo_model(tensor, augment=False, visualize=False)
    pred = non_max_suppression(pred, classes=class_filter)

    boxes = []
    for det in pred:
        if not len(det):
            continue
        det[:, :4] = scale_coords(tensor.shape[2:], det[:, :4], img0.shape).round()
        for *xyxy, conf, cls in reversed(det):
            tl = (int(xyxy[0]), int(xyxy[1]))
            br = (int(xyxy[2]), int(xyxy[3]))
            label = yolo_model.names[int(cls)]
            label_map = {
                'person': 'pedestrian',
                'bicycle': 'cyclist',
                'motorbike': 'cyclist',
                'motorcycle': 'cyclist',
                'bus': 'truck',
                'truck': 'truck',
                'car': 'car'
            }
            mapped = label_map.get(label, label.lower())
            boxes.append(Bbox([tl, br], mapped))
    end = time.time()
    rospy.loginfo(f"[YOLO2D] {len(boxes)} detections in {end-start:.2f}s")
    return boxes

class YOLO3DNode:
    def __init__(self):
        rospy.init_node('yolo3d_inference_node', anonymous=True)
        self.bridge = CvBridge()

        self.calib_file = rospy.get_param('~calib_file', str(ROOT / 'eval' / 'camera_cal' / 'calib_cam_to_cam.txt'))
        weight_file = rospy.get_param('~reg_weights', str(ROOT / 'weights' / 'resnet18.pkl'))
        yolo_weights = rospy.get_param('~yolo_weights', str(ROOT / 'yolov5s.pt'))
        model_select = rospy.get_param('~model_select', 'resnet18')
        yolo_data = rospy.get_param('~yolo_data', str(ROOT / 'data' / 'coco128.yaml'))
        class_filter = rospy.get_param('~classes', [0, 2, 3, 5])

        # 3D Regressor
        backbones = {
            'resnet18': resnet18(pretrained=True),
            'vgg11': vgg11(pretrained=True)
        }
        regressors = {
            'resnet18': ResNet18,
            'vgg11': VGG11
        }
        base = backbones[model_select]
        self.regressor = regressors[model_select](model=base).cuda()
        ckpt = torch.load(weight_file, map_location='cuda')
        self.regressor.load_state_dict(ckpt['model_state_dict'])
        self.regressor.eval()

        # YOLOv5 setup
        device = select_device('')
        self.yolo = DetectMultiBackend(yolo_weights, device=device, dnn=False, data=yolo_data)
        self.imgsz = check_img_size([640, 640], s=self.yolo.stride)

        # Class averages & bins
        self.averages = ClassAverages.ClassAverages()
        self.angle_bins = generate_bins(2)

        # ROS I/O
        rospy.Subscriber('/camera/rgb', Image, self.callback, queue_size=1, buff_size=2**24)
        self.pub = rospy.Publisher('/yolo3d/output_image', Image, queue_size=1)

        rospy.loginfo("YOLO3D ROS node started.")
        rospy.spin()

    def callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            rospy.logerr(f"CV bridge error: {e}")
            return

        img = cv2.resize(frame, (640, 640))  # Direct resize to 640x640

        dets = detect2d(self.yolo, img, self.imgsz, [0, 2, 3, 5])
        rospy.loginfo(f"[YOLO2D] {len(dets)} detections")

        for det in dets:
            if not self.averages.recognized_class(det.detected_class):
                continue
            try:
                dobj = DetectedObject(img, det.detected_class, det.box_2d, self.calib_file)
            except Exception as e:
                rospy.logerr(f"DetectedObject failed: {e}")
                continue

            tensor = torch.zeros([1, 3, 224, 224]).cuda()
            tensor[0] = dobj.img

            orient, conf, dim = self.regressor(tensor)
            orient = orient.cpu().detach().numpy()[0]
            conf = conf.cpu().detach().numpy()[0]
            dim = dim.cpu().detach().numpy()[0] + self.averages.get_item(det.detected_class)

            idx = np.argmax(conf)
            alpha = np.arctan2(orient[idx, 1], orient[idx, 0]) + self.angle_bins[idx] - np.pi

            loc, _ = calc_location(dim, dobj.proj_matrix, det.box_2d, alpha, dobj.theta_ray)
            plot_3d_box(img, dobj.proj_matrix, alpha + dobj.theta_ray, dim, loc)

        out = self.bridge.cv2_to_imgmsg(img, 'bgr8')
        out.header = msg.header
        self.pub.publish(out)

if __name__ == '__main__':
    try:
        YOLO3DNode()
    except rospy.ROSInterruptException:
        pass
