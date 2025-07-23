#!/usr/bin/env python3

import os
import cv2
import rospy
import torch
import numpy as np

from sensor_msgs.msg import Image, PointCloud2
from cv_bridge import CvBridge
from pathlib import Path

from models.common import DetectMultiBackend
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.torch_utils import select_device
from torchvision.models import resnet18, vgg11

import sensor_msgs.point_cloud2 as pc2
import time

from script.Dataset import generate_bins, DetectedObject
from script import ClassAverages
from script.Model import ResNet18, VGG11
from library.Math import calc_location
from library.Plotting import plot_3d_box

# TF2 imports
import tf2_ros
import tf.transformations as tft
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import Marker, MarkerArray

ROOT = Path(__file__).resolve().parent

class Bbox:
    def __init__(self, box_2d, cls):
        self.box_2d = box_2d
        self.detected_class = cls

def detect2d(yolo_model, image, imgsz, class_filter):
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

        backbones = {'resnet18': resnet18(pretrained=True), 'vgg11': vgg11(pretrained=True)}
        regressors = {'resnet18': ResNet18, 'vgg11': VGG11}
        base = backbones[model_select]
        self.regressor = regressors[model_select](model=base).cuda()
        ckpt = torch.load(weight_file, map_location='cuda')
        self.regressor.load_state_dict(ckpt['model_state_dict'])
        self.regressor.eval()

        device = select_device('')
        self.yolo = DetectMultiBackend(yolo_weights, device=device, dnn=False, data=yolo_data)
        self.imgsz = check_img_size([640, 640], s=self.yolo.stride)

        self.averages = ClassAverages.ClassAverages()
        self.angle_bins = generate_bins(2)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.marker_pub = rospy.Publisher('/yolo3d/boxes_lidar', MarkerArray, queue_size=10)
        self.pub = rospy.Publisher('/yolo3d/output_image', Image, queue_size=1)
        self.bev_pub = rospy.Publisher('/yolo3d/bev_image', Image, queue_size=1)

        rospy.Subscriber('/camera/rgb', Image, self.callback, queue_size=1, buff_size=2**24)
        rospy.Subscriber('/velodyne_points', PointCloud2, self.lidar_callback, queue_size=1)

        self.latest_lidar = None

        rospy.loginfo("YOLO3D ROS node started.")
        rospy.spin()

    def lidar_callback(self, msg):
        self.latest_lidar = msg

    def get_transform_matrix(self, target_frame, source_frame):
        try:
            trans = self.tf_buffer.lookup_transform(target_frame, source_frame, rospy.Time(0), rospy.Duration(1.0))
            t = trans.transform.translation
            q = trans.transform.rotation
            T = tft.quaternion_matrix([q.x, q.y, q.z, q.w])
            T[0:3, 3] = [t.x, t.y, t.z]
            return T
        except Exception as e:
            rospy.logwarn(f"TF lookup failed: {e}")
            return None

    def transform_point(self, point, T):
        point_hom = np.append(point, 1.0)
        point_out = T @ point_hom
        return point_out[:3]

    def world_to_bev(self, x, y, res=0.1, side_range=(-40, 40), fwd_range=(0, 70)):
        x_bev = int((x - fwd_range[0]) / res)
        y_bev = int((y - side_range[0]) / res)
        return x_bev, y_bev

    def generate_bev_map(self, points, res=0.1, side_range=(-40, 40), fwd_range=(0, 70), height_range=(-2.5, 1.0)):
        x_points = points[:, 0]
        y_points = points[:, 1]
        z_points = points[:, 2]
        intensities = points[:, 3]

        mask = np.logical_and.reduce((x_points > fwd_range[0], x_points < fwd_range[1],
                                      y_points > side_range[0], y_points < side_range[1]))
        x_points, y_points, z_points, intensities = x_points[mask], y_points[mask], z_points[mask], intensities[mask]

        x_img = ((x_points - fwd_range[0]) / res).astype(np.int32)
        y_img = ((y_points - side_range[0]) / res).astype(np.int32)

        x_max = int((fwd_range[1] - fwd_range[0]) / res)
        y_max = int((side_range[1] - side_range[0]) / res)

        bev_img = np.zeros((x_max, y_max, 3), dtype=np.uint8)

        z_clip = np.clip(z_points, height_range[0], height_range[1])
        z_norm = (z_clip - height_range[0]) / (height_range[1] - height_range[0])
        bev_img[x_img, y_img, 2] = (z_norm * 255).astype(np.uint8)

        intensity_norm = np.clip(intensities, 0, 1)
        bev_img[x_img, y_img, 1] = (intensity_norm * 255).astype(np.uint8)

        counts = {}
        for x, y in zip(x_img, y_img):
            counts[(x, y)] = counts.get((x, y), 0) + 1
        density = np.clip(np.array([counts[(x, y)] for x, y in zip(x_img, y_img)]), 0, 20) / 20
        bev_img[x_img, y_img, 0] = (density * 255).astype(np.uint8)

        return bev_img

    def callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            rospy.logerr(f"CV bridge error: {e}")
            return

        img = cv2.resize(frame, (640, 640))
        dets = detect2d(self.yolo, img, self.imgsz, [0, 2, 3, 5])
        rospy.loginfo(f"[YOLO2D] {len(dets)} detections")

        T = self.get_transform_matrix("velodyne", "camera")
        if T is None:
            return

        marker_array = MarkerArray()

        for i, det in enumerate(dets):
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

            loc_lidar = self.transform_point(loc, T)

            marker = Marker()
            marker.header.frame_id = "velodyne"
            marker.header.stamp = msg.header.stamp
            marker.ns = "yolo3d"
            marker.id = i
            marker.type = Marker.CUBE
            marker.action = Marker.ADD
            marker.pose.position.x = loc_lidar[0]
            marker.pose.position.y = loc_lidar[1]
            marker.pose.position.z = loc_lidar[2]
            marker.pose.orientation.w = 1.0
            marker.scale.x = dim[2]
            marker.scale.y = dim[0]
            marker.scale.z = dim[1]
            marker.color.a = 0.7
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0

            marker_array.markers.append(marker)

        ego = Marker()
        ego.header.frame_id = "velodyne"
        ego.header.stamp = msg.header.stamp
        ego.ns = "ego"
        ego.id = 999
        ego.type = Marker.CUBE
        ego.action = Marker.ADD
        ego.pose.position.x = 0
        ego.pose.position.y = 0
        ego.pose.position.z = 0
        ego.scale.x = 2.0
        ego.scale.y = 1.0
        ego.scale.z = 1.0
        ego.color.a = 1.0
        ego.color.r = 0.0
        ego.color.g = 0.0
        ego.color.b = 1.0
        marker_array.markers.append(ego)

        self.marker_pub.publish(marker_array)
        out = self.bridge.cv2_to_imgmsg(img, 'bgr8')
        out.header = msg.header
        self.pub.publish(out)

        if self.latest_lidar is not None:
            pc = list(pc2.read_points(self.latest_lidar, field_names=("x", "y", "z", "intensity"), skip_nans=True))
            points = np.array(pc)
            bev_img = self.generate_bev_map(points)

            for marker in marker_array.markers:
                if marker.ns == "yolo3d":
                    cx, cy = marker.pose.position.x, marker.pose.position.y
                    w, l = marker.scale.x, marker.scale.y
                    corners = [
                        (cx - w/2, cy - l/2),
                        (cx - w/2, cy + l/2),
                        (cx + w/2, cy + l/2),
                        (cx + w/2, cy - l/2)
                    ]
                    pts = [self.world_to_bev(x, y) for x, y in corners]
                    pts = np.array(pts, np.int32).reshape((-1, 1, 2))
                    cv2.polylines(bev_img, [pts], isClosed=True, color=(0, 0, 255), thickness=2)

            bev_img = cv2.rotate(bev_img, cv2.ROTATE_90_CLOCKWISE)
            bev_msg = self.bridge.cv2_to_imgmsg(bev_img, encoding="bgr8")
            bev_msg.header = msg.header
            self.bev_pub.publish(bev_msg)

if __name__ == '__main__':
    try:
        YOLO3DNode()
    except rospy.ROSInterruptException:
        pass
