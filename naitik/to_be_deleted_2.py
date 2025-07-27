import numpy as np
import cv2
import yaml
import open3d as o3d
from ultralytics import YOLO
from cuml.cluster import DBSCAN as cuDBSCAN
from velo_to_cam import load_tr_velo_to_cam

# ==== Inputs ====
index = 53 # 753 is the frame where the car is turning 

img_path = f"/home/naitik/Downloads/kitti_extracted/kitti_extracted/images_color_left/frame_{index:04d}.jpg"
lidar_path = f"/home/naitik/Downloads/YOLO3D_running/kitti_extracted/kitti_extracted/pointclouds/cloud_{index:04d}.bin"
camera_info_path = f"/home/naitik/Downloads/obj_detect_dept_est/kitti_extracted/kitti_extracted/camera_info/camera_color_left_info_{index:04d}.yaml"
transform_path = f"/home/naitik/Downloads/obj_detect_dept_est/kitti_extracted/kitti_extracted/transforms/frame_{index:04d}.yaml"

def fit_ground_aligned_obb(points):
    # Project to ground plane (X-Y) or (X-Z) depending on your axis convention
    pts_2d = points[:, [0, 1]]  # use [0, 2] if Z is up

    # Fit 2D oriented rectangle in ground plane
    rect = cv2.minAreaRect(pts_2d.astype(np.float32))  # ((cx, cy), (w, h), angle)
    box_2d = cv2.boxPoints(rect)

    # Back-project box corners to 3D (same height as input)
    z_min, z_max = points[:, 2].min(), points[:, 2].max()
    corners_bottom = np.hstack([box_2d, np.full((4, 1), z_min)])
    corners_top    = np.hstack([box_2d, np.full((4, 1), z_max)])
    corners_3d = np.vstack([corners_bottom, corners_top])

    # Create Open3D box
    obb = o3d.geometry.OrientedBoundingBox.create_from_points(o3d.utility.Vector3dVector(corners_3d))
    return obb

# ==== Load Camera Calibration ====
with open(camera_info_path, 'r') as f:
    cam_info = yaml.safe_load(f)
R_rect = np.array(cam_info['R']).reshape(3, 3)
P_rect = np.array(cam_info['P']).reshape(3, 4)
K_rectified = P_rect[:, :3]

fx, fy = K_rectified[0, 0], K_rectified[1, 1]
cx, cy = K_rectified[0, 2], K_rectified[1, 2]

Tr_velo_to_cam = load_tr_velo_to_cam(transform_path)

# ==== Load Data ====
def load_lidar_points(path):
    return np.fromfile(path, dtype=np.float32).reshape(-1, 4)[:, :3]

image = cv2.imread(img_path)
lidar_pts = load_lidar_points(lidar_path)
lidar_pts = lidar_pts[lidar_pts[:, 0] > 0]

# ==== Run YOLOv8 Segmentation ====
model = YOLO("yolov8x-seg.pt")
results = model(image, device='cuda')[0]
H, W = image.shape[:2]

target_classes = {
    'car':        {'color_bgr': (0, 0, 255),     'color_float': (1.0, 0.0, 0.0)},
    'person':     {'color_bgr': (0, 255, 0),     'color_float': (0.0, 1.0, 0.0)},
    'truck':      {'color_bgr': (0, 165, 255),   'color_float': (1.0, 0.647, 0.0)},
    'motorcycle': {'color_bgr': (255, 0, 0),     'color_float': (0.0, 0.0, 1.0)},
    'bus':        {'color_bgr': (255, 0, 255),   'color_float': (1.0, 0.0, 1.0)},
    'bicycle':    {'color_bgr': (255, 255, 0),   'color_float': (0.0, 1.0, 1.0)},
    'traffic light':    {'color_bgr': (0, 255, 255),   'color_float': (0.0, 1.0, 1.0)},
}

boxes_lidar = []

# ==== Draw masks on image ====
seg_viz = image.copy()
for mask, cls_id in zip(results.masks.data, results.boxes.cls):
    class_name = model.names[int(cls_id)]
    color = (0, 255, 0) if class_name == 'car' else (255, 0, 0)

    color = target_classes[class_name]['color_bgr']
    mask_bin = cv2.resize(mask.cpu().numpy().astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
    contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(seg_viz, contours, -1, color, thickness=cv2.FILLED)

    # Add text label at mask center
    M = cv2.moments(mask_bin)
    if M["m00"] > 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        cv2.putText(seg_viz, class_name, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

cv2.imshow("Segmentation", seg_viz)
cv2.waitKey(0)
cv2.destroyAllWindows()

# ==== Project LiDAR Points to Image ====
lidar_hom = np.hstack((lidar_pts, np.ones((lidar_pts.shape[0], 1))))
lidar_cam = (Tr_velo_to_cam @ lidar_hom.T).T[:, :3]
valid_z = lidar_cam[:, 2] > 0
lidar_cam = lidar_cam[valid_z]
lidar_pts = lidar_pts[valid_z]

lidar_rect = (R_rect @ lidar_cam.T).T
lidar_h_rect = np.hstack((lidar_rect, np.ones((lidar_rect.shape[0], 1))))
lidar_img = (P_rect @ lidar_h_rect.T).T
lidar_img[:, 0] /= lidar_img[:, 2]
lidar_img[:, 1] /= lidar_img[:, 2]
img_coords = lidar_img[:, :2].astype(int)

in_bounds = (img_coords[:, 0] >= 0) & (img_coords[:, 0] < W) & (img_coords[:, 1] >= 0) & (img_coords[:, 1] < H)
img_coords = img_coords[in_bounds]
lidar_pts_filtered = lidar_pts[in_bounds]

# ==== Filter LiDAR points using segmentation masks ====
for mask, cls_id in zip(results.masks.data, results.boxes.cls):
    class_name = model.names[int(cls_id)]
    if class_name not in target_classes:
        continue

    mask_bin = cv2.resize(mask.cpu().numpy().astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)

    object_points = []
    for pt_2d, pt_3d in zip(img_coords, lidar_pts_filtered):
        u, v = pt_2d
        if mask_bin[v, u]:
            object_points.append(pt_3d)

    if len(object_points) < 20:
        continue

    object_points = np.array(object_points)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(object_points)
    pcd, _ = pcd.remove_radius_outlier(nb_points=1, radius=1)

    if len(pcd.points) < 10:
        continue

    dbscan = cuDBSCAN(eps=0.8, min_samples=1)
    labels = dbscan.fit(np.asarray(pcd.points)).labels_
    if labels.max() < 0:
        continue

    major_cluster = np.argmax(np.bincount(labels[labels >= 0]))
    pcd = pcd.select_by_index(np.where(labels == major_cluster)[0])

    box = pcd.get_axis_aligned_bounding_box()
    # box = fit_ground_aligned_obb(np.asarray(pcd.points))
    # box.color = [1, 0, 0] if class_name == 'car' else [0, 1, 0]
    box.color = target_classes[class_name]['color_float']
    boxes_lidar.append(box)

# ==== Visualize 3D Point Cloud and Bounding Boxes ====
full_pcd = o3d.geometry.PointCloud()
full_pcd.points = o3d.utility.Vector3dVector(lidar_pts)
full_pcd.colors = o3d.utility.Vector3dVector(np.full_like(lidar_pts, 0.5))

coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0, origin=[0, 0, 0])
o3d.visualization.draw_geometries([full_pcd, coord_frame] + boxes_lidar)
