import cv2
import numpy as np

def create_bird_eye_view(detections, img_shape=(1200, 800), scale=5, max_distance=150):
    """
    Create a bird's eye view visualization of 3D detections
    
    Args:
        detections: List of detection results with location, dimensions, and orientation
        img_shape: Shape of the bird's eye view image (height, width)
        scale: Pixels per meter
        max_distance: Maximum distance to display in meters
    
    Returns:
        bird_eye_img: Bird's eye view image
    """
    bird_eye_img = np.zeros((img_shape[0], img_shape[1], 3), dtype=np.uint8)
    
    # Draw grid
    draw_grid(bird_eye_img, scale, max_distance)
    
    # Draw camera position (center bottom)
    camera_x = img_shape[1] // 2
    camera_y = img_shape[0] - 50
    cv2.circle(bird_eye_img, (camera_x, camera_y), 8, (0, 255, 0), -1)
    cv2.putText(bird_eye_img, "Camera", (camera_x - 25, camera_y + 25), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    # Draw coordinate axes
    draw_axes(bird_eye_img, camera_x, camera_y, scale)
    
    # Draw detections
    valid_detections = 0
    for detection in detections:
        if draw_detection_bev(bird_eye_img, detection, scale, img_shape):
            valid_detections += 1
    
    # Add info text
    cv2.putText(bird_eye_img, f"Detections: {valid_detections}/{len(detections)}", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(bird_eye_img, f"Scale: {scale} px/m", 
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return bird_eye_img

def draw_grid(img, scale, max_distance):
    """Draw grid lines on bird's eye view"""
    height, width = img.shape[:2]
    
    # Vertical lines (every 10 meters)
    for i in range(-max_distance, max_distance + 1, 10):
        x = width // 2 + i * scale
        if 0 <= x < width:
            color = (100, 100, 100) if i != 0 else (150, 150, 150)
            cv2.line(img, (x, 0), (x, height), color, 1)
            if i != 0:
                cv2.putText(img, f"{i}m", (x - 15, height - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
    
    # Horizontal lines (every 10 meters)
    for i in range(0, max_distance + 1, 10):
        y = height - 50 - i * scale
        if 0 <= y < height:
            color = (100, 100, 100) if i != 0 else (150, 150, 150)
            cv2.line(img, (0, y), (width, y), color, 1)
            if i > 0:
                cv2.putText(img, f"{i}m", (5, y - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)

def draw_axes(img, camera_x, camera_y, scale):
    """Draw coordinate axes"""
    # X-axis (left-right)
    cv2.arrowedLine(img, (camera_x, camera_y), (camera_x + 30, camera_y), 
                    (0, 0, 255), 2)  # Red for X
    cv2.putText(img, "X", (camera_x + 35, camera_y + 5), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    
    # Z-axis (forward)
    cv2.arrowedLine(img, (camera_x, camera_y), (camera_x, camera_y - 30), 
                    (255, 0, 0), 2)  # Blue for Z
    cv2.putText(img, "Z", (camera_x + 5, camera_y - 35), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

def draw_detection_bev(img, detection, scale, img_shape):
    """Draw a single detection in bird's eye view with proper orientation"""
    location, dimensions, detected_class, orientation = detection
    
    # KITTI coordinate system: x=right, y=down, z=forward
    x, y, z = location
    
    # Convert to bird's eye view coordinates
    # BEV: x=right (same as KITTI), y=forward (z in KITTI)
    img_x = int(img_shape[1] // 2 + x * scale)
    img_y = int(img_shape[0] - 50 - z * scale)
    
    # Check bounds with margin
    margin = 50
    if (img_x < margin or img_x >= img_shape[1] - margin or 
        img_y < margin or img_y >= img_shape[0] - margin):
        print(f"Detection out of BEV bounds: x={x:.2f}, z={z:.2f}, img_x={img_x}, img_y={img_y}")
        return False
    
    # Get dimensions (h, w, l) - KITTI format
    h, w, l = dimensions
    
    # Create oriented bounding box
    half_width = w / 2
    half_length = l / 2
    
    # Define corner points in local coordinates (relative to object center)
    # In BEV: x=right, y=forward
    corners = np.array([
        [-half_width, -half_length],  # rear left
        [half_width, -half_length],   # rear right
        [half_width, half_length],    # front right
        [-half_width, half_length]    # front left
    ])
    
    # Apply rotation (orientation is around Y-axis in camera coordinates)
    # In BEV, this becomes rotation around vertical axis
    cos_theta = np.cos(orientation)
    sin_theta = np.sin(orientation)
    rotation_matrix = np.array([
        [cos_theta, -sin_theta],
        [sin_theta, cos_theta]
    ])
    
    # Rotate corners
    rotated_corners = corners @ rotation_matrix.T
    
    # Convert to image coordinates
    img_corners = []
    for corner in rotated_corners:
        corner_x = int(img_x + corner[0] * scale)
        corner_y = int(img_y + corner[1] * scale)
        img_corners.append([corner_x, corner_y])
    
    img_corners = np.array(img_corners, dtype=np.int32)
    
    # Color based on class
    color = get_class_color(detected_class)
    
    # Draw oriented bounding box
    cv2.polylines(img, [img_corners], True, color, 2)
    
    # Fill the box with semi-transparent color
    overlay = img.copy()
    cv2.fillPoly(overlay, [img_corners], color)
    cv2.addWeighted(overlay, 0.3, img, 0.7, 0, img)
    
    # Draw center point
    cv2.circle(img, (img_x, img_y), 3, color, -1)
    
    # Draw direction arrow (front of the car)
    arrow_length = max(15, int(l * scale / 3))
    arrow_x = int(img_x + arrow_length * sin_theta)
    arrow_y = int(img_y + arrow_length * cos_theta)
    cv2.arrowedLine(img, (img_x, img_y), (arrow_x, arrow_y), color, 2)
    
    # Add label with distance
    distance = np.sqrt(x**2 + z**2)
    label = f"{detected_class}"
    label_detailed = f"{distance:.1f}m"
    
    # Draw label background
    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
    cv2.rectangle(img, (img_x - label_size[0]//2 - 2, img_y - 25), 
                  (img_x + label_size[0]//2 + 2, img_y - 10), (0, 0, 0), -1)
    
    cv2.putText(img, label, (img_x - label_size[0]//2, img_y - 15), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    cv2.putText(img, label_detailed, (img_x - 15, img_y + 15), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
    
    return True

def get_class_color(detected_class):
    """Get color for different object classes"""
    color_map = {
        'car': (0, 255, 255),      # Yellow
        'pedestrian': (0, 255, 0), # Green
        'truck': (0, 0, 255),      # Red
        'bus': (255, 255, 0),      # Cyan
        'motorcycle': (255, 0, 255), # Magenta
        'bicycle': (255, 165, 0)   # Orange
    }
    return color_map.get(detected_class, (128, 128, 128))  # Gray for unknown
