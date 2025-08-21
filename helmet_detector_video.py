import cv2
import math
import cvzone
from ultralytics import YOLO
import numpy as np
from collections import Counter
import os
import sys

print('='*60)
print('🔍 FILE EXECUTION VERIFICATION')
print('='*60)
print(f'DEBUG: Executing file: {__file__}')
print(f'DEBUG: File exists: {os.path.exists(__file__)}')
print(f'DEBUG: Working directory: {os.getcwd()}')
print(f'DEBUG: Python executable: {sys.executable}')
print(f'DEBUG: Script directory: {os.path.dirname(os.path.abspath(__file__))}')
print('='*60)

# Initialize video capture
video_path = "media/test1.mp4"
cap = cv2.VideoCapture(video_path)

# Get video properties for output
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Create video writer for output
output_path = "media/output_helmet_lpr_detection.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Load all YOLO models
vehicle_model = YOLO("weights/yolov8m.pt")  # For detecting motorcycles/bicycles
helmet_model = YOLO("weights/helmet_detection_trained.pt")  # For helmet detection

# Define class names for helmet detection
classNames = ['With Helmet', 'Without Helmet']

# COCO class IDs we're interested in
MOTORCYCLE_CLASS = 3  # motorcycle in COCO
BICYCLE_CLASS = 1     # bicycle in COCO

def get_fixed_subrectangle(x, y, w, h):
    # always row 1, col 2 in a 3x4 grid
    sub_x = int(x + (2/4) * w)
    sub_y = int(y + (1/3) * h)
    sub_w = int(w / 4)
    sub_h = int(h / 3)
    return sub_x, sub_y, sub_w, sub_h

def detect_helmet_color(img, x1, y1, x2, y2):
    """
    Detect the dominant color of a helmet by analyzing a specific sub-region within the bounding box.
    Uses get_fixed_subrectangle() to get row 1, column 2 in a 3x4 grid within the bounding box.
    This focuses on a precise area of the helmet for better color accuracy.
    Returns the color name and RGB values. Ensures proper BGR to RGB conversion.
    """
    try:
        # Calculate the original bounding box dimensions
        x = x1
        y = y1
        w = x2 - x1
        h = y2 - y1
        
        # Get the fixed sub-rectangle coordinates (row 1, col 2 in 3x4 grid)
        sub_x, sub_y, sub_w, sub_h = get_fixed_subrectangle(x, y, w, h)
        
        # Calculate the actual coordinates for the sub-region
        sub_x1 = sub_x
        sub_y1 = sub_y
        sub_x2 = sub_x + sub_w
        sub_y2 = sub_y + sub_h
        
        # Ensure coordinates are within image bounds
        sub_x1 = max(0, sub_x1)
        sub_y1 = max(0, sub_y1)
        sub_x2 = min(img.shape[1], sub_x2)
        sub_y2 = min(img.shape[0], sub_y2)
        
        # Extract only the specified sub-region (OpenCV image is in BGR format)
        sub_region_bgr = img[sub_y1:sub_y2, sub_x1:sub_x2]
        
        # Check if sub-region is valid
        if sub_region_bgr.size == 0:
            print("Warning: Sub-region is empty, using full bounding box")
            sub_region_bgr = img[y1:y2, x1:x2]
        
        # IMPORTANT: Convert BGR to RGB for accurate color analysis
        sub_region_rgb = cv2.cvtColor(sub_region_bgr, cv2.COLOR_BGR2RGB)
        
        # Convert to HSV for better filtering
        hsv_region = cv2.cvtColor(sub_region_rgb, cv2.COLOR_RGB2HSV)
        
        # Create a mask to exclude very dark and very bright pixels
        # This helps filter out shadows, reflections, and unwanted elements
        mask = cv2.inRange(hsv_region, np.array([0, 30, 30]), np.array([179, 255, 220]))
        
        # Extract only the pixels that pass the mask from the sub-region
        filtered_pixels = sub_region_rgb[mask > 0]
        
        if len(filtered_pixels) > 10:
            # Calculate the mean color from filtered pixels (sub-region only)
            mean_color = np.mean(filtered_pixels, axis=0).astype(int)
        else:
            # Fallback: use the center area of the sub-region
            center_h = max(1, sub_region_rgb.shape[0] // 4)
            center_w = max(1, sub_region_rgb.shape[1] // 4)
            end_h = min(sub_region_rgb.shape[0], center_h * 3)
            end_w = min(sub_region_rgb.shape[1], center_w * 3)
            
            center_region_rgb = sub_region_rgb[center_h:end_h, center_w:end_w]
            if center_region_rgb.size > 0:
                mean_color = np.mean(center_region_rgb.reshape(-1, 3), axis=0).astype(int)
            else:
                # Last resort: use the entire sub-region
                mean_color = np.mean(sub_region_rgb.reshape(-1, 3), axis=0).astype(int)
        
        # Convert RGB to color name
        color_name = rgb_to_color_name(mean_color)
        return color_name, mean_color
        
    except Exception as e:
        print(f"Error in color detection: {e}")
        return "Unknown", [128, 128, 128]

def rgb_to_color_name(rgb):
    """
    Convert RGB values to approximate color names.
    Input: RGB array [r, g, b] with values 0-255
    """
    r, g, b = int(rgb[0]), int(rgb[1]), int(rgb[2])
    
    # Ensure values are in valid range
    r = max(0, min(255, r))
    g = max(0, min(255, g))
    b = max(0, min(255, b))
    
    # Define color ranges (order matters - most specific first)
    # White and light colors
    if r > 200 and g > 200 and b > 200:
        return "White"
    # Black and very dark colors
    elif r < 60 and g < 60 and b < 60:
        return "Black"
    # Red colors (using more sensitive thresholds)
    elif r > 90 and g < 70 and b < 70:
        return "Red"
    # Blue colors
    elif r < 100 and g < 100 and b > 150:
        return "Blue"
    # Green colors
    elif r < 100 and g > 150 and b < 100:
        return "Green"
    # Yellow colors
    elif r > 150 and g > 150 and b < 100:
        return "Yellow"
    # Orange colors
    elif r > 150 and g > 100 and b < 100:
        return "Orange"
    # Purple/Magenta colors
    elif r > 150 and g < 150 and b > 150:
        return "Purple"
    # Cyan colors
    elif r < 150 and g > 150 and b > 150:
        return "Cyan"
    # Brown colors
    elif r > 120 and g > 80 and b < 80 and r > g and g > b:
        return "Brown"
    # Pink colors
    elif r > 180 and g > 150 and b > 150:
        return "Pink"
    # Gray colors (similar RGB values)
    elif abs(r - g) < 30 and abs(g - b) < 30 and abs(r - b) < 30 and 60 <= r <= 200:
        return "Gray"
    # Silver (lighter gray)
    elif abs(r - g) < 20 and abs(g - b) < 20 and abs(r - b) < 20 and r > 150:
        return "Silver"
    # Dark colors that don't fit black
    elif max(r, g, b) < 100:
        return "Dark"
    else:
        return "Mixed"

# Polygon coordinates from user (HTML image map)
polygon = np.array([[-1,11], [1,411], [1,973], [970,982], [1134,584], [687,498], [712,0], [256,0], [199,12], [179,25]], np.int32)
polygon = polygon.reshape((-1, 1, 2))

print(f"Processing video: {video_path}")
print(f"Output will be saved to: {output_path}")
print(f"Video properties: {width}x{height} @ {fps}fps")

frame_count = 0
total_helmets_detected = 0
total_without_helmets_detected = 0
total_motorcycles_detected = 0
total_bicycles_detected = 0
helmet_colors_detected = Counter()  # Track helmet colors

while True:
    success, img = cap.read()
    if not success:
        print("End of video reached")
        break
    
    frame_count += 1
    if frame_count % 30 == 0:  # Print progress every 30  frames
        print(f"Processing frame {frame_count}")
    
    # First, detect vehicles (motorcycles/bicycles) using YOLOv8m
    vehicle_results = vehicle_model(img, stream=True, conf=0.5)
    motorcycles_detected = []
    bicycles_detected = []
    
    for r in vehicle_results:
        boxes = r.boxes
        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cls = int(box.cls[0])
                conf = math.ceil((box.conf[0] * 100)) / 100
                
                # Calculate center of bounding box
                cx, cy = x1 + (x2 - x1) // 2, y1 + (y2 - y1) // 2
                
                # Check if center is inside polygon
                if cv2.pointPolygonTest(polygon, (cx, cy), False) >= 0:
                    if cls == MOTORCYCLE_CLASS:
                        motorcycles_detected.append((x1, y1, x2, y2, conf))
                        total_motorcycles_detected += 1
                        
                        # Draw motorcycle bounding box in blue
                        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                        cvzone.putTextRect(img, f'Motorcycle {conf}', (max(0, x1), max(35, y1)), scale=0.8, thickness=1)
                        
    
    # Detect helmets using the custom helmet model
    helmet_results = helmet_model(img, stream=True, conf=0.3) 
    helmet_detections_found = 0
    for r in helmet_results:
        boxes = r.boxes
        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx, cy = x1 + (x2 - x1) // 2, y1 + (y2 - y1) // 2
                conf = round(box.conf[0].item(), 2)
                cls = int(box.cls[0])
                
                # Check if center is inside polygon
                if cv2.pointPolygonTest(polygon, (cx, cy), False) >= 0:
                    if cls == 0:  # With helmet
                        total_helmets_detected += 1
                        
                        # Detect helmet color
                        color_name, color_rgb = detect_helmet_color(img, x1, y1, x2, y2)
                        helmet_colors_detected[color_name] += 1
                        
                        # Draw green rectangle for with helmet
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        
                        # Display helmet detection with color information
                        text = f'With Helmet {conf} - {color_name}'
                        cvzone.putTextRect(img, text, (max(0, x1), max(35, y1)), scale=0.8, thickness=1)
                        
                        # Add a small color indicator rectangle
                        color_bgr = (int(color_rgb[2]), int(color_rgb[1]), int(color_rgb[0]))  # Convert RGB to BGR
                        cv2.rectangle(img, (x1, y2-20), (x1+60, y2), color_bgr, -1)
                        cv2.rectangle(img, (x1, y2-20), (x1+60, y2), (0, 0, 0), 1)  # Black border
                        
                    elif cls == 1:  # Without helmet
                        total_without_helmets_detected += 1
                        helmet_detections_found += 1
                        # Draw red rectangle for without helmet
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cvzone.putTextRect(img, f'Without Helmet {conf}', (max(0, x1), max(35, y1)), scale=0.8, thickness=1)

    if frame_count % 60 == 0 and helmet_detections_found > 0:
        print(f"  🚨  Without helmet detections found: {helmet_detections_found}")
        
    # Draw the polygon on the frame for visualization
    cv2.polylines(img, [polygon], isClosed=True, color=(0,255,255), thickness=2)
    
    # Write frame to output video
    out.write(img)

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Video processing complete! Output saved to: {output_path}")
print(f"Total frames processed: {frame_count}")
print(f"\n📊 Detection Summary:")
print(f"   🏍️  Motorcycles detected: {total_motorcycles_detected}")
print(f"   🪖  With Helmets detected: {total_helmets_detected}")
print(f"   🚨  Without Helmets detected: {total_without_helmets_detected}")

print(f"\n🪖 Helmet Color Distribution:")
if helmet_colors_detected:
    for color, count in helmet_colors_detected.most_common():
        percentage = (count / sum(helmet_colors_detected.values())) * 100
        print(f"   🎨 {color}: {count} detections ({percentage:.1f}%)")
else:
    print("   No helmet colors detected")

print(f"\n🎨 Detection Colors:")
print(f"   🔵 Blue = Motorcycles")
print(f"   🟠 Orange = Bicycles") 
print(f"   🟢 Green = With Helmet")
print(f"   🔴 Red = Without Helmet")