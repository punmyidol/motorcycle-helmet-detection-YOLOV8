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

MOTORCYCLE_CLASS = 3  # motorcycle in COCO
BICYCLE_CLASS = 1     # bicycle in COCO

# Polygon coordinates from user (HTML image map)
polygon = np.array([[2141,1031], [700,700], [700,5], [15,8], [9,1685], [1810,1691]], np.int32)
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
motorcycle_colors_detected = Counter()  # Track motorcycle colors

def detect_color(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Mask out very low saturation/brightness for colored pixels
    color_mask = (hsv[:,:,1] > 50) & (hsv[:,:,2] > 50)
    hue_values = hsv[:,:,0][color_mask]

    # Check for black / dark pixels
    dark_mask = hsv[:,:,2] < 50  # low brightness
    dark_ratio = np.sum(dark_mask) / (img.shape[0] * img.shape[1])

    if dark_ratio > 0.5:  # If most pixels are dark, consider it black
        return "Black", None

    if len(hue_values) == 0:
        return "Unknown", None

    # Dominant hue
    hist, bins = np.histogram(hue_values, bins=180, range=[0,180])
    dominant_hue = np.argmax(hist)
    h = dominant_hue

    # Map hue → color
    if (h >= 0 and h <= 10) or (h >= 170 and h <= 179):
        color_name = "Red"
    elif h >= 11 and h <= 20:
        color_name = "Orange"
    elif h >= 21 and h <= 30:
        color_name = "Yellow"
    elif h >= 35 and h <= 85:
        color_name = "Green"
    elif h >= 90 and h <= 130:
        color_name = "Blue"
    elif h >= 131 and h <= 160:
        color_name = "Purple"
    elif h >= 161 and h <= 170:
        color_name = "Pink"
    else:
        color_name = "Unknown"


    return color_name, h

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
                        
                        # Extract motorcycle ROI and detect color
                        motorcycle_roi = img[y1:y2, x1:x2]
                        if motorcycle_roi.size > 0:  # Ensure ROI is not empty
                            color_name, hue_value = detect_color(motorcycle_roi)
                            motorcycle_colors_detected[color_name] += 1
                            
                            # Draw motorcycle bounding box in blue
                            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                            cvzone.putTextRect(img, f'Motorcycle {conf}', (max(0, x1), max(35, y1)), scale=0.8, thickness=1)
                            cvzone.putTextRect(img, f'Color: {color_name}', (max(0, x1), max(55, y1-20)), scale=0.6, thickness=1)
                        else:
                            # Draw motorcycle bounding box in blue (fallback)
                            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                            cvzone.putTextRect(img, f'Motorcycle {conf}', (max(0, x1), max(35, y1)), scale=0.8, thickness=1)
                        
                    elif cls == BICYCLE_CLASS:
                        bicycles_detected.append((x1, y1, x2, y2, conf))
                        total_bicycles_detected += 1
                        
                        # Draw bicycle bounding box in orange
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 165, 255), 2)
                        cvzone.putTextRect(img, f'Bicycle {conf}', (max(0, x1), max(35, y1)), scale=0.8, thickness=1)
                        
    
    # Detect helmets using the custom helmet model
    helmet_results = helmet_model(img, stream=True, conf=0.45) 
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
                        
                        # Draw green rectangle for with helmet
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cvzone.putTextRect(img, f'With Helmet {conf}', (max(0, x1), max(35, y1)), scale=0.8, thickness=1)
                        
                        # Debug output for with helmet detections
                        if frame_count % 60 == 0:  # Print every 60 frames
                            print(f"  ✅  With helmet detected: confidence {conf}")
                        
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
print(f"   🚲  Bicycles detected: {total_bicycles_detected}")
print(f"   🪖  With Helmets detected: {total_helmets_detected}")
print(f"   🚨  Without Helmets detected: {total_without_helmets_detected}")

print(f"\n🎨 Motorcycle Colors Detected:")
if motorcycle_colors_detected:
    for color, count in motorcycle_colors_detected.most_common():
        print(f"   🏍️  {color}: {count}")
else:
    print(f"   No motorcycle colors detected")

print(f"\n🎨 Detection Box Colors:")
print(f"   🔵 Blue = Motorcycles")
print(f"   🟠 Orange = Bicycles") 
print(f"   🟢 Green = With Helmet")
print(f"   🔴 Red = Without Helmet")