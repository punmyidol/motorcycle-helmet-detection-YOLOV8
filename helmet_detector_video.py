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
output_path = "media/test1-detected.mp4"
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
polygon_motorcycle = np.array([
    [8, 647],
    [3, 1301],
    [1814, 1299],
    [700, 700],
    [700, 390]
])

polygon_motorcycle = polygon_motorcycle.reshape((-1, 1, 2))
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
                if cv2.pointPolygonTest(polygon_motorcycle, (cx, cy), False) >= 0:
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
                        
    
    # Detect helmets only for detected motorcycles
    helmet_detections_found = 0
    for motorcycle in motorcycles_detected:
        mx1, my1, mx2, my2, mconf = motorcycle
        
        # Expand motorcycle bounding box slightly to include potential helmet area
        # Helmets are typically above the motorcycle, so expand upward
        expansion_factor = 0.3  # 30% expansion
        width = mx2 - mx1
        height = my2 - my1
        
        # Expand upward more than other directions for helmet detection
        expanded_x1 = max(0, int(mx1 - width * expansion_factor * 0.5))
        expanded_y1 = max(0, int(my1 - height * expansion_factor * 1.5))  # More expansion upward
        expanded_x2 = min(img.shape[1], int(mx2 + width * expansion_factor * 0.5))
        expanded_y2 = min(img.shape[0], int(my2 + height * expansion_factor * 0.5))
        
        # Extract expanded ROI for helmet detection
        motorcycle_roi = img[expanded_y1:expanded_y2, expanded_x1:expanded_x2]
        
        if motorcycle_roi.size > 0:
            # Detect helmets in the motorcycle ROI
            helmet_results = helmet_model(motorcycle_roi, stream=True, conf=0.45)
            
            for r in helmet_results:
                boxes = r.boxes
                if boxes is not None:
                    for box in boxes:
                        hx1, hy1, hx2, hy2 = map(int, box.xyxy[0])
                        conf = round(box.conf[0].item(), 2)
                        cls = int(box.cls[0])
                        
                        # Convert ROI coordinates back to full image coordinates
                        full_hx1 = hx1 + expanded_x1
                        full_hy1 = hy1 + expanded_y1
                        full_hx2 = hx2 + expanded_x1
                        full_hy2 = hy2 + expanded_y1
                        
                        # Calculate center of helmet detection
                        hcx, hcy = full_hx1 + (full_hx2 - full_hx1) // 2, full_hy1 + (full_hy2 - full_hy1) // 2
                        
                        # Check if helmet center is inside the general polygon
                        if cv2.pointPolygonTest(polygon, (hcx, hcy), False) >= 0:
                            if cls == 0:  # With helmet
                                total_helmets_detected += 1
                                
                                # Draw green rectangle for with helmet
                                cv2.rectangle(img, (full_hx1, full_hy1), (full_hx2, full_hy2), (0, 255, 0), 2)
                                cvzone.putTextRect(img, f'With Helmet {conf}', (max(0, full_hx1), max(35, full_hy1)), scale=0.8, thickness=1)
                                
                                # Debug output for with helmet detections
                                if frame_count % 60 == 0:  # Print every 60 frames
                                    print(f"  ✅  With helmet detected on motorcycle: confidence {conf}")
                                
                            elif cls == 1:  # Without helmet
                                total_without_helmets_detected += 1
                                helmet_detections_found += 1
                                # Draw red rectangle for without helmet
                                cv2.rectangle(img, (full_hx1, full_hy1), (full_hx2, full_hy2), (0, 0, 255), 2)
                                cvzone.putTextRect(img, f'Without Helmet {conf}', (max(0, full_hx1), max(35, full_hy1)), scale=0.8, thickness=1)
                                
                                # Debug output for without helmet detections
                                if frame_count % 60 == 0:  # Print every 60 frames
                                    print(f"  🚨  Without helmet detected on motorcycle: confidence {conf}")
            
            # Draw expanded ROI rectangle for debugging (optional)
            if frame_count % 60 == 0:  # Only draw occasionally to avoid clutter
                cv2.rectangle(img, (expanded_x1, expanded_y1), (expanded_x2, expanded_y2), (255, 255, 0), 1)

    if frame_count % 60 == 0 and helmet_detections_found > 0:
        print(f"  🚨  Without helmet detections found: {helmet_detections_found}")
        
    # Draw the polygon on the frame for visualization
    # Ensure polygon coordinates are within frame bounds
    frame_height, frame_width = img.shape[:2]
    
    # Clip polygon coordinates to frame bounds
    polygon_clipped = np.clip(polygon.reshape(-1, 2), [0, 0], [frame_width-1, frame_height-1])
    polygon_motorcycle_clipped = np.clip(polygon_motorcycle.reshape(-1, 2), [0, 0], [frame_width-1, frame_height-1])
    
    # Reshape back to the format needed for polylines
    polygon_clipped = polygon_clipped.reshape((-1, 1, 2))
    polygon_motorcycle_clipped = polygon_motorcycle_clipped.reshape((-1, 1, 2))
    
    # Draw polygons with different colors
    cv2.polylines(img, [polygon_clipped], isClosed=True, color=(0,255,255), thickness=3)  # Yellow for general polygon
    cv2.polylines(img, [polygon_motorcycle_clipped], isClosed=True, color=(0,255,0), thickness=3)  # Green for motorcycle polygon
    
    # Add labels for the polygons
    cv2.putText(img, "General Detection Area", (polygon_clipped[0][0][0], polygon_clipped[0][0][1]-10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
    cv2.putText(img, "Motorcycle Detection Area", (polygon_motorcycle_clipped[0][0][0], polygon_motorcycle_clipped[0][0][1]-10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    
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