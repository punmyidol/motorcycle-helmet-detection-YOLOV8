#!/usr/bin/env python3
"""
Live Helmet Detection Script using RKNN models - Optimized for RDK X5 NPU
Real-time helmet detection from camera feed using RKNN accelerated inference
RDK X5 NPU optimizations for maximum performance
"""

import cv2
import math
import numpy as np
from collections import Counter
import os
import sys
import time
from datetime import datetime
import threading
import subprocess
import gc
import psutil
import platform
from rknn.api import RKNN

print('🔴 LIVE HELMET DETECTION SYSTEM - RDK X5 NPU OPTIMIZED')
print(f'Available memory: {psutil.virtual_memory().available // (1024**2)} MB')

# RDK X5 NPU Configuration
# Optimized settings for RDK X5 NPU hardware acceleration
RDK_X5_CONFIG = {
    'max_det': 50,          # Reduced max detections for NPU efficiency
    'imgsz': 640,           # Optimal image size for RDK X5 NPU (640x640)
    'batch_size': 1,        # Single batch for real-time processing
    'half': True,           # Use FP16 for NPU acceleration
    'device': 'auto',       # Auto-detect NPU/GPU/CPU
    'agnostic_nms': True,   # Faster NMS for real-time performance
    'verbose': False,       # Reduce logging overhead
    'stream_buffer': False, # Disable buffering for real-time
    'workers': 1,           # Single worker for NPU
    'save': False,          # Don't save predictions to reduce I/O
}



# Initialize camera capture (0 for default camera, 1 for external camera)
camera_index = 0
cap = cv2.VideoCapture(camera_index)

# Check if camera is available
if not cap.isOpened():
    for i in range(1, 4):  # Try cameras 1, 2, 3
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            camera_index = i
            break
    else:
        print("❌ No camera found. Please check your camera connection.")
        sys.exit(1)

# Set camera properties optimized for RDK X5 NPU
cap.set(cv2.CAP_PROP_FRAME_WIDTH, RDK_X5_CONFIG['imgsz'])  # Use NPU-optimized resolution
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, RDK_X5_CONFIG['imgsz']) 
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer for real-time processing

# Get actual camera properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# RKNN Model Class for NPU inference
class RKNNModel:
    def __init__(self, model_path, target_platform='rk3588'):
        self.model_path = model_path
        self.rknn = RKNN(verbose=False)
        self.target_platform = target_platform
        self.input_size = RDK_X5_CONFIG['imgsz']
        self.load_model()
    
    def load_model(self):
        """Load RKNN model for NPU inference"""
        try:
            print(f"🔄 Loading RKNN model: {self.model_path}")
            
            # Load RKNN model
            ret = self.rknn.load_rknn(self.model_path)
            if ret != 0:
                raise Exception(f"Load RKNN model failed with code: {ret}")
            
            # Initialize runtime environment
            ret = self.rknn.init_runtime(target=self.target_platform)
            if ret != 0:
                raise Exception(f"Init runtime failed with code: {ret}")
            
            print(f"✅ RKNN model loaded successfully: {os.path.basename(self.model_path)}")
            
        except Exception as e:
            print(f"❌ Error loading RKNN model {self.model_path}: {e}")
            raise
    
    def preprocess_image(self, img):
        """Preprocess image for RKNN inference"""
        # Resize image to model input size
        if img.shape[:2] != (self.input_size, self.input_size):
            img = cv2.resize(img, (self.input_size, self.input_size))
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1] and expand dimensions for batch
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)
        
        return img
    
    def inference(self, img):
        """Run inference on preprocessed image"""
        try:
            # Preprocess image
            input_data = self.preprocess_image(img)
            
            # Run inference
            outputs = self.rknn.inference(inputs=[input_data])
            
            return outputs
        except Exception as e:
            print(f"❌ RKNN inference error: {e}")
            return None
    
    def __del__(self):
        """Clean up RKNN resources"""
        try:
            if hasattr(self, 'rknn'):
                self.rknn.release()
        except:
            pass

def parse_rknn_vehicle_output(outputs, conf_threshold=0.5, input_shape=(640, 640)):
    """Parse RKNN vehicle detection output to extract bounding boxes and classes"""
    detections = []
    
    if outputs is None or len(outputs) == 0:
        return detections
    
    try:
        # RKNN YOLOv8 output format: [batch, num_detections, 85] 
        # where 85 = [x_center, y_center, width, height, conf, class0_conf, class1_conf, ...]
        output = outputs[0]  # Get first output
        
        if len(output.shape) == 3:
            output = output[0]  # Remove batch dimension
        
        # Transpose if needed (some RKNN models output [85, num_detections])
        if output.shape[0] == 85:
            output = output.T
        
        for detection in output:
            if len(detection) < 85:
                continue
                
            # Extract box coordinates and confidence
            x_center, y_center, width, height = detection[:4]
            obj_conf = detection[4]
            class_confs = detection[5:]
            
            # Filter by confidence threshold
            if obj_conf < conf_threshold:
                continue
            
            # Get class with highest confidence
            class_id = np.argmax(class_confs)
            class_conf = class_confs[class_id]
            final_conf = obj_conf * class_conf
            
            if final_conf < conf_threshold:
                continue
            
            # Convert center coordinates to corner coordinates
            x1 = int(x_center - width / 2)
            y1 = int(y_center - height / 2)
            x2 = int(x_center + width / 2)
            y2 = int(y_center + height / 2)
            
            # Ensure coordinates are within bounds
            x1 = max(0, min(x1, input_shape[1]))
            y1 = max(0, min(y1, input_shape[0]))
            x2 = max(0, min(x2, input_shape[1]))
            y2 = max(0, min(y2, input_shape[0]))
            
            detections.append({
                'bbox': [x1, y1, x2, y2],
                'confidence': final_conf,
                'class': class_id
            })
    
    except Exception as e:
        print(f"Error parsing RKNN vehicle output: {e}")
        
    return detections

def parse_rknn_helmet_output(outputs, conf_threshold=0.45, input_shape=(640, 640)):
    """Parse RKNN helmet detection output to extract bounding boxes and classes"""
    detections = []
    
    if outputs is None or len(outputs) == 0:
        return detections
    
    try:
        # RKNN YOLOv8 output format for helmet model: [batch, num_detections, 7]
        # where 7 = [x_center, y_center, width, height, conf, class0_conf, class1_conf]
        output = outputs[0]  # Get first output
        
        if len(output.shape) == 3:
            output = output[0]  # Remove batch dimension
        
        # Transpose if needed
        if output.shape[0] == 7:
            output = output.T
        
        for detection in output:
            if len(detection) < 7:
                continue
                
            # Extract box coordinates and confidence
            x_center, y_center, width, height = detection[:4]
            obj_conf = detection[4]
            class_confs = detection[5:7]  # Only 2 classes for helmet detection
            
            # Filter by confidence threshold
            if obj_conf < conf_threshold:
                continue
            
            # Get class with highest confidence
            class_id = np.argmax(class_confs)
            class_conf = class_confs[class_id]
            final_conf = obj_conf * class_conf
            
            if final_conf < conf_threshold:
                continue
            
            # Convert center coordinates to corner coordinates
            x1 = int(x_center - width / 2)
            y1 = int(y_center - height / 2)
            x2 = int(x_center + width / 2)
            y2 = int(y_center + height / 2)
            
            # Ensure coordinates are within bounds
            x1 = max(0, min(x1, input_shape[1]))
            y1 = max(0, min(y1, input_shape[0]))
            x2 = max(0, min(x2, input_shape[1]))
            y2 = max(0, min(y2, input_shape[0]))
            
            detections.append({
                'bbox': [x1, y1, x2, y2],
                'confidence': final_conf,
                'class': class_id
            })
    
    except Exception as e:
        print(f"Error parsing RKNN helmet output: {e}")
        
    return detections

# Load RKNN models with RDK X5 NPU optimizations
try:
    print("🔄 Initializing RKNN models for NPU acceleration...")
    
    # Check if RKNN models exist
    vehicle_rknn_path = "weights/rknn/yolov8m.rknn"
    helmet_rknn_path = "weights/rknn/helmet_detection_trained.rknn"
    
    if not os.path.exists(vehicle_rknn_path):
        print(f"❌ Vehicle RKNN model not found: {vehicle_rknn_path}")
        print("🔧 Please run 'python export_models_to_rknn.py' first to convert models")
        sys.exit(1)
    
    if not os.path.exists(helmet_rknn_path):
        print(f"❌ Helmet RKNN model not found: {helmet_rknn_path}")
        print("🔧 Please run 'python export_models_to_rknn.py' first to convert models")
        sys.exit(1)
    
    # Initialize RKNN models
    vehicle_model = RKNNModel(vehicle_rknn_path)
    helmet_model = RKNNModel(helmet_rknn_path)
    
    print("✅ All RKNN models loaded successfully for NPU acceleration!")
    
except Exception as e:
    print(f"❌ Error loading RKNN models: {e}")
    print("💡 Fallback: Attempting to use original YOLO models...")
    try:
        from ultralytics import YOLO
        vehicle_model = YOLO("weights/yolov8m.pt")
        helmet_model = YOLO("weights/helmet_detection_trained.pt")
        print("✅ Fallback YOLO models loaded (CPU/GPU mode)")
        USING_RKNN = False
    except Exception as fallback_error:
        print(f"❌ Fallback also failed: {fallback_error}")
        sys.exit(1)
else:
    USING_RKNN = True

# Define class names for helmet detection
classNames = ['With Helmet', 'Without Helmet']

MOTORCYCLE_CLASS = 3  # motorcycle in COCO
BICYCLE_CLASS = 1     # bicycle in COCO

# Normalized polygon coordinates as fractions of width and height
# These will be converted to actual pixel coordinates based on camera dimensions
polygon_normalized = np.array([
    [0.1, 0.1],     # Top left
    [0.1, 0.9],     # Bottom left
    [0.9, 0.9],     # Bottom right
    [0.9, 0.1]      # Top right
])

polygon_motorcycle_normalized = np.array([
    [0.004, 0.53],  # Bottom left
    [0.002, 0.95],  # Far bottom left (clipped from 1.20 to 0.95)
    [0.94, 0.95],   # Far bottom right (clipped from 1.20 to 0.95)
    [0.42, 0.63],   # Bottom middle
    [0.42, 0.30]    # Upper middle
])

# Convert normalized coordinates to actual pixel coordinates based on camera dimensions
def convert_normalized_to_pixel_coordinates(normalized_coords, width, height):
    """Convert normalized coordinates (0-1) to pixel coordinates"""
    pixel_coords = []
    for coord in normalized_coords:
        x = int(coord[0] * width)
        y = int(coord[1] * height)
        pixel_coords.append([x, y])
    return np.array(pixel_coords, dtype=np.int32)

# Convert normalized coordinates to actual pixel coordinates
polygon = convert_normalized_to_pixel_coordinates(polygon_normalized, width, height)
polygon_motorcycle = convert_normalized_to_pixel_coordinates(polygon_motorcycle_normalized, width, height)

# Reshape for OpenCV functions
polygon = polygon.reshape((-1, 1, 2))
polygon_motorcycle = polygon_motorcycle.reshape((-1, 1, 2))

print("🔴 Starting live detection with polygon cropping optimization...")
print(f"Polygon area: {cv2.contourArea(polygon):.0f} pixels")

# Initialize counters and statistics
frame_count = 0
total_helmets_detected = 0
total_without_helmets_detected = 0
total_motorcycles_detected = 0
total_bicycles_detected = 0
helmet_colors_detected = Counter()  # Track helmet colors
motorcycle_colors_detected = Counter()  # Track motorcycle colors

def crop_to_polygon(img, polygon):
    """Crop image to polygon bounding box and create mask for polygon area"""
    # Get bounding rectangle of polygon
    x, y, w, h = cv2.boundingRect(polygon)
    
    # Ensure coordinates are within image bounds
    x = max(0, x)
    y = max(0, y)
    w = min(w, img.shape[1] - x)
    h = min(h, img.shape[0] - y)
    
    # Return None if invalid crop area
    if w <= 0 or h <= 0:
        return None, (0, 0), None
    
    # Crop image to bounding rectangle
    cropped_img = img[y:y+h, x:x+w]
    
    # Create mask for the polygon within the cropped area
    polygon_shifted = polygon.copy()
    polygon_shifted[:, :, 0] -= x  # Shift x coordinates
    polygon_shifted[:, :, 1] -= y  # Shift y coordinates
    
    # Create mask
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [polygon_shifted], 255)
    
    # Apply mask to cropped image (set non-polygon areas to black)
    masked_img = cropped_img.copy()
    masked_img[mask == 0] = [0, 0, 0]
    
    return masked_img, (x, y), mask

def adjust_detections_to_full_frame(detections, crop_offset):
    """Adjust detection coordinates from cropped frame back to full frame"""
    adjusted_detections = []
    offset_x, offset_y = crop_offset
    
    for detection in detections:
        x1, y1, x2, y2 = detection[:4]
        # Adjust coordinates back to full frame
        x1_adj = x1 + offset_x
        y1_adj = y1 + offset_y
        x2_adj = x2 + offset_x
        y2_adj = y2 + offset_y
        
        # Keep other detection data (confidence, etc.)
        adjusted = (x1_adj, y1_adj, x2_adj, y2_adj) + detection[4:]
        adjusted_detections.append(adjusted)
    
    return adjusted_detections

def detect_color(img):
    """Detect dominant color in image region - same logic as helmet_detector_video.py"""
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

# Performance tracking
fps_counter = 0
fps_start_time = time.time()
current_fps = 0

# RDK X5 NPU Performance optimizations
frame_skip_count = 0
frame_skip_interval = 2  # Process every 2nd frame for better performance
memory_cleanup_interval = 100  # Clean memory every 100 frames

# Audio alert system
last_alert_time = 0
alert_cooldown = 5.0  # Minimum seconds between alerts to prevent spam
audio_enabled = True

def play_alert_sound(message="Please wear a helmet"):
    """Play audio alert in a separate thread to avoid blocking - Linux compatible"""
    def play_audio():
        try:
            # Detect operating system for audio compatibility
            current_os = platform.system().lower()
            
            if current_os == 'linux':
                # Try espeak first (text-to-speech)
                try:
                    subprocess.run(['espeak', message], check=True, capture_output=True, timeout=3)
                    return
                except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
                    pass
                
                # Try festival as alternative TTS
                try:
                    subprocess.run(['festival', '--tts'], input=message, 
                                 text=True, check=True, capture_output=True, timeout=3)
                    return
                except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
                    pass
                
                # Try aplay with a beep sound file if available
                beep_files = ['/usr/share/sounds/alsa/Rear_Right.wav', 
                             '/usr/share/sounds/sound-icons/bell.wav',
                             '/usr/share/sounds/ubuntu/notifications/Xylo.ogg']
                
                for beep_file in beep_files:
                    if os.path.exists(beep_file):
                        try:
                            subprocess.run(['aplay', beep_file], check=True, 
                                         capture_output=True, timeout=2)
                            return
                        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
                            continue
                
                # Try paplay (PulseAudio) as alternative
                for beep_file in beep_files:
                    if os.path.exists(beep_file):
                        try:
                            subprocess.run(['paplay', beep_file], check=True, 
                                         capture_output=True, timeout=2)
                            return
                        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
                            continue
            
            elif current_os == 'darwin':  # macOS
                subprocess.run(['say', message], check=True, capture_output=True, timeout=3)
                return
            
            elif current_os == 'windows':  # Windows
                # Use Windows SAPI for text-to-speech
                import winsound
                winsound.MessageBeep()
                return
            
            # Final fallback: system beep
            print('\a')  # ASCII bell character
            print(f"🔊 AUDIO ALERT: {message}")
            
        except Exception as e:
            print(f"Audio alert failed: {e}")
            print('\a')  # ASCII bell fallback
            print(f"🔊 AUDIO ALERT: {message}")
    
    # Run audio in separate thread to avoid blocking main detection loop
    if audio_enabled:
        audio_thread = threading.Thread(target=play_audio, daemon=True)
        audio_thread.start()

def should_play_alert():
    """Check if enough time has passed since last alert"""
    global last_alert_time
    current_time = time.time()
    if current_time - last_alert_time >= alert_cooldown:
        last_alert_time = current_time
        return True
    return False

# Create media directory if it doesn't exist
os.makedirs("media", exist_ok=True)

try:
    while True:
        success, img = cap.read()
        if not success:
            print("❌ Failed to read from camera")
            break
        
        frame_count += 1
        frame_skip_count += 1
        
        # Calculate FPS
        fps_counter += 1
        if fps_counter % 30 == 0:  # Update FPS every 30 frames
            current_time = time.time()
            current_fps = 30 / (current_time - fps_start_time)
            fps_start_time = current_time
        
        # RDK X5 NPU Optimization: Skip frames for better performance
        skip_detection = frame_skip_count % frame_skip_interval != 0
        
        # Memory cleanup for NPU efficiency
        if frame_count % memory_cleanup_interval == 0:
            gc.collect()  # Force garbage collection
        
        # Crop frame to polygon area before YOLO processing
        if not skip_detection:
            cropped_img, crop_offset, polygon_mask = crop_to_polygon(img, polygon)
        else:
            cropped_img, crop_offset, polygon_mask = None, None, None
        
        motorcycles_detected = []
        bicycles_detected = []
        
        # First, detect vehicles (motorcycles/bicycles) using RKNN model on cropped area
        if not skip_detection and cropped_img is not None:
            if USING_RKNN:
                vehicle_results = vehicle_model.inference(cropped_img)
            else:
                vehicle_results = vehicle_model(cropped_img, 
                                              stream=False,
                                              conf=0.5, 
                                              verbose=RDK_X5_CONFIG['verbose'],
                                              max_det=RDK_X5_CONFIG['max_det'],
                                              agnostic_nms=RDK_X5_CONFIG['agnostic_nms'])
            
            # Process vehicle detection results
            if USING_RKNN:
                if vehicle_results is not None:
                    detections = parse_rknn_vehicle_output(vehicle_results, conf_threshold=0.5)
                    for detection in detections:
                        x1, y1, x2, y2 = detection['bbox']
                        cls = detection['class']
                        conf = detection['confidence']
                        
                        # Adjust coordinates back to full frame
                        x1_full = x1 + crop_offset[0]
                        y1_full = y1 + crop_offset[1] 
                        x2_full = x2 + crop_offset[0]
                        y2_full = y2 + crop_offset[1]
                        
                        if cls == MOTORCYCLE_CLASS:
                            # Calculate center of motorcycle bounding box
                            cx, cy = x1_full + (x2_full - x1_full) // 2, y1_full + (y2_full - y1_full) // 2
                            
                            # Check if center is inside polygon_motorcycle
                            if cv2.pointPolygonTest(polygon_motorcycle, (cx, cy), False) >= 0:
                                motorcycles_detected.append((x1_full, y1_full, x2_full, y2_full, conf))
                                total_motorcycles_detected += 1
                                
                                # Extract motorcycle ROI and detect color (from full frame)
                                motorcycle_roi = img[y1_full:y2_full, x1_full:x2_full]
                                if motorcycle_roi.size > 0:  # Ensure ROI is not empty
                                    color_name, hue_value = detect_color(motorcycle_roi)
                                    motorcycle_colors_detected[color_name] += 1
                                
                        elif cls == BICYCLE_CLASS:
                            bicycles_detected.append((x1_full, y1_full, x2_full, y2_full, conf))
                            total_bicycles_detected += 1
            else:
                if vehicle_results and len(vehicle_results) > 0:
                    boxes = vehicle_results[0].boxes  # Get first (and only) result since stream=False
                    if boxes is not None:
                        for box in boxes:
                            x1, y1, x2, y2 = box.xyxy[0]
                            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                            cls = int(box.cls[0])
                            conf = math.ceil((box.conf[0] * 100)) / 100
                            
                            # Adjust coordinates back to full frame
                            x1_full = x1 + crop_offset[0]
                            y1_full = y1 + crop_offset[1] 
                            x2_full = x2 + crop_offset[0]
                            y2_full = y2 + crop_offset[1]
                            
                            if cls == MOTORCYCLE_CLASS:
                                # Calculate center of motorcycle bounding box
                                cx, cy = x1_full + (x2_full - x1_full) // 2, y1_full + (y2_full - y1_full) // 2
                                
                                # Check if center is inside polygon_motorcycle
                                if cv2.pointPolygonTest(polygon_motorcycle, (cx, cy), False) >= 0:
                                    motorcycles_detected.append((x1_full, y1_full, x2_full, y2_full, conf))
                                    total_motorcycles_detected += 1
                                    
                                    # Extract motorcycle ROI and detect color (from full frame)
                                    motorcycle_roi = img[y1_full:y2_full, x1_full:x2_full]
                                    if motorcycle_roi.size > 0:  # Ensure ROI is not empty
                                        color_name, hue_value = detect_color(motorcycle_roi)
                                        motorcycle_colors_detected[color_name] += 1
                                    
                            elif cls == BICYCLE_CLASS:
                                bicycles_detected.append((x1_full, y1_full, x2_full, y2_full, conf))
                                total_bicycles_detected += 1
                            
        helmet_detections_found = 0
        current_frame_helmets = 0
        current_frame_no_helmets = 0
        
        # Detect helmets only for detected motorcycles
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
            
            if motorcycle_roi.size > 0 and not skip_detection:
                # Detect helmets in the motorcycle ROI
                if USING_RKNN:
                    helmet_results = helmet_model.inference(motorcycle_roi)
                else:
                    helmet_results = helmet_model(motorcycle_roi, 
                                                stream=False,
                                                conf=0.45, 
                                                verbose=RDK_X5_CONFIG['verbose'],
                                                max_det=RDK_X5_CONFIG['max_det'],
                                                agnostic_nms=RDK_X5_CONFIG['agnostic_nms'])
                
                # Process helmet detection results
                if USING_RKNN:
                    if helmet_results is not None:
                        detections = parse_rknn_helmet_output(helmet_results, conf_threshold=0.45)
                        for detection in detections:
                            hx1, hy1, hx2, hy2 = detection['bbox']
                            conf = detection['confidence']
                            cls = detection['class']
                            
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
                                    current_frame_helmets += 1
                                    
                                elif cls == 1:  # Without helmet
                                    total_without_helmets_detected += 1
                                    current_frame_no_helmets += 1
                                    helmet_detections_found += 1
                                    
                                    # Play audio alert (with cooldown to prevent spam)
                                    if should_play_alert():
                                        play_alert_sound("Wear a helmet")
                else:
                    if helmet_results and len(helmet_results) > 0:
                        boxes = helmet_results[0].boxes  # Get first (and only) result since stream=False
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
                                        current_frame_helmets += 1
                                        
                                    elif cls == 1:  # Without helmet
                                        total_without_helmets_detected += 1
                                        current_frame_no_helmets += 1
                                        helmet_detections_found += 1
                                        
                                        # Play audio alert (with cooldown to prevent spam)
                                        if should_play_alert():
                                            play_alert_sound("Wear a helmet")
                            


        
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
        # Display the frame
        cv2.imshow('Live Helmet Detection', img)
        
        # Handle key presses with NPU optimization controls
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            # Save screenshot
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            screenshot_path = f"media/live_detection_screenshot_{timestamp}.jpg"
            cv2.imwrite(screenshot_path, img)
        elif key == ord('m'):
            # Toggle audio alerts
            audio_enabled = not audio_enabled
        elif key == ord('+') or key == ord('='):
            # Increase performance (reduce frame skip interval)
            if frame_skip_interval > 1:
                frame_skip_interval -= 1
        elif key == ord('-'):
            # Decrease performance (increase frame skip interval)
            if frame_skip_interval < 5:
                frame_skip_interval += 1

except KeyboardInterrupt:
    pass

except Exception as e:
    print(f"❌ Error: {e}")

finally:
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"\n📊 Session Summary:")
    print(f"Frames: {frame_count} | Motorcycles: {total_motorcycles_detected} | Bicycles: {total_bicycles_detected}")
    print(f"With Helmets: {total_helmets_detected} | Without Helmets: {total_without_helmets_detected}")
    if motorcycle_colors_detected:
        colors = ", ".join([f"{color}: {count}" for color, count in motorcycle_colors_detected.most_common()])
        print(f"Motorcycle Colors: {colors}")
