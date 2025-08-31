#!/usr/bin/env python3
"""
Optimized Live Helmet Detection Script using YOLOv8
High-performance real-time helmet detection with FPS optimizations
"""

import cv2
import math
import cvzone
from ultralytics import YOLO
import numpy as np
from collections import Counter
import os
import sys
import time
from datetime import datetime
import threading
import subprocess
import torch

print('='*60)
print('🚀 OPTIMIZED LIVE HELMET DETECTION SYSTEM')
print('='*60)
print(f'DEBUG: Executing file: {__file__}')
print(f'DEBUG: Working directory: {os.getcwd()}')
print(f'DEBUG: Python executable: {sys.executable}')
print('='*60)

# Initialize camera capture (0 for default camera, 1 for external camera)
camera_index = 0
cap = cv2.VideoCapture(camera_index)

# Check if camera is available
if not cap.isOpened():
    print(f"❌ Error: Could not open camera {camera_index}")
    print("💡 Trying alternative camera indices...")
    for i in range(1, 4):  # Try cameras 1, 2, 3
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            camera_index = i
            print(f"✅ Found camera at index {i}")
            break
    else:
        print("❌ No camera found. Please check your camera connection.")
        sys.exit(1)

# OPTIMIZED: Lower resolution for better performance
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)   # Reduced from 1280
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Reduced from 720
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)      # Reduce buffer to prevent lag

# Get actual camera properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

print(f"📹 Camera initialized: {width}x{height} @ {fps}fps")

# Load all YOLO models
print("🔄 Loading YOLO models...")
try:
    # OPTIMIZED: Use smaller, faster models
    vehicle_model = YOLO("weights/yolov8n.pt")  # Changed from yolov8m to yolov8n (nano - faster)
    helmet_model = YOLO("weights/helmet_detection_trained.pt")
    
    # OPTIMIZED: Set device explicitly for better performance
    device = 'mps' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else 'cpu'
    vehicle_model.to(device)
    helmet_model.to(device)
    print(f"✅ Models loaded successfully on {device}!")
except Exception as e:
    print(f"❌ Error loading models: {e}")
    # Fallback to CPU
    try:
        vehicle_model = YOLO("weights/yolov8n.pt")
        helmet_model = YOLO("weights/helmet_detection_trained.pt")
        print("✅ Models loaded on CPU (fallback)")
    except:
        sys.exit(1)

# Define class names for helmet detection
classNames = ['With Helmet', 'Without Helmet']

MOTORCYCLE_CLASS = 3  # motorcycle in COCO
BICYCLE_CLASS = 1     # bicycle in COCO

# OPTIMIZED: Scale original polygon coordinates to new resolution
# Original coordinates were for 1280x720, now we're using 640x480
original_width, original_height = 1280, 720
scale_x = width / original_width  # 640/1280 = 0.5
scale_y = height / original_height  # 480/720 = 0.667

original_polygon = np.array([[-1,11], [1,411], [1,973], [970,982], [1134,584], [687,498], [712,0], [256,0], [199,12], [179,25]], np.int32)

# Scale polygon coordinates to new resolution
scaled_polygon = []
for point in original_polygon:
    x, y = point
    # Scale and clamp to frame boundaries
    scaled_x = max(0, min(width-1, int(x * scale_x)))
    scaled_y = max(0, min(height-1, int(y * scale_y)))
    scaled_polygon.append([scaled_x, scaled_y])

polygon = np.array(scaled_polygon, np.int32)
polygon = polygon.reshape((-1, 1, 2))

print(f"🎯 Detection area defined")
print(f"📐 Polygon scaled from {original_width}x{original_height} to {width}x{height}")
print(f"📍 Scaled polygon points: {scaled_polygon[:3]}... (showing first 3 points)")
print("🔊 Audio alerts enabled - will announce when no helmet detected")
print("🚀 Starting OPTIMIZED live detection... Press 'q' to quit, 's' to save screenshot, 'm' to toggle audio")

# Initialize counters and statistics
frame_count = 0
total_helmets_detected = 0
total_without_helmets_detected = 0
total_motorcycles_detected = 0
total_bicycles_detected = 0
helmet_colors_detected = Counter()

# Performance tracking
fps_counter = 0
fps_start_time = time.time()
current_fps = 0

# Audio alert system
last_alert_time = 0
alert_cooldown = 5.0  # Minimum seconds between alerts to prevent spam
audio_enabled = True

def play_alert_sound(message="Please wear a helmet"):
    """Play audio alert in a separate thread to avoid blocking"""
    def play_audio():
        try:
            # Use macOS 'say' command for text-to-speech
            subprocess.run(['say', message], check=True, capture_output=True)
        except subprocess.CalledProcessError:
            # Fallback: system beep if 'say' command fails
            print('\a')  # ASCII bell character
        except Exception as e:
            print(f"Audio alert failed: {e}")
    
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

# OPTIMIZED: Skip frame processing for better FPS
frame_skip = 2  # Process every 2nd frame
skip_counter = 0

# Create media directory if it doesn't exist
os.makedirs("media", exist_ok=True)

try:
    while True:
        success, img = cap.read()
        if not success:
            print("❌ Failed to read from camera")
            break
        
        frame_count += 1
        skip_counter += 1
        
        # Calculate FPS
        fps_counter += 1
        if fps_counter % 30 == 0:  # Update FPS every 30 frames
            current_time = time.time()
            current_fps = 30 / (current_time - fps_start_time)
            fps_start_time = current_time
        
        # OPTIMIZED: Skip frames for better performance
        process_frame = (skip_counter % frame_skip == 0)
        
        motorcycles_detected = []
        bicycles_detected = []
        current_frame_helmets = 0
        current_frame_no_helmets = 0
        
        if process_frame:
            # OPTIMIZED: Lower confidence and smaller image size for vehicle detection
            vehicle_results = vehicle_model(img, stream=True, conf=0.6, verbose=False, imgsz=320)
            
            for r in vehicle_results:
                boxes = r.boxes
                if boxes is not None:
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0]
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        cls = int(box.cls[0])
                        conf = math.ceil((box.conf[0] * 100)) / 100
                        
                        # OPTIMIZED: Simplified center calculation
                        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                        
                        # Check if center is inside polygon
                        if cv2.pointPolygonTest(polygon, (cx, cy), False) >= 0:
                            if cls == MOTORCYCLE_CLASS:
                                motorcycles_detected.append((x1, y1, x2, y2, conf))
                                total_motorcycles_detected += 1
                                
                            elif cls == BICYCLE_CLASS:
                                bicycles_detected.append((x1, y1, x2, y2, conf))
                                total_bicycles_detected += 1
            
            # OPTIMIZED: Only run helmet detection if vehicles are detected
            if motorcycles_detected or bicycles_detected:
                helmet_results = helmet_model(img, stream=True, conf=0.45, verbose=False, imgsz=320)
                
                for r in helmet_results:
                    boxes = r.boxes
                    if boxes is not None:
                        for box in boxes:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                            conf = round(box.conf[0].item(), 2)
                            cls = int(box.cls[0])
                            
                            # Check if center is inside polygon
                            if cv2.pointPolygonTest(polygon, (cx, cy), False) >= 0:
                                if cls == 0:  # With helmet
                                    total_helmets_detected += 1
                                    current_frame_helmets += 1
                                    
                                elif cls == 1:  # Without helmet
                                    total_without_helmets_detected += 1
                                    current_frame_no_helmets += 1
                                    
                                    # Play audio alert (with cooldown to prevent spam)
                                    if should_play_alert():
                                        play_alert_sound("Wear a helmet")
        
        # OPTIMIZED: Simplified drawing - only draw detected objects
        for x1, y1, x2, y2, conf in motorcycles_detected:
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(img, f'Motorcycle', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            
        for x1, y1, x2, y2, conf in bicycles_detected:
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 165, 255), 2)
            cv2.putText(img, f'Bicycle', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
        
        # OPTIMIZED: Simplified polygon drawing
        cv2.polylines(img, [polygon], isClosed=True, color=(0, 255, 255), thickness=1)
        
        # OPTIMIZED: Simplified stats overlay
        stats_text = [
            f"FPS: {current_fps:.1f}",
            f"Motorcycles: {len(motorcycles_detected)}",
            f"Bicycles: {len(bicycles_detected)}",
            f"Helmets: {current_frame_helmets}",
            f"No Helmet: {current_frame_no_helmets}"
        ]
        
        for i, text in enumerate(stats_text):
            cv2.putText(img, text, (10, 30 + i*25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # OPTIMIZED: Simplified instructions
        audio_status = "ON" if audio_enabled else "OFF"
        cv2.putText(img, f"q:quit s:screenshot m:audio({audio_status})", 
                   (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Display the frame
        cv2.imshow('Optimized Live Helmet Detection', img)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("\n🛑 Quitting live detection...")
            break
        elif key == ord('s'):
            # Save screenshot
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            screenshot_path = f"media/live_detection_screenshot_optimized_{timestamp}.jpg"
            cv2.imwrite(screenshot_path, img)
            print(f"📸 Screenshot saved: {screenshot_path}")
        elif key == ord('m'):
            # Toggle audio alerts
            audio_enabled = not audio_enabled
            status = "enabled" if audio_enabled else "disabled"
            print(f"🔊 Audio alerts {status}")

except KeyboardInterrupt:
    print("\n🛑 Interrupted by user...")

except Exception as e:
    print(f"❌ Error during live detection: {e}")

finally:
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"\n🏁 Optimized Live Detection Session Complete!")
    print(f"📊 Session Summary:")
    print(f"   ⏱️  Total frames processed: {frame_count}")
    print(f"   🏍️  Total motorcycles detected: {total_motorcycles_detected}")
    print(f"   🚲  Total bicycles detected: {total_bicycles_detected}")
    print(f"   🪖  Total with helmets detected: {total_helmets_detected}")
    print(f"   🚨  Total without helmets detected: {total_without_helmets_detected}")
    print(f"   📹  Average FPS: {current_fps:.1f}")
    
    print(f"\n🎨 Detection Colors:")
    print(f"   🔵 Blue = Motorcycles")
    print(f"   🟠 Orange = Bicycles") 
    print(f"   🟢 Green = With Helmet")
    print(f"   🔴 Red = Without Helmet")
    print(f"   🟡 Yellow = Detection Area")
