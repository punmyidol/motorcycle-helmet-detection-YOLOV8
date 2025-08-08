#!/usr/bin/env python3
"""
Advanced Motorcycle Helmet Detection System
Detects motorcycles, identifies riders, and checks helmet compliance for all riders
"""

import cv2
import numpy as np
from ultralytics import YOLO
import time
from typing import Dict, List, Tuple, Optional
import json

class MotorcycleHelmetDetector:
    def __init__(self, model_path: str = 'yolov8m.pt', confidence: float = 0.3):
        """
        Initialize motorcycle helmet detector
        
        Args:
            model_path: Path to YOLO model
            confidence: Confidence threshold for detections
        """
        try:
            self.model = YOLO(model_path)
            print(f"Loaded model: {model_path}")
        except:
            print(f"Could not load {model_path}, falling back to yolov8n.pt")
            self.model = YOLO('yolov8n.pt')
        
        self.confidence = confidence
        
        # COCO class IDs
        self.person_class = 0
        self.motorcycle_class = 3
        self.bicycle_class = 4
        
        # Detection history for tracking
        self.motorcycle_history = {}
        self.rider_history = {}
        
        # Statistics
        self.stats = {
            'total_motorcycles': 0,
            'total_riders': 0,
            'helmeted_riders': 0,
            'non_helmeted_riders': 0,
            'compliance_rate': 0.0
        }
    
    def _detect_helmet(self, frame: np.ndarray, person_bbox: Tuple[int, int, int, int]) -> Tuple[bool, float]:
        """
        Enhanced helmet detection based on head region analysis
        
        Args:
            frame: Input frame
            person_bbox: Person bounding box (x1, y1, x2, y2)
            
        Returns:
            Tuple of (has_helmet, confidence)
        """
        x1, y1, x2, y2 = person_bbox
        
        # Extract head region (upper portion of bounding box)
        head_height = int((y2 - y1) * 0.35)  # Top 35% assumed to be head
        head_region = frame[y1:y1+head_height, x1:x2]
        
        if head_region.size == 0:
            return False, 0.0
        
        # Convert to HSV for better color analysis
        hsv = cv2.cvtColor(head_region, cv2.COLOR_BGR2HSV)
        
        # Check for dark colors typical of helmets (black, dark blue, etc.)
        dark_mask = cv2.inRange(hsv, (0, 0, 0), (180, 255, 120))
        dark_ratio = np.sum(dark_mask > 0) / dark_mask.size
        
        # Check for bright/reflective surfaces typical of helmet visors
        bright_mask = cv2.inRange(hsv, (0, 0, 180), (180, 80, 255))
        bright_ratio = np.sum(bright_mask > 0) / bright_mask.size
        
        # Check for colored helmets (red, blue, etc.)
        colored_mask = cv2.inRange(hsv, (0, 100, 50), (180, 255, 200))
        colored_ratio = np.sum(colored_mask > 0) / colored_mask.size
        
        # Enhanced heuristic combining multiple factors
        helmet_confidence = (dark_ratio * 0.6 + bright_ratio * 0.8 + colored_ratio * 0.7)
        
        # Additional shape analysis - helmets tend to be more rounded
        contours, _ = cv2.findContours(dark_mask + bright_mask + colored_mask, 
                                     cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        shape_score = 0
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest_contour) > 100:  # Minimum size threshold
                # Check roundness (perimeter^2 / area, lower is more circular)
                perimeter = cv2.arcLength(largest_contour, True)
                area = cv2.contourArea(largest_contour)
                if area > 0:
                    circularity = (perimeter * perimeter) / (4 * np.pi * area)
                    shape_score = max(0, 1.0 - (circularity - 1.0) / 2.0)  # Normalize
        
        helmet_confidence = helmet_confidence * 0.8 + shape_score * 0.2
        has_helmet = helmet_confidence > 0.25
        
        return has_helmet, helmet_confidence
    
    def _is_person_on_motorcycle(self, person_bbox: Tuple[int, int, int, int], 
                                motorcycle_bboxes: List[Tuple[int, int, int, int]]) -> Tuple[bool, int]:
        """
        Determine if a person is riding a motorcycle based on spatial relationship
        
        Args:
            person_bbox: Person bounding box (x1, y1, x2, y2)
            motorcycle_bboxes: List of motorcycle bounding boxes
            
        Returns:
            Tuple of (is_riding, motorcycle_index or -1)
        """
        px1, py1, px2, py2 = person_bbox
        person_center_x = (px1 + px2) / 2
        person_center_y = (py1 + py2) / 2
        person_bottom = py2
        
        for i, (mx1, my1, mx2, my2) in enumerate(motorcycle_bboxes):
            # Check if person is positioned above and overlapping with motorcycle
            
            # Horizontal overlap
            horizontal_overlap = max(0, min(px2, mx2) - max(px1, mx1))
            person_width = px2 - px1
            motorcycle_width = mx2 - mx1
            horizontal_ratio = horizontal_overlap / min(person_width, motorcycle_width)
            
            # Person's center should be within or slightly above motorcycle horizontally
            if (mx1 <= person_center_x <= mx2 and horizontal_ratio > 0.3):
                # Person's bottom should be within or slightly above motorcycle's top area
                motorcycle_top_zone = my1 + (my2 - my1) * 0.6  # Top 60% of motorcycle
                if py1 <= motorcycle_top_zone and person_bottom >= my1:
                    return True, i
                    
            # Also check if person is sitting on motorcycle (more relaxed spatial constraints)
            if (horizontal_ratio > 0.2 and 
                person_center_y >= my1 - (py2 - py1) * 0.3 and 
                person_center_y <= my2 + (py2 - py1) * 0.2):
                return True, i
        
        return False, -1
    
    def detect_motorcycle_helmets(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Main detection function for motorcycles and helmet compliance
        
        Args:
            frame: Input video frame
            
        Returns:
            Tuple of (annotated_frame, detection_results)
        """
        # Run YOLO inference
        results = self.model(frame, conf=self.confidence, classes=[0, 3, 4])  # person, motorcycle, bicycle
        
        annotated_frame = frame.copy()
        detection_results = {
            'motorcycles': [],
            'riders': [],
            'compliance_summary': {}
        }
        
        if results[0].boxes is not None:
            boxes = results[0].boxes.xywh.cpu()
            confidences = results[0].boxes.conf.float().cpu().tolist()
            classes = results[0].boxes.cls.int().cpu().tolist()
            
            # Get tracking IDs if available
            track_ids = None
            if results[0].boxes.id is not None:
                track_ids = results[0].boxes.id.int().cpu().tolist()
            else:
                track_ids = list(range(len(boxes)))
            
            # Separate persons and motorcycles
            persons = []
            motorcycles = []
            
            for i, (box, track_id, conf, class_id) in enumerate(zip(boxes, track_ids, confidences, classes)):
                x, y, w, h = box
                # Convert center coordinates to corner coordinates
                x1 = int(x - w/2)
                y1 = int(y - h/2)
                x2 = int(x + w/2)
                y2 = int(y + h/2)
                
                if class_id == self.person_class:
                    persons.append({
                        'bbox': (x1, y1, x2, y2),
                        'track_id': track_id,
                        'confidence': conf,
                        'center': (int(x), int(y))
                    })
                elif class_id in [self.motorcycle_class, self.bicycle_class]:
                    motorcycles.append({
                        'bbox': (x1, y1, x2, y2),
                        'track_id': track_id,
                        'confidence': conf,
                        'center': (int(x), int(y)),
                        'class_name': 'motorcycle' if class_id == self.motorcycle_class else 'bicycle'
                    })
            
            # Draw motorcycles first
            motorcycle_bboxes = []
            for motorcycle in motorcycles:
                mx1, my1, mx2, my2 = motorcycle['bbox']
                track_id = motorcycle['track_id']
                motorcycle_bboxes.append(motorcycle['bbox'])
                
                # Update motorcycle history
                if track_id not in self.motorcycle_history:
                    self.motorcycle_history[track_id] = []
                self.motorcycle_history[track_id].append(motorcycle['center'])
                
                if len(self.motorcycle_history[track_id]) > 30:
                    self.motorcycle_history[track_id] = self.motorcycle_history[track_id][-30:]
                
                # Draw motorcycle
                color = (255, 165, 0)  # Orange for motorcycles
                cv2.rectangle(annotated_frame, (mx1, my1), (mx2, my2), color, 2)
                
                # Label
                label = f"{motorcycle['class_name'].upper()} {track_id}: {motorcycle['confidence']:.2f}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                cv2.rectangle(annotated_frame, (mx1, my1-label_size[1]-10), 
                            (mx1+label_size[0], my1), color, -1)
                cv2.putText(annotated_frame, label, (mx1, my1-5), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Draw motorcycle track history
                if len(self.motorcycle_history[track_id]) > 1:
                    points = np.array(self.motorcycle_history[track_id], dtype=np.int32)
                    cv2.polylines(annotated_frame, [points], False, color, 2)
            
            # Check which persons are on motorcycles and process them
            riders_on_motorcycle = {}
            
            for person in persons:
                px1, py1, px2, py2 = person['bbox']
                track_id = person['track_id']
                
                is_riding, motorcycle_idx = self._is_person_on_motorcycle(person['bbox'], motorcycle_bboxes)
                
                if is_riding:  # Only process riders
                    motorcycle_id = motorcycles[motorcycle_idx]['track_id']
                    
                    # Detect helmet
                    has_helmet, helmet_conf = self._detect_helmet(frame, person['bbox'])
                    
                    # Store rider info
                    rider_info = {
                        'track_id': track_id,
                        'bbox': person['bbox'],
                        'confidence': person['confidence'],
                        'center': person['center'],
                        'has_helmet': has_helmet,
                        'helmet_confidence': helmet_conf,
                        'motorcycle_id': motorcycle_id
                    }
                    
                    # Group riders by motorcycle
                    if motorcycle_id not in riders_on_motorcycle:
                        riders_on_motorcycle[motorcycle_id] = []
                    riders_on_motorcycle[motorcycle_id].append(rider_info)
                    
                    # Update rider history
                    if track_id not in self.rider_history:
                        self.rider_history[track_id] = []
                    self.rider_history[track_id].append(person['center'])
                    
                    if len(self.rider_history[track_id]) > 30:
                        self.rider_history[track_id] = self.rider_history[track_id][-30:]
                    
                    # Draw rider with helmet status
                    if has_helmet:
                        color = (0, 255, 0)  # Green for helmet
                        status = "HELMET ON"
                        self.stats['helmeted_riders'] += 1
                    else:
                        color = (0, 0, 255)  # Red for no helmet
                        status = "NO HELMET"
                        self.stats['non_helmeted_riders'] += 1
                    
                    cv2.rectangle(annotated_frame, (px1, py1), (px2, py2), color, 3)
                    
                    # Draw labels
                    label = f'RIDER {track_id}: {status} ({person["confidence"]:.2f})'
                    if has_helmet:
                        label += f' [H:{helmet_conf:.2f}]'
                    
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                    cv2.rectangle(annotated_frame, (px1, py1-label_size[1]-10), 
                                (px1+label_size[0], py1), color, -1)
                    cv2.putText(annotated_frame, label, (px1, py1-5), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
                    # Draw rider track history
                    if len(self.rider_history[track_id]) > 1:
                        points = np.array(self.rider_history[track_id], dtype=np.int32)
                        cv2.polylines(annotated_frame, [points], False, color, 2)
                    
                    # Highlight head region for helmet detection
                    head_height = int((py2 - py1) * 0.35)
                    cv2.rectangle(annotated_frame, (px1, py1), (px2, py1+head_height), 
                                (255, 255, 0), 1)  # Yellow for head region
                    
                    self.stats['total_riders'] += 1
                    detection_results['riders'].append(rider_info)
            
            # Process compliance for each motorcycle
            for motorcycle_id, riders in riders_on_motorcycle.items():
                self.stats['total_motorcycles'] += 1
                
                # Check compliance for all riders on this motorcycle
                helmeted_count = sum(1 for rider in riders if rider['has_helmet'])
                total_riders = len(riders)
                
                compliance_status = "COMPLIANT" if helmeted_count == total_riders else "NON-COMPLIANT"
                compliance_color = (0, 255, 0) if compliance_status == "COMPLIANT" else (0, 0, 255)
                
                # Add compliance label to motorcycle
                motorcycle = next(m for m in motorcycles if m['track_id'] == motorcycle_id)
                mx1, my1, mx2, my2 = motorcycle['bbox']
                
                compliance_label = f"COMPLIANCE: {compliance_status} ({helmeted_count}/{total_riders} riders)"
                label_size = cv2.getTextSize(compliance_label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                cv2.rectangle(annotated_frame, (mx1, my2), 
                            (mx1+label_size[0], my2+label_size[1]+10), compliance_color, -1)
                cv2.putText(annotated_frame, compliance_label, (mx1, my2+label_size[1]+5), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                detection_results['motorcycles'].append({
                    'motorcycle_id': motorcycle_id,
                    'riders': riders,
                    'compliance_status': compliance_status,
                    'helmeted_count': helmeted_count,
                    'total_riders': total_riders
                })
        
        # Update overall compliance rate
        if self.stats['total_riders'] > 0:
            self.stats['compliance_rate'] = self.stats['helmeted_riders'] / self.stats['total_riders'] * 100
        
        detection_results['compliance_summary'] = self.stats.copy()
        
        return annotated_frame, detection_results
    
    def process_video(self, video_path: str, output_path: Optional[str] = None, 
                     display: bool = True) -> Dict:
        """
        Process video with motorcycle helmet detection
        
        Returns:
            Dictionary with detection statistics
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Processing video: {video_path}")
        print(f"Resolution: {width}x{height}, FPS: {fps}, Total frames: {total_frames}")
        
        # Setup video writer
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        start_time = time.time()
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                annotated_frame, detections = self.detect_motorcycle_helmets(frame)
                
                # Add statistics to frame
                info_text = f"Frame: {frame_count}/{total_frames} | Motorcycles: {len(detections['motorcycles'])} | Riders: {len(detections['riders'])} | Compliance: {self.stats['compliance_rate']:.1f}%"
                cv2.putText(annotated_frame, info_text, (10, 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Write frame
                if writer:
                    writer.write(annotated_frame)
                
                # Display frame
                if display:
                    # Resize display for better viewing if needed
                    display_frame = annotated_frame
                    if max(height, width) > 1200:
                        scale = 1200 / max(height, width)
                        new_w, new_h = int(width * scale), int(height * scale)
                        display_frame = cv2.resize(annotated_frame, (new_w, new_h))
                    
                    cv2.imshow('Motorcycle Helmet Detection', display_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                frame_count += 1
                
                # Print progress
                if frame_count % 30 == 0:
                    elapsed = time.time() - start_time
                    fps_current = frame_count / elapsed
                    print(f"Processed {frame_count}/{total_frames} frames "
                          f"({fps_current:.1f} FPS)")
        
        finally:
            cap.release()
            if writer:
                writer.release()
            if display:
                cv2.destroyAllWindows()
        
        # Print final statistics
        print(f"\n=== MOTORCYCLE HELMET DETECTION STATISTICS ===")
        print(f"Total frames processed: {frame_count}")
        print(f"Total motorcycles detected: {self.stats['total_motorcycles']}")
        print(f"Total riders detected: {self.stats['total_riders']}")
        print(f"Helmeted riders: {self.stats['helmeted_riders']}")
        print(f"Non-helmeted riders: {self.stats['non_helmeted_riders']}")
        print(f"Overall compliance rate: {self.stats['compliance_rate']:.1f}%")
        
        return self.stats

def main():
    # Initialize detector
    detector = MotorcycleHelmetDetector()
    
    # Process test video
    input_video = "media/test.mp4"
    output_video = "media/motorcycle_helmet_detection.mp4"
    
    print("Starting Motorcycle Helmet Detection...")
    print(f"Input: {input_video}")
    print(f"Output: {output_video}")
    
    # Process video
    stats = detector.process_video(input_video, output_video, display=False)
    
    print("Motorcycle helmet detection complete!")

if __name__ == "__main__":
    main() 