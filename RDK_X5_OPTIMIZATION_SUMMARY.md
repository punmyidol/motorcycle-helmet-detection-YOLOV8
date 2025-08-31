# RDK X5 NPU Optimization Summary

## Overview
Successfully optimized `helmet_detector_live.py` with the same logic as `helmet_detector_video.py` while adding RDK X5 NPU-specific performance enhancements.

## Key Changes Made

### ✅ Logic Synchronization
- **Color Detection**: Added the same `detect_color()` function from video script
- **Motorcycle Color Tracking**: Implemented motorcycle color detection and statistics
- **Debug Output**: Added frame-by-frame detection logging like video script
- **Statistics Reporting**: Enhanced session summary with color statistics

### ⚡ RDK X5 NPU Optimizations

#### 1. Performance Configuration
```python
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
```

#### 2. Frame Processing Optimizations
- **Frame Skipping**: Process every 2nd frame by default (configurable)
- **Memory Management**: Automatic garbage collection every 100 frames
- **Buffer Optimization**: Minimal camera buffer size for real-time processing
- **Dynamic Performance Control**: Runtime adjustment of frame skip interval

#### 3. NPU-Specific Features
- **FP16 Acceleration**: Automatic CUDA/NPU detection with FP16 support
- **Optimized Image Size**: 640x640 resolution optimized for RDK X5 NPU
- **Reduced Max Detections**: Limited to 50 detections per frame for efficiency
- **Agnostic NMS**: Faster Non-Maximum Suppression algorithm

### 🎮 Enhanced Controls
- **q**: Quit application
- **s**: Save screenshot with timestamp
- **m**: Toggle audio alerts
- **+**: Increase performance (reduce frame skip)
- **-**: Decrease performance (increase frame skip)

### 📊 Enhanced Statistics Display
- Real-time FPS monitoring
- Memory usage tracking
- NPU status indicator (ACTIVE/SKIPPED)
- Frame skip interval display
- Live detection counts
- Motorcycle color tracking

### 🔧 System Monitoring
- Available memory display at startup
- Real-time memory usage percentage
- Performance configuration logging
- Frame processing progress reporting

## Performance Benefits for RDK X5 NPU

1. **Optimized Throughput**: Frame skipping reduces NPU load while maintaining detection accuracy
2. **Memory Efficiency**: Automatic garbage collection and reduced buffer usage
3. **Real-time Processing**: Minimal latency with optimized inference settings
4. **Dynamic Scaling**: Runtime performance adjustment based on system load
5. **Hardware Acceleration**: Automatic FP16 mode detection for NPU/GPU acceleration

## Usage Instructions

1. **Run the optimized live detector**:
   ```bash
   python helmet_detector_live.py
   ```

2. **Monitor Performance**:
   - Watch FPS counter and memory usage in overlay
   - NPU status shows ACTIVE/SKIPPED based on frame processing
   - Use +/- keys to adjust performance dynamically

3. **Optimal Settings for RDK X5**:
   - Default frame skip interval: 2 (process every 2nd frame)
   - Image size: 640x640 (NPU optimized)
   - FP16 acceleration enabled
   - Reduced max detections for efficiency

## Hardware Requirements
- RDK X5 NPU development board
- Camera (USB or integrated)
- Sufficient memory (monitored in real-time)
- YOLO model weights in `weights/` directory

The optimized script now provides the same detection logic as the video processor while being specifically tuned for RDK X5 NPU performance and real-time processing capabilities.
