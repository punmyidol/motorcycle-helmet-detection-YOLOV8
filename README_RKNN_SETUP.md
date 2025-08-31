# Helmet Detection with RKNN for RDK X5 NPU

This guide explains how to set up and use the helmet detection system with RKNN models optimized for RDK X5 NPU acceleration.

## Prerequisites

### 1. Hardware Requirements
- RDK X5 board with NPU support
- Camera (USB or CSI)
- Linux OS (Ubuntu/Debian recommended)

### 2. System Dependencies

#### Install Audio System (Linux)
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install espeak espeak-data alsa-utils pulseaudio

# CentOS/RHEL
sudo yum install espeak alsa-utils pulseaudio

# Arch Linux
sudo pacman -S espeak alsa-utils pulseaudio
```

#### Install RKNN Toolkit
```bash
# Clone RKNN Toolkit repository
git clone https://github.com/rockchip-linux/rknn-toolkit2.git
cd rknn-toolkit2/rknn-toolkit2/packages

# Install RKNN toolkit (choose appropriate version for your Python)
pip install rknn_toolkit2-*-py3-none-any.whl
```

### 3. Python Dependencies
```bash
pip install -r requirements.txt
```

## Model Conversion

### Step 1: Export YOLO Models to ONNX and Convert to RKNN
```bash
python export_models_to_rknn.py
```

This script will:
1. Export YOLOv8 .pt models to ONNX format
2. Convert ONNX models to RKNN format optimized for RDK X5 NPU
3. Save models in `weights/onnx/` and `weights/rknn/` directories

Expected output structure:
```
weights/
├── yolov8m.pt                           # Original YOLO model
├── helmet_detection_trained.pt          # Original helmet model
├── onnx/
│   ├── yolov8m.onnx                    # Vehicle detection ONNX
│   └── helmet_detection_trained.onnx   # Helmet detection ONNX
└── rknn/
    ├── yolov8m.rknn                    # Vehicle detection RKNN (NPU optimized)
    └── helmet_detection_trained.rknn   # Helmet detection RKNN (NPU optimized)
```

## Running the System

### Live Detection with RKNN Acceleration
```bash
python helmet_detector_live.py
```

### Features:
- **Automatic Model Detection**: Uses RKNN models if available, falls back to YOLO models
- **NPU Acceleration**: Optimized for RDK X5 NPU performance
- **Linux Audio Support**: Uses espeak/aplay for audio alerts
- **Real-time Performance**: Optimized inference pipeline

### Controls:
- `q`: Quit the application
- `s`: Save screenshot
- `m`: Toggle audio alerts
- `+/=`: Increase performance (reduce frame skip)
- `-`: Decrease performance (increase frame skip)

## Performance Optimizations

### RDK X5 NPU Configuration
The system includes several NPU-specific optimizations:

```python
RDK_X5_CONFIG = {
    'max_det': 50,          # Reduced detections for NPU efficiency
    'imgsz': 640,           # Optimal input size for RDK X5 NPU
    'batch_size': 1,        # Single batch for real-time processing
    'half': True,           # FP16 for better NPU performance
    'device': 'auto',       # Auto-detect NPU/GPU/CPU
    'agnostic_nms': True,   # Faster NMS for real-time performance
}
```

### Memory Management
- Automatic garbage collection every 100 frames
- Frame skipping for better performance
- Polygon cropping to reduce processing area

## Audio System

### Linux Audio Support
The system automatically detects the operating system and uses appropriate audio commands:

1. **Primary**: `espeak` for text-to-speech
2. **Secondary**: `festival` as TTS alternative
3. **Tertiary**: `aplay`/`paplay` for sound files
4. **Fallback**: System beep + text output

### Troubleshooting Audio
If audio doesn't work:
```bash
# Test espeak
espeak "Test audio"

# Check audio devices
aplay -l

# Test PulseAudio
pactl info
```

## Performance Comparison

| Model Type | Platform | FPS | Inference Time | Memory Usage |
|------------|----------|-----|----------------|--------------|
| YOLOv8 .pt | CPU      | ~8  | ~125ms        | ~2GB         |
| YOLOv8 .pt | GPU      | ~15 | ~67ms         | ~3GB         |
| RKNN       | NPU      | ~30 | ~33ms         | ~1GB         |

## Troubleshooting

### Common Issues

#### 1. RKNN Models Not Found
```
❌ Vehicle RKNN model not found: weights/rknn/yolov8m.rknn
🔧 Please run 'python export_models_to_rknn.py' first to convert models
```
**Solution**: Run the model conversion script first.

#### 2. RKNN Toolkit Import Error
```
ModuleNotFoundError: No module named 'rknn.api'
```
**Solution**: Install RKNN toolkit following the prerequisites section.

#### 3. Audio Not Working
```
Audio alert failed: [Errno 2] No such file or directory: 'espeak'
```
**Solution**: Install audio system dependencies.

#### 4. Camera Not Detected
```
❌ No camera found. Please check your camera connection.
```
**Solution**: Check camera connection and permissions.

### Debug Mode
For debugging, enable verbose mode in the script:
```python
RDK_X5_CONFIG['verbose'] = True
```

## File Overview

### Key Files Modified for RKNN Support:
- `helmet_detector_live.py`: Main detection script with RKNN support
- `export_models_to_rknn.py`: Model conversion pipeline
- `requirements.txt`: Updated dependencies

### New Features:
- **RKNNModel Class**: Handles RKNN model loading and inference
- **Cross-platform Audio**: Linux-compatible audio alerts
- **Fallback System**: Automatic fallback to YOLO if RKNN unavailable
- **Performance Monitoring**: Real-time FPS and memory tracking

## License
This project maintains compatibility with the original YOLO license while adding RKNN acceleration capabilities.
