#!/usr/bin/env python3
"""
Export YOLOv8 models to ONNX and then convert to RKNN format for RDK X5 NPU
This script handles the complete conversion pipeline: PT -> ONNX -> RKNN
"""

import os
import sys
from ultralytics import YOLO
import onnx
from rknn.api import RKNN

def export_yolo_to_onnx(model_path, output_path, img_size=640):
    """Export YOLOv8 .pt model to ONNX format"""
    print(f"🔄 Exporting {model_path} to ONNX...")
    
    try:
        # Load the YOLOv8 model
        model = YOLO(model_path)
        
        # Export to ONNX with RDK X5 optimized settings
        success = model.export(
            format='onnx',
            imgsz=img_size,
            optimize=True,
            half=True,  # Use FP16 for better NPU performance
            simplify=True,  # Simplify the model
            opset=11,  # ONNX opset version compatible with RKNN
            dynamic=False,  # Static input shapes for NPU optimization
            batch=1  # Single batch for real-time inference
        )
        
        if success:
            print(f"✅ Successfully exported to ONNX: {output_path}")
            return True
        else:
            print(f"❌ Failed to export {model_path} to ONNX")
            return False
            
    except Exception as e:
        print(f"❌ Error exporting {model_path} to ONNX: {e}")
        return False

def convert_onnx_to_rknn(onnx_path, rknn_path, target_platform='rk3588'):
    """Convert ONNX model to RKNN format for RDK X5 NPU"""
    print(f"🔄 Converting {onnx_path} to RKNN...")
    
    try:
        # Initialize RKNN object
        rknn = RKNN(verbose=True)
        
        # Configure RKNN for RDK X5 (RK3588 NPU)
        print(f"📋 Configuring RKNN for {target_platform}...")
        ret = rknn.config(
            mean_values=[[0, 0, 0]],  # YOLOv8 preprocessing
            std_values=[[255, 255, 255]],  # YOLOv8 normalization
            target_platform=target_platform,
            optimization_level=3,  # Maximum optimization
            compress_weight=True,  # Compress weights for NPU efficiency
            quantized_dtype='asymmetric_quantized-u8',  # 8-bit quantization
            quantized_algorithm='normal',
            float_dtype='float16'  # Use FP16 for better performance
        )
        
        if ret != 0:
            print(f"❌ RKNN config failed with code: {ret}")
            return False
        
        # Load ONNX model
        print("📥 Loading ONNX model...")
        ret = rknn.load_onnx(model=onnx_path)
        if ret != 0:
            print(f"❌ Load ONNX failed with code: {ret}")
            return False
        
        # Build RKNN model
        print("🔨 Building RKNN model...")
        ret = rknn.build(do_quantization=True)
        if ret != 0:
            print(f"❌ Build RKNN failed with code: {ret}")
            return False
        
        # Export RKNN model
        print(f"💾 Exporting RKNN model to {rknn_path}...")
        ret = rknn.export_rknn(rknn_path)
        if ret != 0:
            print(f"❌ Export RKNN failed with code: {ret}")
            return False
        
        # Cleanup
        rknn.release()
        
        print(f"✅ Successfully converted to RKNN: {rknn_path}")
        return True
        
    except Exception as e:
        print(f"❌ Error converting {onnx_path} to RKNN: {e}")
        return False

def main():
    """Main function to export all models"""
    print("🚀 Starting YOLOv8 to RKNN conversion pipeline...")
    
    # Check if weights directory exists
    weights_dir = "weights"
    if not os.path.exists(weights_dir):
        print(f"❌ Weights directory '{weights_dir}' not found!")
        sys.exit(1)
    
    # Create output directories
    onnx_dir = "weights/onnx"
    rknn_dir = "weights/rknn"
    os.makedirs(onnx_dir, exist_ok=True)
    os.makedirs(rknn_dir, exist_ok=True)
    
    # Models to convert
    models_to_convert = [
        {
            'pt_path': 'weights/yolov8m.pt',
            'onnx_path': 'weights/onnx/yolov8m.onnx',
            'rknn_path': 'weights/rknn/yolov8m.rknn',
            'name': 'Vehicle Detection Model (YOLOv8m)'
        },
        {
            'pt_path': 'weights/helmet_detection_trained.pt',
            'onnx_path': 'weights/onnx/helmet_detection_trained.onnx',
            'rknn_path': 'weights/rknn/helmet_detection_trained.rknn',
            'name': 'Helmet Detection Model (Custom)'
        }
    ]
    
    success_count = 0
    total_models = len(models_to_convert)
    
    for model_info in models_to_convert:
        print(f"\n{'='*60}")
        print(f"🔄 Processing: {model_info['name']}")
        print(f"{'='*60}")
        
        # Check if PT model exists
        if not os.path.exists(model_info['pt_path']):
            print(f"❌ Model file not found: {model_info['pt_path']}")
            continue
        
        # Step 1: Export PT to ONNX
        if export_yolo_to_onnx(model_info['pt_path'], model_info['onnx_path']):
            # Step 2: Convert ONNX to RKNN
            if convert_onnx_to_rknn(model_info['onnx_path'], model_info['rknn_path']):
                success_count += 1
                print(f"✅ Complete pipeline success for {model_info['name']}")
            else:
                print(f"❌ RKNN conversion failed for {model_info['name']}")
        else:
            print(f"❌ ONNX export failed for {model_info['name']}")
    
    print(f"\n{'='*60}")
    print(f"📊 CONVERSION SUMMARY")
    print(f"{'='*60}")
    print(f"✅ Successfully converted: {success_count}/{total_models} models")
    
    if success_count == total_models:
        print("🎉 All models successfully converted to RKNN format!")
        print("📁 ONNX models saved in: weights/onnx/")
        print("📁 RKNN models saved in: weights/rknn/")
        print("\n🔥 Your models are now ready for RDK X5 NPU acceleration!")
    else:
        print(f"⚠️  {total_models - success_count} model(s) failed to convert")
        print("Please check the error messages above and ensure:")
        print("  - RKNN toolkit is properly installed")
        print("  - All model files exist in the weights directory")
        print("  - You have sufficient disk space")

if __name__ == "__main__":
    main()
