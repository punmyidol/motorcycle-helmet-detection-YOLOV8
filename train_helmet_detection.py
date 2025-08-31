#!/usr/bin/env python3
"""
Helmet Detection Training Script using YOLOv8
This script trains a YOLOv8 model for helmet detection using GPU acceleration.
"""

import os
import sys
import torch
from ultralytics import YOLO
import yaml
from pathlib import Path

def check_gpu_availability():
    """Check if GPU is available and display GPU information"""
    if torch.cuda.is_available():
        print(f"✅ GPU is available!")
        print(f"   GPU Device: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA Version: {torch.version.cuda}")
        print(f"   PyTorch Version: {torch.__version__}")
        return True
    elif torch.backends.mps.is_available():
        print(f"✅ Apple Metal Performance Shaders (MPS) is available!")
        print(f"   PyTorch Version: {torch.__version__}")
        return True
    else:
        print("⚠️  No GPU acceleration available, training will use CPU")
        return False

def setup_data_yaml():
    """Setup the data.yaml file with correct paths"""
    data_yaml_path = "helmet-detection-2/data.yaml"
    
    # Read the existing data.yaml
    with open(data_yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    
    # Update paths to be absolute
    base_path = os.path.abspath("helmet-detection-2")
    data['train'] = os.path.join(base_path, "train", "images")
    data['val'] = os.path.join(base_path, "train", "images")  # Using train as val since no val folder
    
    # Save updated data.yaml
    with open(data_yaml_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)
    
    print(f"📝 Updated data.yaml with absolute paths:")
    print(f"   Train: {data['train']}")
    print(f"   Val: {data['val']}")
    print(f"   Classes: {data['nc']} ({data['names']})")
    
    return data_yaml_path

def train_helmet_model():
    """Train the helmet detection model using YOLOv8"""
    
    print("🏗️  Setting up helmet detection training...")
    print("="*60)
    
    # Check GPU availability
    gpu_available = check_gpu_availability()
    
    # Setup data configuration
    data_yaml_path = setup_data_yaml()
    
    # Initialize YOLOv8 model
    # Using YOLOv8n (nano) for faster training, can switch to YOLOv8s, YOLOv8m, etc.
    print("\n🤖 Loading YOLOv8 model...")
    model = YOLO('yolov8n.pt')  # Load a pretrained model
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"🎯 Training device: {device}")
    
    # Training parameters
    training_params = {
        'data': data_yaml_path,
        'epochs': 50,  # Adjust based on your needs
        'imgsz': 640,
        'batch': 16,  # Adjust based on GPU memory
        'device': device,
        'project': 'runs/detect',
        'name': 'helmet_detection_training',
        'save': True,
        'save_period': 10,  # Save every 10 epochs
        'patience': 10,  # Early stopping patience
        'optimizer': 'Adam',
        'lr0': 0.01,
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 3,
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
        'box': 7.5,
        'cls': 0.5,
        'dfl': 1.5,
        'pose': 12.0,
        'kobj': 1.0,
        'label_smoothing': 0.0,
        'nbs': 64,
        'overlap_mask': True,
        'mask_ratio': 4,
        'dropout': 0.0,
        'val': True,
        'plots': True,
        'verbose': True
    }
    
    print("\n🚀 Starting training with parameters:")
    for key, value in training_params.items():
        print(f"   {key}: {value}")
    
    print("\n" + "="*60)
    print("🎯 TRAINING STARTED - This may take a while...")
    print("="*60)
    
    try:
        # Train the model
        results = model.train(**training_params)
        
        print("\n" + "="*60)
        print("✅ TRAINING COMPLETED SUCCESSFULLY!")
        print("="*60)
        
        # Get the best model path
        best_model_path = model.trainer.best
        print(f"📍 Best model saved at: {best_model_path}")
        
        # Copy the best model to weights directory with desired name
        weights_dir = "weights"
        os.makedirs(weights_dir, exist_ok=True)
        target_model_path = os.path.join(weights_dir, "helmet_detection_trained.pt")
        
        import shutil
        shutil.copy2(best_model_path, target_model_path)
        print(f"📦 Model copied to: {target_model_path}")
        
        # Display training results
        print(f"\n📊 Training Results Summary:")
        print(f"   Final mAP50: {results.results_dict.get('metrics/mAP50(B)', 'N/A')}")
        print(f"   Final mAP50-95: {results.results_dict.get('metrics/mAP50-95(B)', 'N/A')}")
        print(f"   Training completed in {results.results_dict.get('train/time', 'N/A')} seconds")
        
        return target_model_path
        
    except Exception as e:
        print(f"\n❌ Training failed with error: {str(e)}")
        print("🔧 Troubleshooting tips:")
        print("   1. Check if dataset paths are correct")
        print("   2. Verify GPU memory is sufficient")
        print("   3. Try reducing batch size")
        print("   4. Check YOLO installation")
        return None

def validate_model(model_path):
    """Validate the trained model"""
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return False
    
    try:
        print(f"\n🔍 Validating trained model: {model_path}")
        model = YOLO(model_path)
        
        # Run validation
        val_results = model.val(data="helmet-detection-2/data.yaml")
        
        print("✅ Model validation completed successfully!")
        print(f"   mAP50: {val_results.box.map50}")
        print(f"   mAP50-95: {val_results.box.map}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model validation failed: {str(e)}")
        return False

def main():
    """Main training function"""
    print("🎯 HELMET DETECTION MODEL TRAINING")
    print("="*60)
    print(f"📁 Working directory: {os.getcwd()}")
    print(f"🐍 Python version: {sys.version}")
    print("="*60)
    
    # Check if dataset exists
    if not os.path.exists("helmet-detection-2"):
        print("❌ Dataset directory 'helmet-detection-2' not found!")
        print("   Please ensure the dataset is in the correct location.")
        return
    
    # Train the model
    model_path = train_helmet_model()
    
    if model_path:
        # Validate the model
        validate_model(model_path)
        
        print("\n🎉 HELMET DETECTION TRAINING PIPELINE COMPLETED!")
        print("="*60)
        print(f"✅ Trained model available at: {model_path}")
        print("🚀 Ready to run inference with helmet_detector_video.py")
        print("="*60)
    else:
        print("\n❌ Training pipeline failed!")
        print("Please check the error messages above and try again.")

if __name__ == "__main__":
    main()
