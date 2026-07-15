import os
import torch
from ultralytics import YOLO

def inspect_yolo_model(model_path):
    print("=" * 60)
    print(f"INSPECTING MODEL: {model_path}")
    print("=" * 60)

    if not os.path.exists(model_path):
        print(f"[ERROR] File does not exist: {model_path}\n")
        return

    # File size
    file_size_mb = os.path.getsize(model_path) / (1024 * 1024)
    print(f"File Size          : {file_size_mb:.2f} MB")

    try:
        # Load model using Ultralytics
        model = YOLO(model_path)
        
        # Basic task and classes
        print(f"Task               : {model.task}")
        print(f"Number of Classes  : {len(model.names)}")
        print(f"Class Names        : {model.names}")
        
        # PyTorch Model parameters
        pytorch_model = model.model
        total_params = sum(p.numel() for p in pytorch_model.parameters())
        trainable_params = sum(p.numel() for p in pytorch_model.parameters() if p.requires_grad)
        
        print(f"Total Parameters   : {total_params:,} ({total_params/1e6:.2f} Million)")
        print(f"Trainable Params   : {trainable_params:,}")
        
        # Determine YOLO version based on layers
        layer_types = set(m.__class__.__name__ for m in pytorch_model.modules())
        yolo_version = "Unknown YOLO Version"
        
        # YOLO11 uses C3k2 and C2fAttn
        if "C3k2" in layer_types:
            yolo_version = "YOLO11"
        # YOLOv10 uses PSA / C2f
        elif "PSA" in layer_types and "C2f" in layer_types:
            yolo_version = "YOLOv10"
        # YOLOv9 uses RepNCSPELAN4 or SPPELAN
        elif "RepNCSPELAN4" in layer_types or "SPPELAN" in layer_types:
            yolo_version = "YOLOv9"
        # YOLOv8 uses C2f but not C3k2, PSA, or RepNCSPELAN4
        elif "C2f" in layer_types:
            yolo_version = "YOLOv8"
        # YOLOv5 uses C3
        elif "C3" in layer_types:
            yolo_version = "YOLOv5"

        # Guess the scale (n, s, m, l, x) based on parameter counts
        # (YOLOv8 scales: n ~3.2M, s ~11.2M, m ~25.9M, l ~43.7M, x ~68.2M)
        # (YOLO11 scales: n ~2.6M, s ~9.4M, m ~20.1M, l ~25.3M, x ~56.9M)
        scale_guess = "Unknown"
        if yolo_version == "YOLO11":
            if total_params < 4e6: scale_guess = "Nano (n)"
            elif total_params < 12e6: scale_guess = "Small (s)"
            elif total_params < 22e6: scale_guess = "Medium (m)"
            elif total_params < 35e6: scale_guess = "Large (l)"
            else: scale_guess = "X-Large (x)"
        elif yolo_version == "YOLOv8":
            if total_params < 5e6: scale_guess = "Nano (n)"
            elif total_params < 15e6: scale_guess = "Small (s)"
            elif total_params < 35e6: scale_guess = "Medium (m)"
            elif total_params < 55e6: scale_guess = "Large (l)"
            else: scale_guess = "X-Large (x)"
        
        print(f"Detected YOLO Version: {yolo_version} ({scale_guess})")
        print(f"Unique Layer Types : {sorted(list(layer_types))}")
        
        # Check if there is training metadata (embedded in the ckpt)
        if hasattr(model, "ckpt") and model.ckpt is not None:
            ckpt = model.ckpt
            print("\n--- Checkpoint Metadata ---")
            if isinstance(ckpt, dict):
                print(f"Ultralytics Version: {ckpt.get('version', 'Unknown')}")
                print(f"Train Epochs       : {ckpt.get('epoch', 'Unknown')}")
                
                # Check for training arguments / overrides
                train_args = ckpt.get('train_args', {})
                if train_args:
                    print("Training Configurations (Hyperparameters):")
                    keys_to_show = ['imgsz', 'batch', 'epochs', 'lr0', 'lrf', 'momentum', 'weight_decay', 'optimizer', 'close_mosaic', 'rect']
                    for k in keys_to_show:
                        if k in train_args:
                            print(f"  - {k:<15}: {train_args[k]}")
        
        # Check overrides attribute
        if hasattr(model, "overrides") and model.overrides:
            print("\n--- Model Overrides / Export Config ---")
            for k, v in model.overrides.items():
                if v is not None and k in ['imgsz', 'task', 'mode', 'batch', 'epochs', 'data']:
                    print(f"{k:<18}: {v}")
                    
        # Device information
        device_name = "CUDA (GPU)" if torch.cuda.is_available() else "CPU"
        print(f"\nCurrent Inference Device: {device_name}")

    except Exception as e:
        print(f"[ERROR] Failed to read detailed model metadata: {e}")
        
    print("=" * 60 + "\n")

if __name__ == "__main__":
    # Inspect the local models in the workspace
    models_to_check = [
       "models/best_fire_detection_v1.pt"
    ]
    
    for path in models_to_check:
        inspect_yolo_model(path)