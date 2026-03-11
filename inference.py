from ultralytics import YOLO
import sys
import os
import torch
import ultralytics

# PyTorch 2.6+ security update workaround for loading YOLO weights
torch.serialization.add_safe_globals([ultralytics.nn.tasks.DetectionModel])
import os

def main():
    if len(sys.argv) < 2:
        print("Usage: python inference.py <path_to_image_or_video> OR python inference.py --webcam")
        sys.exit(1)

    if sys.argv[1] == '--webcam':
        source = '0'  # 0 is usually the default webcam index in OpenCV
        print("Using webcam for inference... (Press 'q' in the window to quit)")
    else:
        source = sys.argv[1]
    #
    # Load the best model weights from training.
    # YOLO typically saves runs to runs/detect/train/weights/best.pt
    model_path = 'best (2).pt'
    
    if not os.path.exists(model_path):
        print(f"Error: Could not find trained model at {model_path}")
        print("Please ensure you have trained the model first using train.py")
        sys.exit(1)
#.\venv\Scripts\python inference.py videoplayback.mp4
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    # Run inference
    print(f"Running inference on {source}...")
    # 'show' attempts to display results on screen, 'save' writes to runs/detect/predict directory
    results = model(source, show=True, save=True)
    print("Inference completed. Results saved in the runs/detect/predict directory.")

if __name__ == '__main__':
    main()
