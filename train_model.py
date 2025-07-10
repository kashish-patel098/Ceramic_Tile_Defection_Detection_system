import os
from ultralytics import YOLO

# Set working directory to the script location
os.chdir(os.path.dirname(os.path.abspath(__file__)))

data_yaml = 'data.yaml'  # Already points to correct folders
model_arch = 'yolov8n.pt'  # You can change to yolov8s.pt, yolov8m.pt, etc.
epochs = 10
imgsz = 640

# Load model
model = YOLO(model_arch)

# Train
results = model.train(
    data=data_yaml,
    epochs=epochs,
    imgsz=imgsz,
    project='runs/train',
    name='yolov8n-ceramic-defect',
    exist_ok=True
)

# Validate
metrics = model.val()
print('Validation metrics:', metrics)

# Save best model path
best_model_path = model.ckpt_path if hasattr(model, 'ckpt_path') else None
if best_model_path:
    print(f'Best model saved at: {best_model_path}')
else:
    print('Check runs/train/yolov8n-ceramic-defect/weights for best model.') 