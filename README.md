# 🔍 Ceramic Defect Detection with YOLOv8

A comprehensive computer vision solution for detecting ceramic defects using YOLOv8 and Streamlit.

## 🚀 Features

- **🎯 Defect Detection**: Detect 3 types of ceramic defects (edge-chipping, hole, line)
- **📱 Web Interface**: User-friendly Streamlit application
- **📷 Multiple Input Methods**: Single image, batch processing, real-time camera
- **⚙️ Customizable**: Adjustable confidence thresholds and settings
- **📊 Analytics**: Detailed detection statistics and visualizations

## 📋 Prerequisites

- Python 3.8+
- CUDA-compatible GPU (optional, for faster training)
- Webcam (for real-time detection)

## 🛠️ Installation

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/ceramic-defect-detection.git
cd ceramic-defect-detection
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Verify Model Files
Ensure the trained model exists:
```bash
ls runs/train/yolov8n-ceramic-defect/weights/best.pt
```

## 🚀 Quick Start

### Option 1: Easy Launcher
```bash
python run_app.py
```

### Option 2: Direct Streamlit
```bash
streamlit run app.py
```

The application will open at `http://localhost:8501`

## 📖 Usage Guide

### 🎯 Single Image Detection
1. Go to **"📷 Single Image"** tab
2. Upload an image (JPG, PNG, BMP)
3. Adjust confidence threshold in sidebar
4. View detection results with bounding boxes

### 📁 Batch Processing
1. Go to **"📁 Batch Processing"** tab
2. Select multiple images from your PC
3. Click "Process Batch Images"
4. View batch statistics and individual results

### 🎥 Real-time Camera
1. Go to **"🎥 Real-time Camera"** tab
2. Select camera ID (usually 0)
3. Click "Start Camera"
4. Point camera at ceramic objects

### 📊 Model Information
1. Go to **"📊 Model Info"** tab
2. View model configuration and training details

## 🎯 Defect Classes

The model detects 3 types of ceramic defects:

| Defect Type | Description |
|-------------|-------------|
| **edge-chipping** | Chipped or damaged edges |
| **hole** | Holes, punctures, or cavities |
| **line** | Linear defects, cracks, or scratches |

## 📊 Model Performance

- **Architecture**: YOLOv8n (nano)
- **Input Size**: 640x640 pixels
- **Classes**: 3 defect types
- **Training**: Custom dataset with ceramic images

## 📁 Project Structure

```
ceramic-defect-detection/
├── 📄 app.py                    # Streamlit application
├── 📄 train_model.py            # Training script
├── 📄 test_model.py             # Testing script
├── 📄 run_app.py                # Launcher script
├── 📄 requirements.txt           # Dependencies
├── 📄 data.yaml                 # Dataset configuration
├── 📄 yolov8n.pt               # Pre-trained YOLO model
├── 📁 runs/
│   └── train/yolov8n-ceramic-defect/weights/
│       └── 📄 best.pt          # Trained model
├── 📁 train/                    # Training dataset
│   ├── images/                  # Training images
│   └── labels/                  # Training labels
├── 📁 valid/                    # Validation dataset
│   ├── images/                  # Validation images
│   └── labels/                  # Validation labels
└── 📁 test/                     # Test dataset
    ├── images/                  # Test images
    └── labels/                  # Test labels
```

## 🔧 Training Your Own Model

### 1. Prepare Dataset
Organize your images and labels:
```
train/
├── images/          # Training images
└── labels/          # YOLO format labels

valid/
├── images/          # Validation images
└── labels/          # YOLO format labels
```

### 2. Update Configuration
Edit `data.yaml`:
```yaml
path: .  # Dataset root directory
train: train/images  # Train images
val: valid/images    # Validation images

# Classes
names:
  0: edge-chipping
  1: hole
  2: line
```

### 3. Train Model
```bash
python train_model.py
```

## 🎨 Customization

### Adjusting Confidence Threshold
- **Higher values** (0.7-1.0): Fewer but more confident detections
- **Lower values** (0.1-0.3): More detections but may include false positives
- **Recommended**: 0.25 for balanced performance

### Adding New Defect Types
1. Update `data.yaml` with new class names
2. Retrain model with updated dataset
3. Update class names in `app.py`

## 🔧 Troubleshooting

### Model Not Found
```
❌ Model not found at: runs/train/yolov8n-ceramic-defect/weights/best.pt
```
**Solution**: Run `python train_model.py` to train the model

### Camera Issues
```
❌ Could not open camera
```
**Solutions**:
- Check camera connection
- Try different camera ID (0, 1, 2, etc.)
- Ensure camera permissions

### Dependencies Issues
```
ModuleNotFoundError: No module named 'ultralytics'
```
**Solution**: `pip install -r requirements.txt`

## 📈 Performance Tips

- **GPU Acceleration**: Use CUDA-compatible GPU for faster processing
- **Image Size**: Larger images provide better detail but slower processing
- **Batch Size**: Process multiple images for efficiency
- **Confidence Threshold**: Adjust based on your specific needs

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Ultralytics**: YOLOv8 implementation
- **Streamlit**: Web application framework
- **OpenCV**: Computer vision library
- **Roboflow**: Dataset management platform

## 📞 Support

For questions or issues:
1. Check the [troubleshooting section](#troubleshooting)
2. Review the [documentation](README_streamlit.md)
3. Open an [issue](https://github.com/yourusername/ceramic-defect-detection/issues)

---

**Happy Defect Detection! 🔍✨**

Made with ❤️ for quality control automation 