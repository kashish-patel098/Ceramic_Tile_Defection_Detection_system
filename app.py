import streamlit as st
import cv2 
import numpy as np
from ultralytics import YOLO
import os
import tempfile
from pathlib import Path
import time
from PIL import Image


# Set page config
st.set_page_config(
    page_title="Ceramic Defect Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed"  # Changed from 'expanded' to 'collapsed' for better UX
)

# Custom CSS for better styling, including sidebar
st.markdown("""
<style>
    /* Main header styling */
    .main-header {
        font-size: 2.8rem;
        color: #0a2342;
        text-align: center;
        margin-bottom: 2.5rem;
        font-weight: 700;
        letter-spacing: 1px;
    }
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(135deg, #e3e9f7 0%, #f7fafc 100%);
        border-right: 1px solid #e0e0e0;
        box-shadow: 2px 0 8px rgba(10,35,66,0.04);
        padding-top: 2rem;
    }
    /* Add settings icon near sidebar toggle */
    [data-testid="collapsedControl"]::before {
        content: "\\2699  "; /* Unicode for ⚙️ */
        font-size: 1.3rem;
        color: #1f77b4;
        margin-right: 0.3rem;
        vertical-align: middle;
        position: relative;
        top: 1px;
        left: -2px;
        transition: color 0.2s;
        pointer-events: none;
    }
    [data-testid="collapsedControl"]:hover::before {
        color: #0a2342;
    }
    /* Sidebar title */
    .css-1d391kg, .css-1v0mbdj {
        color: #0a2342 !important;
        font-size: 1.3rem !important;
        font-weight: 600 !important;
        margin-bottom: 1.5rem !important;
    }
    /* Metric card styling */
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.2rem;
        border-radius: 0.7rem;
        margin: 0.7rem 0;
        box-shadow: 0 2px 8px rgba(10,35,66,0.06);
    }
    /* Detection box styling */
    .detection-box {
        border: 2px solid #1f77b4;
        border-radius: 0.7rem;
        padding: 1.2rem;
        margin: 1.2rem 0;
        background: #f7fafc;
    }
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        background: #e3e9f7;
        border-radius: 0.7rem 0.7rem 0 0;
        padding: 0.5rem 0.5rem 0 0.5rem;
        margin-bottom: 0.5rem;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 1.1rem;
        font-weight: 500;
        color: #0a2342;
        padding: 0.7rem 1.5rem;
        border-radius: 0.7rem 0.7rem 0 0;
        margin-right: 0.2rem;
        transition: background 0.2s;
    }
    .stTabs [aria-selected="true"] {
        background: #fff;
        color: #1f77b4;
        border-bottom: 2px solid #1f77b4;
    }
    /* Button styling */
    .stButton > button {
        background: linear-gradient(90deg, #1f77b4 0%, #0a2342 100%);
        color: #fff;
        font-weight: 600;
        border-radius: 0.5rem;
        border: none;
        padding: 0.7rem 1.5rem;
        margin: 0.5rem 0;
        transition: background 0.2s, box-shadow 0.2s;
        box-shadow: 0 2px 8px rgba(10,35,66,0.08);
    }
    .stButton > button:hover {
        background: linear-gradient(90deg, #0a2342 0%, #1f77b4 100%);
        color: #fff;
        box-shadow: 0 4px 16px rgba(10,35,66,0.12);
    }
    /* General spacing */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

class CeramicDefectDetector:
    def __init__(self):
        """Initialize the detector with the trained model."""
        self.model_path = 'runs/train/yolov8n-ceramic-defect/weights/best.pt'
        self.class_names = ['edge-chipping', 'hole', 'line']
        self.model = None
        self.load_model()
    
    def load_model(self):
        """Load the YOLO model."""
        try:
            if os.path.exists(self.model_path):
                self.model = YOLO(self.model_path)
                # st.success("✅ Model loaded successfully!")
            else:
                st.error(f"❌ Model not found at: {self.model_path}")
                st.info("Please ensure you have trained the model first using train_model.py")
        except Exception as e:
            st.error(f"❌ Error loading model: {str(e)}")
    
    def predict_image(self, image, conf_threshold=0.25):
        """Run prediction on an image."""
        if self.model is None:
            return None
        
        try:
            results = self.model.predict(
                source=image,
                conf=conf_threshold,
                verbose=False
            )
            return results[0]
        except Exception as e:
            st.error(f"❌ Prediction error: {str(e)}")
            return None
    
    def analyze_results(self, result):
        """Analyze and return detection results."""
        if result is None or result.boxes is None:
            return {
                'total_detections': 0,
                'class_counts': {name: 0 for name in self.class_names},
                'detections': []
            }
        
        detections = []
        class_counts = {name: 0 for name in self.class_names}
        
        for box in result.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            class_name = self.class_names[class_id]
            coords = box.xyxy[0].cpu().numpy()
            
            detection = {
                'class_name': class_name,
                'confidence': confidence,
                'coordinates': coords
            }
            detections.append(detection)
            class_counts[class_name] += 1
        
        return {
            'total_detections': len(detections),
            'class_counts': class_counts,
            'detections': detections
        }



def main():
    # Header
    st.markdown('<h1 class="main-header">🔍 Ceramic Tile Defect Detection System</h1>', unsafe_allow_html=True)
    
    
    # Initialize detector
    detector = CeramicDefectDetector()
    
    # Sidebar
    st.sidebar.title("⚙️ Settings")
    
    # Confidence threshold
    conf_threshold = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.1,
        max_value=1.0,
        value=0.25,
        step=0.05,
        help="Adjust the confidence level for detections"
    )

    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📷 Single Image", 
        "📁 Batch Processing", 
        "🎥 Real-time Camera", 
        "📊 Model Info"
    ])

    # Tab 1: Single Image Detection
    with tab1:
        st.header("📷 Single Image Detection")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['jpg', 'jpeg', 'png', 'bmp'],
            help="Upload an image to detect ceramic defects"
        )
        
        if uploaded_file is not None:
            # Display original image
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Original Image")
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image")
            
            # Run detection
            with st.spinner("🔍 Detecting defects..."):
                result = detector.predict_image(image, conf_threshold)
                analysis = detector.analyze_results(result)
            
            # Display results
            with col2:
                st.subheader("Detection Results")
                
                if result is not None:
                    # Plot annotated image
                    annotated_img = result.plot()
                    annotated_img_pil = Image.fromarray(annotated_img)
                    st.image(annotated_img_pil, caption="Detected Defects")
                    
                    # Display statistics
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric("Total Detections", analysis['total_detections'])
                    
                    if analysis['total_detections'] > 0:
                        st.write("**Detections by class:**")
                        for class_name, count in analysis['class_counts'].items():
                            if count > 0:
                                st.write(f"• {class_name}: {count}")
                        
                        st.write("**Detailed detections:**")
                        for i, detection in enumerate(analysis['detections'], 1):
                            coords = detection['coordinates']
                            st.write(f"{i}. **{detection['class_name']}** "
                                   f"(confidence: {detection['confidence']:.3f})")
                            st.write(f"   Coordinates: ({coords[0]:.1f}, {coords[1]:.1f}) "
                                   f"to ({coords[2]:.1f}, {coords[3]:.1f})")
                    else:
                        st.info("✅ No defects detected!")
                    st.markdown('</div>', unsafe_allow_html=True)
    
    # Tab 2: Batch Processing
    with tab2:
        st.header("📁 Batch Processing")
        
        # File uploader for multiple images
        uploaded_files = st.file_uploader(
            "Choose multiple image files",
            type=['jpg', 'jpeg', 'png', 'bmp'],
            accept_multiple_files=True,
            help="Select multiple images to process in batch"
        )
        
        if uploaded_files:
            st.write(f"📸 **Selected {len(uploaded_files)} images**")
            
            # Show selected files
            with st.expander("📋 Selected Files"):
                for i, file in enumerate(uploaded_files, 1):
                    st.write(f"{i}. {file.name}")
            
            # Process batch button
            if st.button("🚀 Process Batch Images", type="primary"):
                if len(uploaded_files) > 0:
                    with st.spinner("🔍 Processing batch images..."):
                        try:
                            # Create temporary directory for uploaded files
                            with tempfile.TemporaryDirectory() as temp_dir:
                                # Save uploaded files to temp directory
                                temp_files = []
                                for uploaded_file in uploaded_files:
                                    # Create temp file path
                                    temp_path = os.path.join(temp_dir, uploaded_file.name)
                                    # Save uploaded file
                                    with open(temp_path, 'wb') as f:
                                        f.write(uploaded_file.getbuffer())
                                    temp_files.append(temp_path)
                                
                                # Process images
                                results = []
                                total_detections = 0
                                batch_class_counts = {name: 0 for name in detector.class_names}
                                
                                # Process each image individually
                                progress_bar = st.progress(0)
                                status_text = st.empty()
                                
                                for i, temp_file in enumerate(temp_files):
                                    status_text.text(f"Processing image {i+1}/{len(temp_files)}: {os.path.basename(temp_file)}")
                                    
                                    # Run prediction on single image
                                    result = detector.predict_image(temp_file, conf_threshold)
                                    if result is not None:
                                        results.append(result)
                                        
                                        # Count detections
                                        if result.boxes is not None:
                                            total_detections += len(result.boxes)
                                            for box in result.boxes:
                                                class_id = int(box.cls[0])
                                                class_name = detector.class_names[class_id]
                                                batch_class_counts[class_name] += 1
                                    
                                    # Update progress
                                    progress_bar.progress((i + 1) / len(temp_files))
                                
                                status_text.text("✅ Processing completed!")
                                
                                # Display batch results
                                st.subheader("📊 Batch Results")
                                
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Images Processed", len(uploaded_files))
                                with col2:
                                    st.metric("Total Detections", total_detections)
                                with col3:
                                    avg_detections = total_detections / len(uploaded_files) if len(uploaded_files) > 0 else 0
                                    st.metric("Avg Detections/Image", f"{avg_detections:.2f}")
                                
                                # Show detections by class
                                if total_detections > 0:
                                    st.write("**Detections by class:**")
                                    for class_name, count in batch_class_counts.items():
                                        if count > 0:
                                            st.write(f"• {class_name}: {count}")
                                else:
                                    st.info("✅ No defects detected in any image!")
                                
                                # Show individual results
                                st.subheader("📸 Individual Results")
                                
                                # Create columns for displaying results
                                cols = st.columns(min(3, len(uploaded_files)))
                                
                                for i, (uploaded_file, result) in enumerate(zip(uploaded_files, results)):
                                    col_idx = i % 3
                                    
                                    with cols[col_idx]:
                                        st.write(f"**{uploaded_file.name}**")
                                        
                                        if result is not None and result.boxes is not None:
                                            # Create annotated image
                                            annotated_img = result.plot()
                                            annotated_img_pil = Image.fromarray(annotated_img)
                                            
                                            # Resize for display
                                            annotated_img_pil.thumbnail((300, 300))
                                            st.image(annotated_img_pil, caption=f"Detections: {len(result.boxes)}", use_column_width=True)
                                            
                                            # Show detection details
                                            for j, box in enumerate(result.boxes):
                                                class_id = int(box.cls[0])
                                                confidence = float(box.conf[0])
                                                class_name = detector.class_names[class_id]
                                                st.write(f"  {j+1}. {class_name} ({confidence:.3f})")
                                        else:
                                            st.info("No defects detected")
                                
                                st.success("✅ Batch processing completed!")
                                
                        except Exception as e:
                            st.error(f"❌ Batch processing error: {str(e)}")
                else:
                    st.warning("⚠️ Please select at least one image to process.")
        else:
            st.info("📁 Upload multiple images to process them in batch")
            st.write("**Supported formats:** JPG, JPEG, PNG, BMP")
            st.write("**Tip:** You can select multiple files by holding Ctrl (or Cmd on Mac) while clicking")
    
    # Tab 3: Real-time Camera
    with tab3:
        st.header("🎥 Real-time Camera Detection")
        
        st.info("⚠️ **Note:** Real-time camera detection requires camera access and may not work in all environments.")
        
        # Camera settings
        camera_id = st.number_input(
            "Camera ID",
            min_value=0,
            max_value=10,
            value=0,
            help="Camera device ID (usually 0 for built-in webcam)"
        )
        
        # Camera placeholder
        camera_placeholder = st.empty()
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📹 Start Camera", type="primary"):
                st.session_state.camera_active = True
                st.session_state.camera_id = camera_id
        
        with col2:
            if st.button("⏹️ Stop Camera"):
                st.session_state.camera_active = False
        
        # Camera feed
        if st.session_state.get('camera_active', False):
            try:
                cap = cv2.VideoCapture(camera_id)
                
                if not cap.isOpened():
                    st.error("❌ Could not open camera")
                    st.session_state.camera_active = False
                else:
                    st.info("📹 Camera is active. Press 'Stop Camera' to stop.")
                    
                    # Create a placeholder for the camera feed
                    camera_feed = st.empty()
                    
                    while st.session_state.get('camera_active', False):
                        ret, frame = cap.read()
                        if not ret:
                            st.error("❌ Could not read frame")
                            break
                        
                        # Run prediction
                        result = detector.predict_image(frame, conf_threshold)
                        
                        if result is not None:
                            # Draw results on frame
                            annotated_frame = result.plot()
                            
                            # Convert to PIL and display
                            annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                            pil_image = Image.fromarray(annotated_frame_rgb)
                            
                            camera_feed.image(pil_image, caption="Live Detection")
                        
                        # Small delay to prevent overwhelming the UI
                        time.sleep(0.1)
                    
                    cap.release()
                    
            except Exception as e:
                st.error(f"❌ Camera error: {str(e)}")
                st.session_state.camera_active = False
    
    # Tab 4: Model Information
    with tab4:
        st.header("📊 Model Information")
        
        # Model details
        st.subheader("Model Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Model Architecture:** YOLOv8n")
            st.write("**Classes:** 3 (edge-chipping, hole, line)")
            st.write("**Input Size:** 640x640")
        
        with col2:
            st.write("**Model Path:**", detector.model_path)
            st.write("**Confidence Threshold:**", conf_threshold)
            st.write("**Status:**", "✅ Loaded" if detector.model is not None else "❌ Not Loaded")
        
        # Training info
        st.subheader("Training Information")
        
        train_results_path = "runs/train/yolov8n-ceramic-defect"
        if os.path.exists(train_results_path):
            st.write("**Training Results:** Available")
            
            # Display training metrics if available
            metrics_file = os.path.join(train_results_path, "results.csv")
            if os.path.exists(metrics_file):
                st.write("**Training Metrics:** Available")
                st.info("Check runs/train/yolov8n-ceramic-defect/ for detailed training results")
        else:
            st.warning("⚠️ Training results not found. Run train_model.py first.")
        
        # Usage instructions
        st.subheader("📖 Usage Instructions")
        
        st.markdown("""
        **How to use this application:**
        
        1. **Single Image Detection:**
           - Upload an image using the file uploader
           - Adjust confidence threshold in sidebar
           - View detection results with bounding boxes
        
        2. **Batch Processing:**
           - Enter path to folder containing test images
           - Click "Process Batch Images"
           - Results are saved to runs/predict/streamlit_batch/
        
        3. **Real-time Camera:**
           - Select camera ID (usually 0 for built-in webcam)
           - Click "Start Camera" to begin live detection
           - Click "Stop Camera" to end
        
        4. **Model Information:**
           - View model configuration and training details
        """)

    

if __name__ == "__main__":
    main() 