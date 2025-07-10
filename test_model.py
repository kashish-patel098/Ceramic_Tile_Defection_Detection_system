import os
import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
from pathlib import Path
import time

# Set working directory to the script location
os.chdir(os.path.dirname(os.path.abspath(__file__)))

class ModelTester:
    def __init__(self, model_path='runs/train/yolov8n-ceramic-defect/weights/best.pt'):
        """Initialize the model tester with the trained model."""
        self.model = YOLO(model_path)
        self.class_names = ['edge-chipping', 'hole', 'line']
        
    def test_single_image(self, image_path, conf_threshold=0.25, save_results=True):
        """Test the model on a single image."""
        print(f"Testing image: {image_path}")
        
        # Run prediction
        results = self.model.predict(
            source=image_path,
            conf=conf_threshold,
            save=save_results,
            project='runs/predict',
            name='single_image_test'
        )
        
        # Print results
        for result in results:
            if result.boxes is not None:
                boxes = result.boxes
                print(f"Found {len(boxes)} defects:")
                
                for i, box in enumerate(boxes):
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    class_name = self.class_names[class_id]
                    coords = box.xyxy[0].cpu().numpy()
                    
                    print(f"  {i+1}. {class_name} (confidence: {confidence:.3f})")
                    print(f"     Coordinates: x1={coords[0]:.1f}, y1={coords[1]:.1f}, x2={coords[2]:.1f}, y2={coords[3]:.1f}")
            else:
                print("No defects detected.")
        
        return results
    
    def test_batch_images(self, test_folder='test/images', conf_threshold=0.25):
        """Test the model on all images in the test folder."""
        print(f"Testing all images in: {test_folder}")
        
        # Run batch prediction
        results = self.model.predict(
            source=test_folder,
            conf=conf_threshold,
            save=True,
            project='runs/predict',
            name='batch_test'
        )
        
        print(f"Processed {len(results)} images")
        return results
    
    def evaluate_model(self, data_yaml='data.yaml'):
        """Evaluate model performance on test set."""
        print("Evaluating model performance...")
        
        # Run validation on test set
        metrics = self.model.val(data=data_yaml)
        
        print("\nModel Performance Metrics:")
        print(f"mAP50: {metrics.box.map50:.4f}")
        print(f"mAP50-95: {metrics.box.map:.4f}")
        print(f"Precision: {metrics.box.mp:.4f}")
        print(f"Recall: {metrics.box.mr:.4f}")
        
        return metrics
    
    def test_with_camera(self, camera_id=0):
        """Test the model with real-time camera feed."""
        print("Starting real-time detection with camera...")
        print("Press 'q' to quit, 's' to save current frame")
        
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            print("Error: Could not open camera")
            return
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break
            
            # Run prediction on frame
            results = self.model.predict(frame, conf=0.25, verbose=False)
            
            # Draw results on frame
            annotated_frame = results[0].plot()
            
            # Display frame
            cv2.imshow('YOLO Detection', annotated_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                timestamp = int(time.time())
                cv2.imwrite(f'captured_frame_{timestamp}.jpg', annotated_frame)
                print(f"Saved frame as captured_frame_{timestamp}.jpg")
        
        cap.release()
        cv2.destroyAllWindows()
    
    def analyze_predictions(self, results):
        """Analyze prediction results and show statistics."""
        total_detections = 0
        class_counts = {name: 0 for name in self.class_names}
        
        for result in results:
            if result.boxes is not None:
                boxes = result.boxes
                total_detections += len(boxes)
                
                for box in boxes:
                    class_id = int(box.cls[0])
                    class_name = self.class_names[class_id]
                    class_counts[class_name] += 1
        
        print("\nDetection Statistics:")
        print(f"Total detections: {total_detections}")
        print("Detections by class:")
        for class_name, count in class_counts.items():
            print(f"  {class_name}: {count}")
        
        return class_counts

def main():
    """Main function to run different testing scenarios."""
    print("YOLO Model Testing Script")
    print("=" * 40)
    
    # Initialize model tester
    tester = ModelTester()
    
    while True:
        print("\nChoose testing option:")
        print("1. Test single image")
        print("2. Test batch of images")
        print("3. Evaluate model performance")
        print("4. Real-time camera detection")
        print("5. Exit")
        
        choice = input("\nEnter your choice (1-5): ").strip()
        
        if choice == '1':
            # Test single image
            image_path = input("Enter image path (or press Enter for first test image): ").strip()
            if not image_path:
                # Use first image from test folder
                test_images = list(Path('test/images').glob('*.jpg'))
                if test_images:
                    image_path = str(test_images[0])
                    print(f"Using: {image_path}")
                else:
                    print("No test images found!")
                    continue
            
            if os.path.exists(image_path):
                results = tester.test_single_image(image_path)
                tester.analyze_predictions(results)
            else:
                print(f"Image not found: {image_path}")
        
        elif choice == '2':
            # Test batch of images
            print("Testing all images in test/images folder...")
            results = tester.test_batch_images()
            tester.analyze_predictions(results)
            print("Results saved in runs/predict/batch_test/")
        
        elif choice == '3':
            # Evaluate model performance
            tester.evaluate_model()
        
        elif choice == '4':
            # Real-time camera detection
            camera_id = input("Enter camera ID (default: 0): ").strip()
            camera_id = int(camera_id) if camera_id.isdigit() else 0
            tester.test_with_camera(camera_id)
        
        elif choice == '5':
            print("Exiting...")
            break
        
        else:
            print("Invalid choice. Please enter 1-5.")

if __name__ == "__main__":
    main() 