#!/usr/bin/env python3
"""
Ceramic Defect Detection - Streamlit App Runner
This script checks prerequisites and launches the Streamlit application.
"""

import os
import sys
import subprocess
import importlib.util

def check_dependencies():
    """Check if all required packages are installed."""
    required_packages = [
        'streamlit',
        'ultralytics', 
        'cv2',
        'PIL',
        'numpy'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        if package == 'cv2':
            try:
                import cv2
            except ImportError:
                missing_packages.append('opencv-python')
        elif package == 'PIL':
            try:
                import PIL
            except ImportError:
                missing_packages.append('pillow')
        else:
            if importlib.util.find_spec(package) is None:
                missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\n💡 Install missing packages with:")
        print("   pip install -r requirements.txt")
        return False
    
    print("✅ All dependencies are installed!")
    return True

def check_model():
    """Check if the trained model exists."""
    model_path = 'runs/train/yolov8n-ceramic-defect/weights/best.pt'
    
    if os.path.exists(model_path):
        print(f"✅ Model found at: {model_path}")
        return True
    else:
        print(f"❌ Model not found at: {model_path}")
        print("\n💡 Please train the model first:")
        print("   python train_model.py")
        return False

def check_data_structure():
    """Check if the required data structure exists."""
    required_dirs = ['test/images', 'train/images', 'valid/images']
    missing_dirs = []
    
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            missing_dirs.append(dir_path)
    
    if missing_dirs:
        print("⚠️  Missing directories:")
        for dir_path in missing_dirs:
            print(f"   - {dir_path}")
        print("\n💡 Ensure your dataset is properly organized.")
        return False
    
    print("✅ Data structure looks good!")
    return True

def run_streamlit():
    """Run the Streamlit application."""
    print("\n🚀 Starting Streamlit application...")
    print("📱 The app will open in your browser at: http://localhost:8501")
    print("⏹️  Press Ctrl+C to stop the application")
    print("-" * 50)
    
    try:
        subprocess.run(['streamlit', 'run', 'app.py'], check=True)
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error running Streamlit: {e}")
    except FileNotFoundError:
        print("\n❌ Streamlit not found. Install with: pip install streamlit")

def main():
    """Main function to check prerequisites and run the app."""
    print("🔍 Ceramic Defect Detection - Streamlit App")
    print("=" * 50)
    
    # Check prerequisites
    print("\n📋 Checking prerequisites...")
    
    deps_ok = check_dependencies()
    model_ok = check_model()
    data_ok = check_data_structure()
    
    if not deps_ok:
        print("\n❌ Please install missing dependencies first.")
        return
    
    if not model_ok:
        print("\n❌ Please train the model first.")
        return
    
    if not data_ok:
        print("\n⚠️  Some data directories are missing, but continuing...")
    
    # Ask user if they want to continue
    print("\n" + "=" * 50)
    response = input("🚀 Ready to launch the app? (y/n): ").lower().strip()
    
    if response in ['y', 'yes', '']:
        run_streamlit()
    else:
        print("👋 Application not launched.")

if __name__ == "__main__":
    main() 