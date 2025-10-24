# test_yolo.py
# Simple script to test YOLOv8 installation and basic object detection (inference).
# Uses a pre-trained model on a sample image.

from ultralytics import YOLO
import os

# --- Configuration ---
# Load a pre-trained YOLOv8n model 
model = YOLO('yolov8n.pt')

# Define the path to your sample image
image_path = 'sample_image.jpg'
# --- End Configuration ---

print(f"Running detection on: {image_path}")

# Check if the image file exists
if not os.path.exists(image_path):
    print(f"Error: Image file not found at {image_path}")
    print("Please download an image, save it as 'sample_image.jpg' in the project folder, and try again.")
else:
    # Run inference on the source
    # show=True will display the results in a window (if possible)
    # save=True will save the results to a 'runs/detect/predict' folder
    # conf=0.25 sets a confidence threshold (only show detections above 25% confidence)
    try:
        results = model.predict(source=image_path, show=True, save=True, conf=0.25, project='.', name='predict')

        print("\nDetection complete.")
        print("Results saved in the 'runs/detect/predict' folder.")

        # Optional: Print some details about the detections
        for r in results:
            if r.boxes:
                print(f"Detected {len(r.boxes)} objects.")
                for box in r.boxes:
                    class_id = int(box.cls[0])
                    class_name = model.names[class_id]
                    confidence = float(box.conf[0])
                    print(f"  - Class: {class_name}, Confidence: {confidence:.2f}")
            else:
                print("No objects detected with confidence > 0.5.")

    except Exception as e:
        print(f"\nAn error occurred during detection: {e}")
        print("This might happen if your environment doesn't support displaying images directly.")
        print("Check the 'runs/detect/predict' folder for the saved image result.")

# --- End Run Detection ---