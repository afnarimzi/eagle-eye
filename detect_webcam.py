import cv2
from ultralytics import YOLO
import os

# Load a pre-trained YOLOv8n model
model = YOLO('yolov8n.pt')

# Webcam index 
webcam_index = 0

# Confidence threshold for detection
conf_threshold = 0.25

# --- Webcam Processing ---
print(f"Opening webcam index: {webcam_index}")

# Open the webcam using OpenCV
# Pass the index instead of a file path
cap = cv2.VideoCapture(webcam_index)

# Check if webcam opened successfully
if not cap.isOpened():
    print(f"Error: Could not open webcam index {webcam_index}")
    print("Make sure a webcam is connected and drivers are installed.")
else:
    print("Webcam opened successfully. Starting live detection...")
    print("Press 'q' on the display window to quit.")

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error reading frame from webcam.")
            break

        # Run YOLOv8 inference on the frame
        results = model.predict(source=frame, conf=conf_threshold, verbose=False)
        result = results[0] 

        # --- Draw detections on the frame ---
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            class_name = model.names[cls_id]

            cv2.rectangle(frame, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)
            label = f"{class_name} {conf:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow('YOLOv8 Webcam Detection', frame)

        # Check for 'q' key press to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("'q' pressed, stopping detection.")
            break

    # Release the webcam and close windows
    cap.release()
    cv2.destroyAllWindows()
    print("Webcam closed and resources released.")

