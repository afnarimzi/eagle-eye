import cv2
from ultralytics import YOLO
import os


# Load a pre-trained YOLOv8n model
model = YOLO('yolov8n.pt')

video_path = 'vidio1.mp4' 

conf_threshold = 0.25


# --- Video Processing ---
print(f"Opening video file: {video_path}")


if not os.path.exists(video_path):
    print(f"Error: Video file not found at {video_path}")
else:
    # Open the video file using OpenCV
    cap = cv2.VideoCapture(video_path)

    # Check if video opened successfully
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
    else:
        print("Video opened successfully. Starting detection...")
        print("Press 'q' on the display window to quit.")

        # Loop through the video frames
        while True:
            ret, frame = cap.read()

            if not ret:
                print("End of video reached or error reading frame.")
                break

            # Run YOLOv8 inference on the frame
            results = model.predict(source=frame, conf=conf_threshold, verbose=False) 

            result = results[0]

            # Iterate through detected boxes
            for box in result.boxes:
                # Get coordinates (xyxy format: top-left x, top-left y, bottom-right x, bottom-right y)
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                # Get confidence score
                conf = float(box.conf[0])
                # Get class ID
                cls_id = int(box.cls[0])
                # Get class name from model's names dictionary
                class_name = model.names[cls_id]

                # Draw the bounding box rectangle on the frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2) # Green box

                # Create the label text (Class Name + Confidence)
                label = f"{class_name} {conf:.2f}"

                # Put the label text above the bounding box
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            

            # Display the resulting frame in a window
            cv2.imshow('YOLOv8 Live Detection', frame)

            # Wait for 1 millisecond and check if the 'q' key is pressed
            # If 'q' is pressed, break the loop
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("'q' pressed, stopping detection.")
                break

        # Release the video capture object and close display windows
        cap.release()
        cv2.destroyAllWindows()
        print("Video closed and resources released.")

# --- End Video Processing ---