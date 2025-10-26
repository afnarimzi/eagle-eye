import cv2
from ultralytics import YOLO
import os

model = YOLO('yolov8n.pt')
video_path = 'vidio1.mp4' 
conf_threshold = 0.25

max_frames_to_process = 50

print(f"Opening video file: {video_path}")
if not os.path.exists(video_path):
    print(f"Error: Video file not found at {video_path}")
else:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
    else:
        print("Video opened successfully. Analyzing YOLO output...")
        print("Press 'q' on the display window to quit early.")

        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                print("End of video reached or error reading frame.")
                break

            if max_frames_to_process is not None and frame_count >= max_frames_to_process:
                print(f"Reached maximum frames to process ({max_frames_to_process}). Stopping.")
                break

            frame_count += 1
            print(f"\n--- Processing Frame {frame_count} ---")

            # Run YOLOv8 inference
            results = model.predict(source=frame, conf=conf_threshold, verbose=False)
            result = results[0] 

            # --- Analyze and Print Detections ---
            if not result.boxes:
                print("No objects detected in this frame.")
            else:
                print(f"Detected {len(result.boxes)} objects:")
                # The result.boxes object contains all detection data
                for i, box in enumerate(result.boxes):
                    # Bounding Box Coordinates:
                    # box.xyxy[0] gives tensor([x1, y1, x2, y2]) (top-left, bottom-right)
                    xyxy_coords = box.xyxy[0].cpu().numpy().astype(int) # Convert to NumPy array of integers
                    # box.xywh[0] gives tensor([cx, cy, w, h]) (center-x, center-y, width, height)
                    xywh_coords = box.xywh[0].cpu().numpy().astype(int)

                    # Confidence Score:
                    conf = float(box.conf[0].cpu().numpy()) # Convert to standard Python float

                    # Class ID and Name:
                    cls_id = int(box.cls[0].cpu().numpy()) # Convert to standard Python int
                    class_name = model.names[cls_id]

                    print(f"  Detection {i+1}:")
                    print(f"    Class: {class_name} (ID: {cls_id})")
                    print(f"    Confidence: {conf:.3f}") # Print confidence with 3 decimal places
                    print(f"    BBox (xyxy): [{xyxy_coords[0]}, {xyxy_coords[1]}, {xyxy_coords[2]}, {xyxy_coords[3]}]")
                    print(f"    BBox Center (cx, cy): ({xywh_coords[0]}, {xywh_coords[1]})")
                    print(f"    BBox Size (w, h): ({xywh_coords[2]}, {xywh_coords[3]})")

                    # --- Draw on frame for visual confirmation ---
                    cv2.rectangle(frame, (xyxy_coords[0], xyxy_coords[1]), (xyxy_coords[2], xyxy_coords[3]), (0, 255, 0), 2)
                    label = f"{class_name} {conf:.2f}"
                    cv2.putText(frame, label, (xyxy_coords[0], xyxy_coords[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Display the frame (optional, but helpful)
            cv2.imshow('YOLOv8 Output Analysis', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("'q' pressed, stopping analysis.")
                break

        cap.release()
        cv2.destroyAllWindows()
        print("\nVideo closed and resources released.")