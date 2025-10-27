import numpy as np
import cv2
from ultralytics import YOLO
from spatialmath import SE3 
import os
import time


MODEL_PATH = 'weights/best.pt' 
VIDEO_PATH = 'vidio1.mp4' 

COLLISION_THRESHOLD_M = 100.0 # Safety margin in meters
CONF_THRESHOLD = 0.10         

# Simplified Camera Parameters 
FOCAL_LENGTH_PX = 700.0
BASELINE_M = 0.5   
CX = 640.0        
CY = 360.0        
FPS = 30.0         
TIME_STEP = 1.0 / FPS
FRAME_LIMIT = 100 

#  CORE SPATIAL MATH FUNCTIONS ---

def triangulate_pos(u_l, v_l, u_r, v_r, f, B, cx, cy):
    """Calculates 3D position (X, Y, Z) relative to the left camera."""
    disparity = u_l - u_r
    if disparity <= 0.5:
        return np.array([np.nan, np.nan, np.nan])
    
    Z = (f * B) / disparity
    X = ((u_l - cx) * Z) / f
    Y = ((v_l - cy) * Z) / f
    return np.array([X, Y, Z])

def predict_collision(aircraft_pos, aircraft_vel, bird_pos, bird_vel, max_time, step, threshold):
    """Projects paths forward to check for collision."""
    time_to_impact = np.inf
    min_distance = np.inf

    for t in np.arange(0, max_time, step):
        future_aircraft_pos = aircraft_pos + aircraft_vel * t
        future_bird_pos = bird_pos + bird_vel * t
        distance = np.linalg.norm(future_aircraft_pos - future_bird_pos)
        
        if distance < min_distance:
            min_distance = distance
            
        if distance < threshold:
            time_to_impact = t
            return True, time_to_impact, min_distance

    return False, time_to_impact, min_distance

#  MAIN INTEGRATION PIPELINE ---

def run_detection_pipeline(video_path, model_path, frame_limit):
    
    print(f"Loading custom model from: {model_path}...")
    model = YOLO(model_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Cannot open video file {video_path} using OpenCV. Check path/codec.")
        return

    positions_history = []
    
    # --- SIMULATED AIRCRAFT STATE (World Frame) ---
    AIRCRAFT_POS_W = np.array([0.0, 0.0, 0.0])
    AIRCRAFT_VEL_W = np.array([70.0, 0.0, 0.0])
    T_WC = SE3() # Camera Frame = World Frame for simplicity

    frame_count = 0
    print("\nStarting frame processing...")

    while True:
        ret, frame = cap.read()
        if not ret or frame_count >= frame_limit:
            break
        
        frame_count += 1
        
        #  Run YOLO Detection
        results = model.predict(source=frame, conf=CONF_THRESHOLD, verbose=False)
        
        # Check if ANY object was detected in the frame
        if results and results[0].boxes:
            box = results[0].boxes[0]
            class_found = model.names[int(box.cls[0].item())] 
            
            # --- Get 2D Pixel Coordinates (Left Camera) ---
            u_left, v_left = box.xywh[0][0].item(), box.xywh[0][1].item()
            
            # --- Simulate Right Camera Detection (Creates Disparity) ---
            # Assume a fixed bird 100m away (Disparity is calculated from Z=100m)
            simulated_disparity = (FOCAL_LENGTH_PX * BASELINE_M) / 100.0 
            u_right = u_left - simulated_disparity
            v_right = v_left 
            
            #  Triangulation (3D Position in Camera Frame) ---
            P_C = triangulate_pos(u_left, v_left, u_right, v_right, FOCAL_LENGTH_PX, BASELINE_M, CX, CY)
            
            if np.any(np.isnan(P_C)): continue

            #  Coordinate Transform (3D Position in World Frame) ---
            P_W = T_WC * P_C # P_W is the bird's world position
            
            positions_history.append((P_W, frame_count * TIME_STEP))

            # Tracking, Velocity, and Prediction (Run every 5 frames) ---
            if len(positions_history) >= 5 and frame_count % 5 == 0:
                
                # Use the last two points in history for instantaneous velocity
                pos_t1, time_t1 = positions_history[-2]
                pos_t2, time_t2 = positions_history[-1]
                
                delta_time = time_t2 - time_t1
                
                if delta_time > 0:
                    # BIRD_VEL_TRACKED = (pos_t2 - pos_t1) / delta_time # Actual tracked velocity
                    
                    # --- FORCED COLLISION SIMULATION (Override Tracking) ---
                    # Aggressive velocity towards the plane (-50.0 m/s along X)
                    BIRD_VEL_W_COLLIDE = np.array([-50.0, 1.0, -1.0]) 
                    
                    # Collision Prediction using the FORCED velocity
                    is_collision, t_impact, min_dist = predict_collision(
                        AIRCRAFT_POS_W, AIRCRAFT_VEL_W, pos_t2, BIRD_VEL_W_COLLIDE, 
                        max_time=10.0, step=TIME_STEP, threshold=COLLISION_THRESHOLD_M
                    )
                    
                    if is_collision:
                        print(f"\n*** FRAME {frame_count}: COLLISION ALERT (TTI) ***")
                        print(f"  DETECTED CLASS: {class_found}")
                        print(f"  Time to Impact: {t_impact:.2f} seconds")
                        print(f"  Min Predicted Distance: {min_dist:.2f} m")
                    
        else:
            positions_history = []
            
    cap.release()
    print("\nPipeline finished processing video frames.")

