import numpy as np
import time 

# --- Simplified Stereo Camera Parameters ---
focal_length_px = 700.0
baseline_m = 0.5
cx = 960.0
cy = 540.0

# --- Triangulation Function ---
def triangulate_point(u_left, v_left, u_right, v_right, f, B, cx, cy):
    disparity = u_left - u_right
    if disparity <= 0:
        return np.nan, np.nan, np.nan
    else:
        Z = (f * B) / disparity
        X = ((u_left - cx) * Z) / f
        Y = ((v_left - cy) * Z) / f
        return np.array([X, Y, Z]) 

# --- Simulate Detections at Time 1 ---
print("--- Time 1 ---")
# BBox center from left camera at Time 1
u_left_t1 = 1100.0
v_left_t1 = 500.0
# BBox center from right camera at Time 1
u_right_t1 = 1050.0
v_right_t1 = 500.0

print(f"Left BBox Center (t1): ({u_left_t1}, {v_left_t1})")
print(f"Right BBox Center (t1): ({u_right_t1}, {v_right_t1})")

# Calculate 3D Position at Time 1
pos_t1 = triangulate_point(u_left_t1, v_left_t1, u_right_t1, v_right_t1,
                           focal_length_px, baseline_m, cx, cy)

if np.any(np.isnan(pos_t1)): # Check if any element is NaN
    print("Could not calculate initial 3D position.")
else:
    print(f"Position (t1) [X,Y,Z] (m): [{pos_t1[0]:.2f}, {pos_t1[1]:.2f}, {pos_t1[2]:.2f}]")

    # --- Simulate Detections at Time 2 (e.g., 0.1 seconds later) ---
    print("\n--- Time 2 (0.1s later) ---")
    time_diff_s = 0.1 # Time difference in seconds

    # Assume bird moved slightly forward (smaller disparity) and slightly right/up
    u_left_t2 = 1105.0 # Moved right in left image
    v_left_t2 = 498.0 # Moved up in left image
    u_right_t2 = 1056.0 # Also moved right, but disparity changed
    v_right_t2 = 498.0 # Moved up in right image

    print(f"Left BBox Center (t2): ({u_left_t2}, {v_left_t2})")
    print(f"Right BBox Center (t2): ({u_right_t2}, {v_right_t2})")

    # Calculate 3D Position at Time 2
    pos_t2 = triangulate_point(u_left_t2, v_left_t2, u_right_t2, v_right_t2,
                               focal_length_px, baseline_m, cx, cy)

    if np.any(np.isnan(pos_t2)):
        print("Could not calculate second 3D position.")
    else:
        print(f"Position (t2) [X,Y,Z] (m): [{pos_t2[0]:.2f}, {pos_t2[1]:.2f}, {pos_t2[2]:.2f}]")

        # --- Calculate Velocity ---
        # Velocity = (Change in Position) / (Change in Time)
        velocity_mps = (pos_t2 - pos_t1) / time_diff_s # Meters per second

        print("\n--- Calculated Velocity (Relative to Left Camera) ---")
        print(f"Velocity [Vx, Vy, Vz] (m/s): [{velocity_mps[0]:.2f}, {velocity_mps[1]:.2f}, {velocity_mps[2]:.2f}]")

        # Calculate speed (magnitude of the velocity vector)
        speed_mps = np.linalg.norm(velocity_mps)
        print(f"Speed (m/s): {speed_mps:.2f}")