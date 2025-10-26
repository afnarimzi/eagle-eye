import numpy as np

# --- Simplified Stereo Camera Parameters ---
focal_length_px = 700.0
baseline_m = 0.5 # 50 cm apart
cx = 960.0 # Image center X (of 1920 width)
cy = 540.0 # Image center Y (of 1080 height)

# --- Triangulation Function ---
def triangulate_point(u_left, v_left, u_right, v_right, f, B, cx, cy):
    """Calculates 3D position (X, Y, Z) relative to the left camera
    using simplified parallel stereo geometry."""
    disparity = u_left - u_right
    if disparity <= 0:
        return np.nan, np.nan, np.nan
    else:
        Z = (f * B) / disparity
        X = ((u_left - cx) * Z) / f
        Y = ((v_left - cy) * Z) / f 
        return X, Y, Z

# --- Simulate YOLO Detections ---
# Imagine YOLO detected the same bird in both cameras:

# Detection 1: Bird detected in LEFT camera
# Bounding Box (hypothetical): [x_min, y_min, x_max, y_max] pixels
bbox_left = [1080, 490, 1120, 510]
# Calculate center point (u, v) for the left detection
u_left = (bbox_left[0] + bbox_left[2]) / 2.0
v_left = (bbox_left[1] + bbox_left[3]) / 2.0

# Detection 2: SAME bird detected in RIGHT camera
# Bounding Box (hypothetical): Should be shifted horizontally left (smaller u values)
bbox_right = [1030, 490, 1070, 510]
# Calculate center point (u, v) for the right detection
u_right = (bbox_right[0] + bbox_right[2]) / 2.0
v_right = (bbox_right[1] + bbox_right[3]) / 2.0 # v should be very similar to v_left

print("--- Simulated Detections ---")
print(f"Left BBox Center (u, v): ({u_left}, {v_left})")
print(f"Right BBox Center (u, v): ({u_right}, {v_right})")

# --- Calculate 3D Position ---
X, Y, Z = triangulate_point(u_left, v_left, u_right, v_right,
                            focal_length_px, baseline_m, cx, cy)

if np.isnan(Z):
    print("\nCould not calculate 3D position (invalid disparity).")
else:
    print("\n--- Calculated 3D Position (Relative to Left Camera) ---")
    print(f"X (meters - right): {X:.2f}")
    print(f"Y (meters - down): {Y:.2f}") # Y pixel axis points down, negative is up
    print(f"Z (meters - forward): {Z:.2f}")