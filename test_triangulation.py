import numpy as np


# Focal length of the cameras (in pixels) - assume same for both
focal_length_px = 700.0

# Baseline: Distance between the centers of the two cameras (in meters)
baseline_m = 0.5 

# Principal Point (image center in pixels) - assume same for both
# Often near the center of the image resolution (e.g., width/2, height/2)
cx = 960.0 # Example: center of a 1920-width image
cy = 540.0 # Example: center of a 1080-height image

# --- Detected Pixel Coordinates  ---

# Pixel coordinates of the bird detected in the LEFT camera image (u = column, v = row)
u_left = 1100.0
v_left = 500.0

# Pixel coordinates of the SAME bird detected in the RIGHT camera image
u_right = 1050.0
v_right = 500.0 # Note: In ideal parallel cameras, v_left and v_right are the same

# --- Triangulation Calculation ---

print("--- Inputs ---")
print(f"Focal Length (px): {focal_length_px}")
print(f"Baseline (m): {baseline_m}")
print(f"Principal Point (cx, cy): ({cx}, {cy})")
print(f"Left Camera Detection (u, v): ({u_left}, {v_left})")
print(f"Right Camera Detection (u, v): ({u_right}, {v_right})")

# Calculate Disparity
# Disparity is the difference in the horizontal pixel position (u) between the two images
disparity = u_left - u_right
print(f"\nCalculated Disparity (pixels): {disparity}")

# Basic Triangulation Formulas (for parallel cameras, Z relative to left camera)
if disparity <= 0:
    print("\nError: Disparity must be positive for this simple model.")
    # In real systems, disparity can be negative if point is behind cameras,
    # or zero/small if very far away or error in detection.
    X = Y = Z = np.nan # Not a Number
else:
    # Calculate Depth (Z) - Distance from the LEFT camera baseline plane
    Z = (focal_length_px * baseline_m) / disparity

    # Calculate X coordinate (relative to the LEFT camera center)
    X = ( (u_left - cx) * Z ) / focal_length_px

    # Calculate Y coordinate (relative to the LEFT camera center)
    Y = ( (v_left - cy) * Z ) / focal_length_px

    print("\n--- Calculated 3D Position (Relative to Left Camera) ---")
    print(f"X (meters - right): {X:.2f}")
    print(f"Y (meters - down): {Y:.2f}") # Y pixel axis often points down
    print(f"Z (meters - forward): {Z:.2f}")

