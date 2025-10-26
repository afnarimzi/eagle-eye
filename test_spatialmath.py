import numpy as np
from spatialmath import SE3 

# --- Define a Point ---
# Bird is 100m ahead (X), 20m right (Y), 10m up (Z)
bird_position_initial = np.array([100, 20, 10])
print(f"Initial Bird Position (X, Y, Z): {bird_position_initial}")

# --- Define a Rotation Transformation ---
# Let's rotate the bird's position 90 degrees counter-clockwise
# around the Z-axis (upwards axis).
# SE3.Rz(angle_in_radians) creates a rotation around Z.
# We need numpy for pi: np.pi / 2 is 90 degrees.
rotation_transform = SE3.Rz(np.pi / 2)
print("\nRotation Transformation Object (90 deg around Z):")
print(rotation_transform)

# --- Apply the Rotation ---
# Apply the rotation to the initial position
# SE3 objects use * for transformation
bird_position_rotated = rotation_transform * bird_position_initial
print(f"\nRotated Bird Position (X, Y, Z): {bird_position_rotated}")

# --- Combine Rotation and Translation ---
# Let's define a combined pose: rotate 90 deg around Z, then move 5m along the NEW X-axis
# Transformations are applied right-to-left: First rotate (Rz), then translate (Tx)
combined_transform = SE3.Tx(5) * SE3.Rz(np.pi / 2)
print("\nCombined Transformation Object (Rotate then Translate):")
print(combined_transform)

# Apply the combined transform to the original point
bird_position_final = combined_transform * bird_position_initial
print(f"\nFinal Bird Position after Rotate & Translate (X, Y, Z): {bird_position_final}")