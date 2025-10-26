import numpy as np
from spatialmath import SE3 

# --- Define a Point ---
# Represent a point in 3D space (e.g., bird's initial position)
# Using a NumPy array: [X, Y, Z] in meters
# Let's say the bird is 100m ahead (X), 20m right (Y), 10m up (Z)
bird_position_initial = np.array([100, 20, 10])
print(f"Initial Bird Position (X, Y, Z): {bird_position_initial}")

# --- Define a Transformation  ---
# Let's say we want to represent the bird moving 5 meters forward (positive X)
# We create an SE3 object representing this translation
# SE3.Tx(d) creates a translation along the X-axis by distance d
move_forward_transform = SE3.Tx(5)
print("\nTransformation Object (Moving 5m in X):")
print(move_forward_transform)
# You can still access the 4x4 matrix if needed:
# print("\nTransformation Matrix:")
# print(move_forward_transform.A)


# --- Apply the Transformation ---
# The SE3 object can directly transform a 3D NumPy array vector
bird_position_new = move_forward_transform * bird_position_initial

print(f"\nNew Bird Position (X, Y, Z): {bird_position_new}")

# --- Direct Calculation ---
# For simple translation, you can just add vectors directly
move_vector = np.array([5, 0, 0])
bird_position_new_simple = bird_position_initial + move_vector
print(f"\nNew Bird Position (Simple Addition): {bird_position_new_simple}")