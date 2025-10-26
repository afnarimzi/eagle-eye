import numpy as np
from spatialmath import SE3


# Define the bird's position IN THE CAMERA'S FRAME (C)
# Let's say 100m ahead (X_c), 20m right (Y_c), 10m down (Z_c) relative TO THE CAMERA
P_C = np.array([100, 20, 10])
print(f"Bird Position in Camera Frame (P_C): {P_C}")

# Define the POSE of the Camera Frame (C) relative TO THE WORLD FRAME (W)
# Let's say the camera is located at X=5, Y=2, Z=1 in the World Frame
# AND it's rotated 90 degrees clockwise around the World's Z-axis (pointing down)
# We build this pose: First, rotation around Z by -90 deg (-np.pi/2), then translate.
# Remember: transformations combine right-to-left. Translate * Rotate
T_WC = SE3.Tz(1) * SE3.Ty(2) * SE3.Tx(5) * SE3.Rz(-np.pi / 2)
print(f"\nPose of Camera Frame relative to World Frame (T_WC):\n{T_WC}")


# To find the bird's position in the World Frame (P_W),
# we apply the camera's pose transformation (T_WC) to the point (P_C)
# Formula: P_W = T_WC * P_C
P_W = T_WC * P_C
print(f"\nCalculated Bird Position in World Frame (P_W):\n{P_W}")

# --- Transform Point from World Frame back to Camera Frame ---
# We can also go the other way. We need the inverse transformation:
# The pose of the World Frame relative to the Camera Frame (T_CW)
T_CW = T_WC.inv() # Calculate the inverse
print(f"\nPose of World Frame relative to Camera Frame (T_CW):\n{T_CW}")

# Apply the inverse transform to the world point P_W to get back P_C
# Formula: P_C_check = T_CW * P_W
P_C_check = T_CW * P_W
print(f"\nCalculated Bird Position back in Camera Frame (P_C_check):\n{P_C_check}")
print(f"(Should be close to the original P_C: {P_C})")