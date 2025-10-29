##   Kestrel 


This project develops a proof-of-concept for an onboard, real-time computer vision system designed to detect birds, track their 3D movement, and predict potential mid-air collisions with UAVs and validate UAV missions using GPS waypoints and coordinate transformations across WGS84, ECEF, NED, and FRD frames


### Methodology and Innovation


The system is engineered in three integrated stages, solving key technical challenges:

#### 1. Data Solution: Synthetic Training
Challenge: The extreme scarcity of real-world cockpit footage showing birds in various conditions.

Solution: Solved the problem by leveraging Generative AI (Stable Diffusion) to synthesize a custom, robust dataset of 324 aerial bird images for model training. The dataset was meticulously labeled using manual Data Labelling (Bounding Boxes).

#### 2. 3D Localization and Tracking
Challenge: Converting the flat, 2D pixel coordinates from a camera into the bird's true position and velocity in 3D space.

Solution: Integrated Stereo Triangulation and Spatial Math (using SE3 transforms) to convert simulated two-camera detections into precise 3D position vectors and calculate the bird's subsequent velocity.

#### 3. Real-Time Threat Assessment
Challenge: Determining if the calculated 3D path of the bird intersects the aircraft's path.

Solution: Implemented a path projection algorithm that uses the calculated bird velocity and the aircraft's simulated flight vector to compute the predicted Closest Point of Approach (CPA) and trigger a COLLISION ALERT if the distance falls below the safety threshold.
