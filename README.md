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


####Project Deliverables & Results

1. Model Performance
   
Custom Model Trained: A custom best.pt YOLOv8 model was successfully fine-tuned on the GPU using the synthetic dataset.

Accuracy: The model achieved high accuracy metrics (e.g., mAP50 > 0.99) on the validation set, validating the quality of the synthetic data.

2. Final Pipeline Output
   
The final_pipeline.py script successfully ran the end-to-end simulation:

System Validation: The pipeline consistently detects the simulated bird, calculates its path, and triggers a verifiable COLLISION ALERT when the prediction logic dictates a threat below the 100-meter safety threshold.

Actionable Metric: The system outputs the exact Time-to-Impact (TTI), providing the essential warning metric for a pilot or autonomous system.


#### Repository contents

File/Folder,Description
final_pipeline.py,"The complete, integrated script. Contains all YOLO loading, triangulation, velocity, and collision prediction logic."
weights/best.pt,Final Model Output: Custom-trained YOLOv8 model weights (must be placed locally for execution).
labels/,Contains the 324 essential .txt bounding box label files (the manual ground truth).
data.yaml,YOLO configuration file defining the dataset structure and the single class (bird).
requirements.txt,"Lists all necessary Python dependencies (PyTorch, Ultralytics, spatialmath, etc.)."
