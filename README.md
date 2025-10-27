##  Project: Eagle Eye - Avian Hazard Avoidance System (A-HAMS)

This project develops a proof-of-concept for an onboard, real-time computer vision system designed to detect birds, track their 3D movement, and predict potential mid-air collisions with aircraft.


##  Project Overview

The primary objective was to create a functional prototype that bridges the gap between conventional 2D object detection and safety-critical 3D prediction. The pipeline demonstrates how to use modern AI to provide pilots with crucial Time-to-Impact (TTI) warnings.


## Core Technology & Methodology

The solution integrates four essential components:

AI Object Detection (YOLOv8): Used as the core sensing element to identify birds in video frames.

Synthetic Data Generation: We solved the critical data scarcity problem by leveraging Generative AI (Stable Diffusion) to create 324 unique training images of birds under various aerial conditions (a fundamental necessity for robust aviation models).

3D Spatial Math: We implemented principles of Stereo Triangulation and Coordinate Frame Transformation (SE3) to convert the bird's 2D pixel coordinates into precise 3D physical coordinates (meters).

Collision Prediction: The system performs path projection using the calculated 3D velocity vectors of the bird and the simulated aircraft to determine if their paths will intersect within a set safety margin.


## Project Deliverables & Results

1. Model Training and Performance
Custom Model Trained: A custom best.pt YOLOv8 model was successfully fine-tuned on the GPU using the synthetic dataset.

Accuracy: The model achieved high accuracy metrics (e.g., mAP50 > 0.99) on the validation set, proving that synthetic data can effectively train an AI for this task.

2. Full Integrated Pipeline Output
The final_pipeline.py script successfully links the detection output to the threat assessment:

3D Tracking Verified: The pipeline accurately calculates the bird's changing 3D position and velocity vector based on simulated stereo input.

Collision Alert Verified: Under simulated aggressive flight conditions, the system consistently triggers a COLLISION ALERT when the distance falls below the set safety threshold (100 meters), calculating the exact Time-to-Impact (TTI).
