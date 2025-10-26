import numpy as np
import time

time_step = 0.1 # Simulate checking every 0.1 seconds
simulation_duration = 10.0 # Simulate for 10 seconds into the future
collision_threshold_m = 50.0 # Minimum safe distance in meters


aircraft_pos = np.array([0.0, 0.0, 10.0]) # Starts at origin, 10m altitude
# Velocity [Vx, Vy, Vz] in meters per second. Assume takeoff speed.
aircraft_vel = np.array([70.0, 0.0, 0.0]) # 70 m/s (~250 km/h) along X

# Bird State (relative to the SAME world origin)
# Let's use the position and velocity we calculated before, but imagine
# we transformed them into the world frame.
# Example: Bird starts ahead and to the right, moving somewhat towards plane's path.
# Position (t1 in world frame - hypothetical)
bird_pos = np.array([700.0, 50.0, 15.0]) # 700m ahead, 50m right, 15m altitude
# Velocity (calculated in world frame - hypothetical)
bird_vel = np.array([-15.0, -5.0, 0.5]) # Moving towards plane (neg X), left (neg Y), slightly up (pos Z)

print("--- Initial State ---")
print(f"Aircraft Position: {aircraft_pos}, Velocity: {aircraft_vel} m/s")
print(f"Bird Position: {bird_pos}, Velocity: {bird_vel} m/s")
print(f"Collision Threshold: {collision_threshold_m} m")

# --- Simulation Loop ---
print("\n--- Simulating Future Positions ---")
collision_predicted = False
time_to_collision = np.inf # Initialize to infinity

current_time = 0.0
while current_time <= simulation_duration:
    # Predict future positions using: new_pos = initial_pos + velocity * time
    future_aircraft_pos = aircraft_pos + aircraft_vel * current_time
    future_bird_pos = bird_pos + bird_vel * current_time

    # Calculate the distance between the two future positions
    distance = np.linalg.norm(future_aircraft_pos - future_bird_pos)

    print(f"Time: {current_time:.1f}s | Distance: {distance:.2f} m")

    # Check if distance is below the collision threshold
    if distance < collision_threshold_m:
        print(f"  *** COLLISION PREDICTED at {current_time:.1f} seconds! (Distance: {distance:.2f}m) ***")
        collision_predicted = True
        time_to_collision = current_time
        break # Stop simulation once collision is predicted

    current_time += time_step

if not collision_predicted:
    print(f"\nNo collision predicted within {simulation_duration} seconds.")
else:
    print(f"\nCollision alert! Predicted impact time: {time_to_collision:.1f} seconds.")

