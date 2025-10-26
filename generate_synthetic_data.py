import torch
from diffusers import StableDiffusionPipeline
import os
import time # To add delays if needed and for unique filenames

# --- Configuration ---
# Model ID (can change later if needed)
model_id = "CompVis/stable-diffusion-v1-4"

# Output folder for generated images
output_folder = "synthetic_dataset"
os.makedirs(output_folder, exist_ok=True) # Create folder if needed

# Parameters for generation
num_inference_steps = 30 # Balance quality/speed
num_images_per_prompt = 1 # Generate one image per prompt for now

# Lists of elements to combine for prompts
bird_types = ["Black Kite", "House Crow", "Rock Pigeon", "Cattle Egret"]
weather_conditions = ["clear blue sky", "overcast sky", "hazy day"]
viewpoints = ["seen from slightly below", "eye-level view", "seen from slightly above"]
distances = ["far distance", "medium distance", "relatively close"]
times_of_day = ["midday sun", "golden hour sunset", "low light dusk"]

# --- Load Model ---
print(f"Loading model: {model_id}")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load the pipeline
if device == "cpu":
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
else:
    # If using GPU later (e.g., on Colab)
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe = pipe.to(device)
print("Model loaded.")

# --- Generation Loop (To Be Added) ---
print("\nStarting image generation loop...")

# --- Generation Loop ---
print("\nStarting image generation loop...")
image_counter = 0

# Loop through combinations
for bird in bird_types:
    for weather in weather_conditions:
        for view in viewpoints:
            for dist in distances:
                for tod in times_of_day:
                    # Construct the prompt
                    prompt = f"Photorealistic, {view}, single {bird} flying against {weather}, {dist}, {tod}"

                    # Construct a unique filename
                    # Example: synthetic_dataset/Black_Kite_clear_blue_sky_seen_from_slightly_below_far_distance_midday_sun_000.png
                    safe_bird = bird.replace(" ", "_")
                    safe_weather = weather.replace(" ", "_")
                    safe_view = view.replace(" ", "_")
                    safe_dist = dist.replace(" ", "_")
                    safe_tod = tod.replace(" ", "_")
                    base_filename = f"{safe_bird}_{safe_weather}_{safe_view}_{safe_dist}_{safe_tod}_{image_counter:03d}.png"
                    output_filepath = os.path.join(output_folder, base_filename)

                    print(f"\nGenerating image {image_counter+1}...")
                    print(f"  Prompt: {prompt}")
                    print(f"  Saving to: {output_filepath}")

                    try:
                        # Generate the image (only 1 per prompt for now)
                        image = pipe(prompt, num_inference_steps=num_inference_steps).images[0]

                        # Save the image
                        image.save(output_filepath)
                        print("  Image saved.")
                        image_counter += 1

                        # Optional: Add a small delay if needed, e.g., time.sleep(1)

                    except Exception as e:
                        print(f"  ERROR generating image for this prompt: {e}")
                        # Continue to the next prompt even if one fails

print(f"\nGenerated {image_counter} images in total.")
print("\nImage generation complete.")

print("\nImage generation complete.")

# --- End Script ---