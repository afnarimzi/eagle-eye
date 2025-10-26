import torch
from diffusers import StableDiffusionPipeline
import os 

# --- Configuration ---
# Use a smaller, faster Stable Diffusion model for this test
model_id = "CompVis/stable-diffusion-v1-4" 

# Define the prompt 
prompt = "Vast clear blue sky, midday sun, photorealistic, single Black Kite soaring very high, seen from below"
output_folder = "generated_images" 
base_filename = "generated_kite_cockpit2.png" 
output_filepath = os.path.join(output_folder, base_filename) 

os.makedirs(output_folder, exist_ok=True) 
# --- End Configuration ---

# --- Generate Image ---
print(f"Loading model: {model_id}")
print("This might take a while the first time it downloads the model...")

try:
    # Check if CUDA (GPU) is available, otherwise use CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load the pipeline (the pre-trained model and associated code)
    if device == "cpu":
        pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
    else:
        pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
        pipe = pipe.to(device)

    print("Model loaded. Generating image...")

    image = pipe(prompt, num_inference_steps=30).images[0]

    print(f"Image generated successfully. Saving to {output_filepath}") 

    # Save the image
    image.save(output_filepath) 

    print("Image saved.")

except Exception as e:
    print(f"\nAn error occurred: {e}")
    print("Model loading or image generation failed.")
    print("This could be due to memory limitations (especially on CPU) or network issues.")

