import torch
from diffusers import StableDiffusionPipeline
import os

# --- Configuration ---
# Use a smaller, faster Stable Diffusion model for this test
# model_id = "runwayml/stable-diffusion-v1-5" # Larger model
model_id = "CompVis/stable-diffusion-v1-4" # Slightly smaller/older
# Or even smaller/faster ones if needed (might require searching Hugging Face)

# Define the prompt (what image you want to create)
prompt = "A realistic photo of a single goose flying high in a clear blue sky"

# Define where to save the image
output_filename = "generated_bird_image.png"
# --- End Configuration ---

# --- Generate Image ---
print(f"Loading model: {model_id}")
print("This might take a while the first time it downloads the model...")

try:
    # Check if CUDA (GPU) is available, otherwise use CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load the pipeline (the pre-trained model and associated code)
    # If using CPU, specify torch_dtype=torch.float32 to avoid potential errors
    if device == "cpu":
        pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
    else:
        # If you had a GPU, you might use float16 for speed/less memory
        pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
        pipe = pipe.to(device) # Move the model to the GPU

    print("Model loaded. Generating image...")

    # Generate the image
    # num_inference_steps controls quality vs speed (lower is faster, maybe less quality)
    image = pipe(prompt, num_inference_steps=30).images[0]

    print(f"Image generated successfully. Saving to {output_filename}")

    # Save the image
    image.save(output_filename)

    print("Image saved.")

except Exception as e:
    print(f"\nAn error occurred: {e}")
    print("Model loading or image generation failed.")
    print("This could be due to memory limitations (especially on CPU) or network issues.")

# --- End Generate Image ---