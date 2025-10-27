import os
import random
import shutil
from tqdm import tqdm 

# Path to the folder containing  original images
source_image_folder = "eagle_eye_dataset"
# Path to the folder containing original .txt labels
source_label_folder = "labels"
# Path to the base output folder where train/val will be created
output_base_folder = "dataset"
# Desired split ratio (e.g., 0.8 means 80% train, 20% validation)
train_split_ratio = 0.8

train_img_dir = os.path.join(output_base_folder, "images", "train")
val_img_dir = os.path.join(output_base_folder, "images", "val")
train_lbl_dir = os.path.join(output_base_folder, "labels", "train")
val_lbl_dir = os.path.join(output_base_folder, "labels", "val")

os.makedirs(train_img_dir, exist_ok=True)
os.makedirs(val_img_dir, exist_ok=True)
os.makedirs(train_lbl_dir, exist_ok=True)
os.makedirs(val_lbl_dir, exist_ok=True)
print("Created output directories.")

# Assume image files are .png and labels are .txt
# List all image files
all_images = [f for f in os.listdir(source_image_folder) if f.lower().endswith('.png')]
# Shuffle the list randomly
random.shuffle(all_images)
print(f"Found {len(all_images)} images.")

# Calculate split index
split_index = int(len(all_images) * train_split_ratio)

# Split into training and validation lists
train_files = all_images[:split_index]
val_files = all_images[split_index:]
print(f"Splitting into {len(train_files)} training files and {len(val_files)} validation files.")

# --- Copy Files ---
print("\nCopying training files...")
for filename in tqdm(train_files, desc="Training Set"):
    base_name = os.path.splitext(filename)[0]
    # Copy image
    shutil.copy(os.path.join(source_image_folder, filename),
                os.path.join(train_img_dir, filename))
    # Copy corresponding label
    label_filename = base_name + ".txt"
    label_src_path = os.path.join(source_label_folder, label_filename)
    if os.path.exists(label_src_path):
         shutil.copy(label_src_path, os.path.join(train_lbl_dir, label_filename))
    else:
         print(f"Warning: Label file not found for training image {filename}")


print("\nCopying validation files...")
for filename in tqdm(val_files, desc="Validation Set"):
    base_name = os.path.splitext(filename)[0]
    # Copy image
    shutil.copy(os.path.join(source_image_folder, filename),
                os.path.join(val_img_dir, filename))
    # Copy corresponding label
    label_filename = base_name + ".txt"
    label_src_path = os.path.join(source_label_folder, label_filename)
    if os.path.exists(label_src_path):
         shutil.copy(label_src_path, os.path.join(val_lbl_dir, label_filename))
    else:
         print(f"Warning: Label file not found for validation image {filename}")

print("\nDataset splitting complete.")