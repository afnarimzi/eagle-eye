import cv2
import os

# --- Configuration ---
# Define the path to your sample image
image_path = 'sample_image.jpg'
# --- End Configuration ---

# --- Load and Display Image ---
print(f"Loading image: {image_path}")

# Check if the image file exists
if not os.path.exists(image_path):
    print(f"Error: Image file not found at {image_path}")
else:
    # Load the image using OpenCV
    # cv2.imread() reads the image from the specified file path
    img = cv2.imread(image_path)

    # Check if the image was loaded successfully
    if img is None:
        print(f"Error: Could not load image from {image_path}. Check file format/corruption.")
    else:
        print("Image loaded successfully. Displaying...")
        # Display the image in a window named "Sample Image"
        # cv2.imshow() creates a window and shows the image in it
        cv2.imshow('Sample Image', img)

        # Wait indefinitely until a key is pressed
        # cv2.waitKey(0) pauses the script. The window stays open
        # until you press any key on your keyboard while the window is active.
        print("Press any key on the image window to close.")
        cv2.waitKey(0)

        # Close all OpenCV windows
        # cv2.destroyAllWindows() cleans up and closes the display window
        cv2.destroyAllWindows()
        print("Window closed.")
# --- End Load and Display Image ---