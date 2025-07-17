# data_exploration.py
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

def explore_data(raw_data_path="data/raw"):
    """
    Loads and displays images from the raw data directory.
    """
    if not os.path.exists(raw_data_path):
        print(f"Error: Raw data directory not found at {raw_data_path}")
        return

    image_files = [f for f in os.listdir(raw_data_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]

    if not image_files:
        print(f"No image files found in {raw_data_path}. Please place some images (e.g., .jpg, .png) there.")
        return

    print(f"Found {len(image_files)} image(s) in {raw_data_path}. Displaying them now...")

    for i, image_file in enumerate(image_files):
        image_path = os.path.join(raw_data_path, image_file)
        print(f"Loading image: {image_path}")

        try:
            # Read the image using OpenCV
            image = cv2.imread(image_path)

            if image is None:
                print(f"Warning: Could not load image {image_file}. It might be corrupted or an unsupported format.")
                continue

            # OpenCV reads images in BGR format, Matplotlib expects RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            plt.figure(figsize=(10, 8))
            plt.imshow(image_rgb)
            plt.title(f"Image {i+1}: {image_file} (Shape: {image.shape})")
            plt.axis('off')
            plt.show()

        except Exception as e:
            print(f"Error processing image {image_file}: {e}")

if __name__ == "__main__":
    explore_data()