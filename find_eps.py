import os
import face_recognition
import numpy as np
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import logging
from tqdm import tqdm
import cv2  # <-- 1. IMPORT CV2

# --- Configure Logging ---
logging.basicConfig(level=logging.INFO, format='%(message)s')

# --- Configuration ---
SOURCE_DIR = "source_images_subset"  # Using the small subset for speed
DETECTOR_MODEL = "cnn"
RESIZE_WIDTH = 400  # <-- 2. ADD RESIZE_WIDTH (using a fast value)


def find_optimal_eps():
    """
    Analyzes face encodings with resizing for performance and a dynamic progress bar.
    """
    logging.info(f"Scanning '{SOURCE_DIR}' for images...")
    encodings = []

    image_paths = []
    for root, dirs, files in os.walk(SOURCE_DIR):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(root, file))

    if not image_paths:
        logging.error(f"No images found in '{SOURCE_DIR}'. Please add photos to analyze.")
        return

    progress_bar = tqdm(image_paths, desc="Starting analysis...")

    for image_path in progress_bar:
        filename = os.path.basename(image_path)
        try:
            # --- 3. REPLACED IMAGE LOADING WITH RESIZING LOGIC ---
            progress_bar.set_description(f"Loading & Resizing {filename}")
            image = cv2.imread(image_path)

            if image is None:
                tqdm.write(f"Warning: Skipping corrupted image: {filename}")
                continue

            (h, w) = image.shape[:2]

            if w > RESIZE_WIDTH:
                r = float(RESIZE_WIDTH) / w
                dim = (RESIZE_WIDTH, int(h * r))
                image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # --- END OF REPLACEMENT ---

            progress_bar.set_description(f"Detecting faces in {filename}")
            locations = face_recognition.face_locations(rgb_image, model=DETECTOR_MODEL)

            progress_bar.set_description(f"Encoding {len(locations)} faces in {filename}")
            found_encodings = face_recognition.face_encodings(rgb_image, locations)

            encodings.extend(found_encodings)
        except Exception as e:
            tqdm.write(f"Warning: Skipping {filename} due to error: {e}")

    if len(encodings) < 2:
        logging.error("Not enough faces found to perform analysis. Need at least 2.")
        return

    logging.info(f"\nFound {len(encodings)} total faces. Calculating nearest neighbor distances...")

    # ... (The rest of the script for plotting remains exactly the same) ...
    neighbors = NearestNeighbors(n_neighbors=2)
    nbrs = neighbors.fit(encodings)
    distances, indices = nbrs.kneighbors(encodings)

    distances = np.sort(distances, axis=0)
    distances = distances[:, 1]

    plt.figure(figsize=(10, 6))
    plt.plot(distances)
    plt.title('K-Distance Graph for Epsilon Tuning (CNN Model)')
    plt.xlabel('Faces (sorted by distance to nearest neighbor)')
    plt.ylabel('Distance (eps value)')
    plt.grid(True)
    plt.savefig('eps_distance_plot_cnn.png') # Changed filename to avoid overwriting
    logging.info("Plot has been saved to eps_distance_plot_cnn.png")
    logging.info("\nDisplaying plot. Look for the 'elbow' or 'knee' of the curve.")
    logging.info("Close the plot window to exit the script.")
    plt.show()


if __name__ == "__main__":
    find_optimal_eps()