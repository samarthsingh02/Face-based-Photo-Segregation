import os
import face_recognition
import numpy as np
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import logging
from tqdm import tqdm


# --- Configure Logging ---
# (Using a simple logger for this helper script)
logging.basicConfig(level=logging.INFO, format='%(message)s')

# --- Configuration ---
SOURCE_DIR = "source_images"
# Using HOG is faster and good enough for this analysis
DETECTOR_MODEL = "cnn"


def find_optimal_eps():
    """
    Analyzes face encodings to help find an optimal eps value for DBSCAN
    by plotting a k-distance graph.
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

    for image_path in tqdm(image_paths, desc="Finding Faces (eps analysis)"):
        try:
            image = face_recognition.load_image_file(image_path)
            locations = face_recognition.face_locations(image, model=DETECTOR_MODEL)
            found_encodings = face_recognition.face_encodings(image, locations)
            encodings.extend(found_encodings)
        except Exception as e:
            # We can use tqdm.write to print messages without breaking the progress bar
            tqdm.write(f"Warning: Skipping {os.path.basename(image_path)} due to error: {e}")

    if len(encodings) < 2:
        logging.error("Not enough faces found to perform analysis. Need at least 2.")
        return

    logging.info(f"\nFound {len(encodings)} total faces. Calculating nearest neighbor distances...")

    # Calculate the distance to the 2nd nearest neighbor (k=2) for each point.
    # We use k=2 because each point's closest neighbor is itself at distance 0.
    neighbors = NearestNeighbors(n_neighbors=2)
    nbrs = neighbors.fit(encodings)
    distances, indices = nbrs.kneighbors(encodings)

    # Get the distances to the nearest neighbor and sort them
    distances = np.sort(distances, axis=0)
    distances = distances[:, 1]

    # Plot the k-distance graph
    plt.figure(figsize=(10, 6))
    plt.plot(distances)
    plt.title('K-Distance Graph for Epsilon Tuning')
    plt.xlabel('Faces (sorted by distance to nearest neighbor)')
    plt.ylabel('Distance (eps value)')
    plt.grid(True)

    # Save the plot to a file before showing it
    plt.savefig('eps_distance_plot.png')
    logging.info("Plot has been saved to eps_distance_plot.png")

    logging.info("\nDisplaying plot. Look for the 'elbow' or 'knee' of the curve.")
    logging.info("Close the plot window to exit the script.")
    plt.show()


if __name__ == "__main__":
    find_optimal_eps()