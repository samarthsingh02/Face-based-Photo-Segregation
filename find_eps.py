import os
import yaml
import logging
import cv2
from tqdm import tqdm
import numpy as np
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import core_engine  # Import your engine

# --- Configure Logging ---
logging.basicConfig(level=logging.INFO, format='%(message)s')

# --- Load Configuration ---
try:
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    active_model_name = config['active_model']
    preset_settings = config['presets'][active_model_name]
    SOURCE_DIR = config['directory_paths']['source']
except Exception as e:
    print(f"CRITICAL ERROR loading config.yaml: {e}")
    exit()


def find_optimal_eps():
    """
    Analyzes face encodings for the active preset to find an optimal eps value.
    """
    logging.info(f"--- Finding optimal EPS for preset: '{active_model_name}' ---")

    # Use the process_images function from the core engine to get all embeddings
    # We pass an empty set for existing_paths to force it to process all images.
    face_data, _, _ = core_engine.process_images(
        SOURCE_DIR, preset_settings, existing_paths=set()
    )

    if not face_data or len(face_data) < 2:
        logging.error("Not enough faces found to perform analysis. Need at least 2.")
        return

    encodings = np.array([data['encoding'] for data in face_data])
    logging.info(f"\nFound {len(encodings)} total faces. Calculating nearest neighbor distances...")

    # ... (The rest of the script for calculating distances and plotting is the same) ...
    neighbors = NearestNeighbors(n_neighbors=2)
    nbrs = neighbors.fit(encodings)
    distances, indices = nbrs.kneighbors(encodings)
    distances = np.sort(distances, axis=0)[:, 1]

    plt.figure(figsize=(10, 6))
    plt.plot(distances)
    plt.title(f'K-Distance Graph for Epsilon Tuning ({active_model_name})')
    plt.xlabel('Faces (sorted by distance)')
    plt.ylabel(f"Distance (eps) - Metric: {preset_settings['clustering']['metric']}")
    plt.grid(True)

    plot_filename = f'eps_plot_{active_model_name}.png'
    plt.savefig(plot_filename)
    logging.info(f"Plot has been saved to {plot_filename}")
    plt.show()


if __name__ == "__main__":
    find_optimal_eps()