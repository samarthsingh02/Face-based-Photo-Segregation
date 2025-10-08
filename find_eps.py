import os
import yaml
import logging
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import NearestNeighbors
import core_engine

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

    # --- CHANGE: Call the correct function based on the library in the preset ---
    face_data = None
    if preset_settings['library'] == 'dlib':
        # Force re-processing of all images for this analysis
        face_data, _, _ = core_engine.process_images_dlib(
            SOURCE_DIR, preset_settings, existing_paths=set()
        )
    elif preset_settings['library'] == 'deepface':
        # Force re-processing of all images for this analysis
        face_data, _, _ = core_engine.process_images_deepface(
            SOURCE_DIR, preset_settings, existing_paths=set()
        )
    else:
        logging.error(f"Unknown library specified in preset: {preset_settings['library']}")
        return
    # --- END OF CHANGE ---

    if not face_data or len(face_data) < 2:
        logging.error("Not enough faces found to perform analysis. Need at least 2.")
        return

    logging.info(f"\nFound {len(face_data)} total faces. Calculating nearest neighbor distances...")
    encodings = np.array([data['encoding'] for data in face_data])

    neighbors = NearestNeighbors(n_neighbors=2)
    nbrs = neighbors.fit(encodings)
    distances, indices = nbrs.kneighbors(encodings)
    distances = np.sort(distances, axis=0)[:, 1]

    plt.figure(figsize=(10, 6))
    plt.plot(distances)
    plt.title(f'K-Distance Graph for Epsilon Tuning ({active_model_name})')
    plt.xlabel('Faces (sorted by distance to nearest neighbor)')
    plt.ylabel(f"Distance (eps) - Metric: {preset_settings['clustering']['metric']}")
    plt.grid(True)

    plot_filename = f'eps_plot_{active_model_name}.png'
    plt.savefig(plot_filename)
    logging.info(f"Plot has been saved to {plot_filename}")
    plt.show()


if __name__ == "__main__":
    find_optimal_eps()