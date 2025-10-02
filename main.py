import os
import shutil
import time
import logging
import json
import cv2
import yaml

import face_recognition
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics.cluster import homogeneity_score, completeness_score, v_measure_score

# --- Load Configuration ---
# This block now correctly loads all settings from the YAML file.
try:
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Directory Paths
    SOURCE_DIR = config['directory_paths']['source']
    OUTPUT_DIR = config['directory_paths']['output']
    LOG_FILE = config['directory_paths']['log_file']
    GROUND_TRUTH_FILE = config['directory_paths']['ground_truth']

    # Model Parameters
    DETECTOR_MODEL = config['model_parameters']['detector']
    EPS_VALUE = config['model_parameters']['clustering']['eps']
    MIN_SAMPLES = config['model_parameters']['clustering']['min_samples']

    # Processing Settings
    RESIZE_WIDTH = config['processing_settings']['resize_width']

    # Output Settings
    FOLDER_PREFIX = config['output_settings']['folder_prefix']
    UNKNOWNS_FOLDER = config['output_settings']['unknowns_folder']

except FileNotFoundError:
    print(f"CRITICAL: config.yaml not found. Please ensure it exists.")
    exit()
except KeyError as e:
    print(f"CRITICAL: Missing or incorrect key in config.yaml: {e}")
    exit()


# --- Logger Configuration ---
def init_logging():
    """Sets up the logging configuration using the path from config."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(LOG_FILE, mode='a'),  # Use the LOG_FILE variable
            logging.StreamHandler()
        ]
    )


# --- Phase 1: Data Ingestion & Feature Extraction ---
def process_images():
    logging.info("Starting image processing...")
    all_face_data = []
    image_paths = []
    for root, dirs, files in os.walk(SOURCE_DIR):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(root, file))

    if not image_paths:
        logging.warning(f"No images found in the '{SOURCE_DIR}' directory.")
        return None, 0, 0

    total_faces_found = 0
    for i, image_path in enumerate(image_paths):
        logging.info(f"Processing image {i + 1}/{len(image_paths)}: {os.path.basename(image_path)}")
        try:
            image = cv2.imread(image_path)
            (h, w) = image.shape[:2]

            # Use RESIZE_WIDTH from config
            if w > RESIZE_WIDTH:
                r = float(RESIZE_WIDTH) / w
                dim = (RESIZE_WIDTH, int(h * r))
                image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
                logging.info(f"  Resized large image to {dim[0]}x{dim[1]}")

            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            locations = face_recognition.face_locations(rgb_image, model=DETECTOR_MODEL)
            logging.info(f"  Found {len(locations)} face(s) using '{DETECTOR_MODEL}' model.")
            total_faces_found += len(locations)
            encodings = face_recognition.face_encodings(rgb_image, locations)
            for encoding in encodings:
                all_face_data.append({'image_path': image_path, 'encoding': encoding})
        except Exception as e:
            logging.error(f"  Could not process {os.path.basename(image_path)}. Error: {e}")

    return all_face_data, len(image_paths), total_faces_found

# --- Phase 2: Clustering & Analysis ---
def cluster_faces(all_face_data):
    if not all_face_data:
        logging.warning("No faces were detected to cluster.")
        return None, 0, 0

    logging.info("Starting face clustering...")
    encodings = np.array([data['encoding'] for data in all_face_data])

    # Use EPS_VALUE and MIN_SAMPLES from config
    clt = DBSCAN(metric="euclidean", eps=EPS_VALUE, min_samples=MIN_SAMPLES)
    clt.fit(encodings)

    unique_labels = set(clt.labels_)
    num_people = len(unique_labels) - (1 if -1 in unique_labels else 0)
    num_unknowns = list(clt.labels_).count(-1)
    logging.info(f"Found {num_people} unique people (clusters) and {num_unknowns} unknown faces.")

    for i, label in enumerate(clt.labels_):
        all_face_data[i]['cluster_id'] = label

    return all_face_data, num_people, num_unknowns

# --- Phase 3: File Organization & Output ---
def organize_files(all_face_data):
    if not all_face_data:
        logging.info("No data to organize.")
        return

    logging.info("Organizing files into folders...")
    copied_files = set()
    for data in all_face_data:
        cluster_id = data['cluster_id']
        image_path = data['image_path']
        if cluster_id == -1:
            # Use UNKNOWNS_FOLDER from config
            folder_name = UNKNOWNS_FOLDER
        else:
            # Use FOLDER_PREFIX from config
            folder_name = f"{FOLDER_PREFIX}{cluster_id + 1}"
        destination_folder = os.path.join(OUTPUT_DIR, folder_name)
        os.makedirs(destination_folder, exist_ok=True)
        copy_key = (image_path, destination_folder)
        if copy_key not in copied_files:
            shutil.copy(image_path, destination_folder)
            copied_files.add(copy_key)
    logging.info("File organization complete!")

# --- Evaluation Functions ---
def calculate_clustering_metrics(all_face_data):
    """
    Calculates and formats clustering quality metrics by comparing DBSCAN results
    to the ground truth derived from the folder structure.
    """
    if not all_face_data:
        logging.warning("Cannot calculate metrics: No face data provided.")
        return "\n    No face data to calculate metrics."

    # 1. Derive ground truth labels from the directory structure.
    # Assumes a structure like: .../source_images/Person_Name/image.jpg
    true_labels = [os.path.basename(os.path.dirname(data['image_path'])) for data in all_face_data]
    predicted_labels = [data['cluster_id'] for data in all_face_data]

    # 2. Check if the ground truth is meaningful for clustering evaluation.
    # Metrics are only useful if there are at least two different people to compare.
    unique_true_labels = set(true_labels)
    if len(unique_true_labels) < 2:
        logging.warning(f"Cannot calculate meaningful metrics. Only found {len(unique_true_labels)} unique source folder(s).")
        report = f"""
    ----------------- CLUSTERING METRICS -----------------
    - Status: Not Calculated
    - Reason: Meaningful metrics require at least two distinct
              people (source folders) to compare against.
    ------------------------------------------------------
        """
        return report

    # 3. Calculate the standard clustering metrics.
    homogeneity = homogeneity_score(true_labels, predicted_labels)
    completeness = completeness_score(true_labels, predicted_labels)
    v_measure = v_measure_score(true_labels, predicted_labels)

    # 4. Build a more descriptive and clearly ordered report.
    report_lines = [
        "\n",
        "----------------- CLUSTERING METRICS -----------------",
        f"- V-Measure   : {v_measure:.4f} (The overall balanced score)",
        f"- Homogeneity : {homogeneity:.4f} (Measures if each cluster contains only one person)",
        f"- Completeness: {completeness:.4f} (Measures if all photos of a person are in one cluster)",
        "------------------------------------------------------"
    ]
    return "\n".join(report_lines)

# --- Logging Summary ---
def log_run_summary(stats):
    # ... (This function is now correct as it uses the right key) ...
    # ... (No changes needed in this function) ...
    summary = f"""
    -------------------- RUN SUMMARY --------------------
    - Timestamp           : {stats['timestamp']}
    - Detector Model      : {stats['detector_model']}
    - DBSCAN eps          : {stats['eps_value']}
    - Images Processed    : {stats['images_processed']}
    - Faces Detected      : {stats['faces_detected']}
    - People Found        : {stats['people_found']}
    - Unknowns / Outliers : {stats['unknown_faces']} face(s)
    - Total Execution Time: {stats['execution_time']:.2f} seconds
    -----------------------------------------------------
    """
    summary += stats.get("clustering_report", "")
    logging.info(summary)

# --- Main Execution ---
if __name__ == "__main__":
    init_logging()
    start_time = time.time()

    run_stats = {
        "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
        "detector_model": DETECTOR_MODEL,
        "eps_value": EPS_VALUE
    }

    logging.info(f"-- Starting New Run --")
    logging.info(f"Parameters: detector={DETECTOR_MODEL}, eps={EPS_VALUE}")

    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR)

    face_data, img_count, face_count = process_images()
    run_stats["images_processed"] = img_count
    run_stats["faces_detected"] = face_count

    if face_data:
        clustered_data, people_count, unknown_count = cluster_faces(face_data)
        run_stats["people_found"] = people_count
        run_stats["unknown_faces"] = unknown_count
        organize_files(clustered_data)
        run_stats["clustering_report"] = calculate_clustering_metrics(clustered_data)
    else:
        run_stats["people_found"] = 0
        run_stats["unknown_faces"] = 0
        run_stats["clustering_report"] = "No faces processed."

    end_time = time.time()
    run_stats["execution_time"] = end_time - start_time

    log_run_summary(run_stats)