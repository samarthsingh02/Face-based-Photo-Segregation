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
from collections import Counter
from tqdm import tqdm

# --- Load Configuration ---
try:
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # 1. Get the name of the active preset
    active_model_name = config['active_model']

    # 2. Load the settings from that specific preset
    preset_settings = config['presets'][active_model_name]

    DETECTOR_MODEL = preset_settings['detector']
    EPS_VALUE = preset_settings['eps']
    RESIZE_WIDTH = preset_settings['resize_width']

    # 3. Load the general settings from their correct locations
    SOURCE_DIR = config['directory_paths']['source']
    OUTPUT_DIR = config['directory_paths']['output']
    LOG_FILE = config['directory_paths']['log_file']
    GROUND_TRUTH_FILE = config['directory_paths']['ground_truth']

    MIN_SAMPLES = config['clustering_settings']['min_samples']

    FOLDER_PREFIX = config['output_settings']['folder_prefix']
    UNKNOWNS_FOLDER = config['output_settings']['unknowns_folder']

except FileNotFoundError:
    # We use print here because the logger might not be initialized yet
    print("CRITICAL ERROR: config.yaml not found. Please ensure it exists in the project directory.")
    exit()
except KeyError as e:
    print(f"CRITICAL ERROR: Missing or incorrect key in config.yaml. Please check your file. Error: {e}")
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


def process_images():
    """
    Scans the source directory, resizes images, finds faces, and extracts encodings,
    displaying a progress bar using tqdm.
    """
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

    # --- CHANGE: The main loop is now wrapped in tqdm ---
    for image_path in tqdm(image_paths, desc="Processing Images"):
        # --- REMOVED: The old per-image logging message is no longer needed ---
        try:
            image = cv2.imread(image_path)

            # Added a check for corrupted images that OpenCV can't open
            if image is None:
                tqdm.write(f"Warning: Skipping corrupted or unreadable image: {os.path.basename(image_path)}")
                continue

            (h, w) = image.shape[:2]

            if w > RESIZE_WIDTH:
                r = float(RESIZE_WIDTH) / w
                dim = (RESIZE_WIDTH, int(h * r))
                image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            locations = face_recognition.face_locations(rgb_image, model=DETECTOR_MODEL)
            total_faces_found += len(locations)
            encodings = face_recognition.face_encodings(rgb_image, locations)

            for encoding in encodings:
                all_face_data.append({'image_path': image_path, 'encoding': encoding})

        except Exception as e:
            # Use tqdm.write() to print messages without breaking the progress bar
            tqdm.write(f"ERROR: Could not process {os.path.basename(image_path)}. Error: {e}")

    # Log a final summary after the loop is complete
    logging.info(f"Completed processing. Found a total of {total_faces_found} faces in {len(image_paths)} images.")
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
    Calculates a comprehensive clustering report, separating metrics for single-person
    photos from an analysis of group photos.
    """
    if not all_face_data:
        logging.warning("Cannot calculate metrics: No face data provided.")
        return "\n    No face data to calculate metrics."

    # --- 1. Separate single-person data from group photo data ---
    single_person_data = []
    group_photo_data = []
    for data in all_face_data:
        # Assumes group photos are in a folder starting with '_' (e.g., _group_photos)
        if os.path.basename(os.path.dirname(data['image_path'])).startswith('_'):
            group_photo_data.append(data)
        else:
            single_person_data.append(data)

    if not single_person_data:
        return "\n    No single-person photos found to calculate metrics."

    # --- 2. Calculate metrics ONLY on single-person photos ---
    true_labels = [os.path.basename(os.path.dirname(data['image_path'])) for data in single_person_data]
    predicted_labels = [data['cluster_id'] for data in single_person_data]

    if len(set(true_labels)) < 2:
        # ... (error handling for < 2 unique labels remains the same) ...
        return "\n    Metrics not calculated: Need at least two source folders of single people."

    homogeneity = homogeneity_score(true_labels, predicted_labels)
    completeness = completeness_score(true_labels, predicted_labels)
    v_measure = v_measure_score(true_labels, predicted_labels)

    # --- 3. Create a cluster-to-name map based on the clean, single-person data ---
    cluster_to_name_map = {}
    unique_predicted_labels = set(predicted_labels)
    for cluster_id in unique_predicted_labels:
        if cluster_id == -1: continue
        names_in_cluster = [true_labels[i] for i, pred_label in enumerate(predicted_labels) if pred_label == cluster_id]
        if names_in_cluster:
            winner_name = Counter(names_in_cluster).most_common(1)[0][0]
            cluster_to_name_map[cluster_id] = winner_name

    # --- 4. Identify mismatches ONLY for single-person photos ---
    mismatches = []
    for i, data in enumerate(single_person_data):
        true_name = true_labels[i]
        cluster_id = predicted_labels[i]
        predicted_name = cluster_to_name_map.get(cluster_id, "Unknown")
        if predicted_name != true_name:
            filename = os.path.basename(data['image_path'])
            mismatches.append(f"  - FAILED: '{filename}' (True: {true_name}) was predicted as '{predicted_name}'")

    # --- 5. Analyze the group photos separately ---
    group_photo_summary = []
    if group_photo_data:
        group_predictions = [cluster_to_name_map.get(data['cluster_id'], "Unknown") for data in group_photo_data]
        group_counts = Counter(group_predictions)
        group_photo_summary.append("\n- Group Photo Analysis:")
        for name, count in group_counts.items():
            group_photo_summary.append(f"  - Identified {count} face(s) as '{name}'")

    # --- 6. Build the final, combined report ---
    report_lines = [
        "\n",
        "----------------- CLUSTERING METRICS (Single Photos Only) -----------------",
        f"- V-Measure   : {v_measure:.4f} (Overall balanced score)",
        f"- Homogeneity : {homogeneity:.4f} (Purity of clusters)",
        f"- Completeness: {completeness:.4f} (Correctness of clusters)",
        "\n- Cluster to Name Mapping:",
        f"  - {cluster_to_name_map}",
        "\n- Mismatches (Single Photos Only):",
    ]
    report_lines.extend(mismatches if mismatches else ["  - None. Great job!"])
    report_lines.extend(group_photo_summary)
    report_lines.append("-----------------------------------------------------------------")

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
    - Resize Width        : {stats['resize_width']} px
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
        "eps_value": EPS_VALUE,
        "resize_width": RESIZE_WIDTH
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