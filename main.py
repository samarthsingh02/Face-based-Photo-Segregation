import argparse
import os
import shutil
import time
import logging  # Import the logging library

import face_recognition
import numpy as np
from sklearn.cluster import DBSCAN


# --- Logger Configuration ---
def init_logging():
    """Sets up the logging configuration to log to both a file and the console."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("run_log.txt"),  # Log to a file
            logging.StreamHandler()  # Also print to the console
        ]
    )


# --- Phase 0: Setup & Configuration ---
# ... (argparse and directory setup remains the same) ...
parser = argparse.ArgumentParser(description="Segregate photos based on faces.")
parser.add_argument("--detector", type=str, default="hog", choices=["hog", "cnn"], help="Face detection model to use.")
parser.add_argument("--eps", type=float, default=0.5, help="DBSCAN epsilon value.")
args = parser.parse_args()

SOURCE_DIR = "source_images"
OUTPUT_DIR = "output_sorted"


# --- Phase 1: Data Ingestion & Feature Extraction ---
def process_images():
    logging.info("Starting image processing...")
    all_face_data = []
    image_paths = [os.path.join(SOURCE_DIR, f) for f in os.listdir(SOURCE_DIR) if
                   f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    if not image_paths:
        logging.warning("No images found in the source_images directory.")
        return None, 0, 0

    total_faces_found = 0
    for i, image_path in enumerate(image_paths):
        logging.info(f"Processing image {i + 1}/{len(image_paths)}: {os.path.basename(image_path)}")
        try:
            image = face_recognition.load_image_file(image_path)
            locations = face_recognition.face_locations(image, model=args.detector)
            logging.info(f"  Found {len(locations)} face(s) using '{args.detector}' model.")
            total_faces_found += len(locations)
            encodings = face_recognition.face_encodings(image, locations)
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

    clt = DBSCAN(metric="euclidean", eps=args.eps, min_samples=2)
    clt.fit(encodings)

    # Calculate cluster statistics
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
            folder_name = "Unknowns"
        else:
            folder_name = f"Person_{cluster_id + 1}"
        destination_folder = os.path.join(OUTPUT_DIR, folder_name)
        os.makedirs(destination_folder, exist_ok=True)
        copy_key = (image_path, destination_folder)
        if copy_key not in copied_files:
            shutil.copy(image_path, destination_folder)
            copied_files.add(copy_key)
    logging.info("File organization complete!")


# --- Logging Summary ---
def log_run_summary(stats):
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
    logging.info(summary)


# --- Main Execution ---
if __name__ == "__main__":
    init_logging()
    start_time = time.time()

    run_stats = {
        "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
        "detector_model": args.detector,
        "eps_value": args.eps
    }

    logging.info(f"-- Starting New Run --")
    logging.info(f"Parameters: detector={args.detector}, eps={args.eps}")

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
    else:
        run_stats["people_found"] = 0
        run_stats["unknown_faces"] = 0

    end_time = time.time()
    run_stats["execution_time"] = end_time - start_time

    log_run_summary(run_stats)