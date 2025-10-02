import argparse
import os
import shutil
import time
import logging
import json
import cv2



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
parser.add_argument("--eps", type=float, default=0.45, help="DBSCAN epsilon value.")
args = parser.parse_args()

SOURCE_DIR = "source_images"
OUTPUT_DIR = "output_sorted"


def process_images():
    """
    Scans the source directory, resizes large images, finds faces, and extracts their encodings.
    """
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
            # Load the image with OpenCV to check its size
            image = cv2.imread(image_path)
            (h, w) = image.shape[:2]

            # Resize if the image is very large
            if w > 1000:
                r = 1000.0 / w
                dim = (1000, int(h * r))
                image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
                logging.info(f"  Resized large image to {dim[0]}x{dim[1]}")

            # Convert from BGR (OpenCV's format) to RGB (face_recognition's format)
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Face Detection (now use the resized rgb_image)
            locations = face_recognition.face_locations(rgb_image, model=args.detector)
            logging.info(f"  Found {len(locations)} face(s) using '{args.detector}' model.")

            # --- FIX: Update face count using the correct locations ---
            total_faces_found += len(locations)

            # Feature Extraction (also use the resized rgb_image)
            encodings = face_recognition.face_encodings(rgb_image, locations)

            # --- FIX: Append the encodings from the RESIZED image ---
            for encoding in encodings:
                all_face_data.append({'image_path': image_path, 'encoding': encoding})

            # --- FIX: REMOVED the entire block that re-loaded and re-processed the original image ---

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


def generate_accuracy_report(clustered_data):
    """
    Compares clustering results to a ground truth file and generates a report.
    """
    ground_truth_path = "ground_truth.json"
    if not os.path.exists(ground_truth_path):
        logging.warning("ground_truth.json not found. Skipping accuracy report.")
        return "Ground truth file not found."

    with open(ground_truth_path, 'r') as f:
        ground_truth = json.load(f)

    # First, let's figure out which cluster ID corresponds to which person's name.
    # We'll use a "voting" system. The name that appears most in a cluster "wins" it.
    cluster_to_name_map = {}
    clusters = {data['cluster_id'] for data in clustered_data if data['cluster_id'] != -1}

    for cluster_id in clusters:
        votes = {}
        # Get all images belonging to this cluster
        images_in_cluster = [data['image_path'] for data in clustered_data if data['cluster_id'] == cluster_id]

        for image_path in images_in_cluster:
            filename = os.path.basename(image_path)
            true_names = ground_truth.get(filename, [])
            for name in true_names:
                votes[name] = votes.get(name, 0) + 1

        if votes:
            winner_name = max(votes, key=votes.get)
            cluster_to_name_map[cluster_id] = winner_name

    # Now, generate the report
    report = ["\n", "---------------- ACCURACY REPORT -----------------"]
    correct_classifications = 0
    incorrect_classifications = []

    for data in clustered_data:
        filename = os.path.basename(data['image_path'])
        cluster_id = data['cluster_id']

        predicted_name = cluster_to_name_map.get(cluster_id, "Unknown")
        true_names = ground_truth.get(filename, ["Unknown"])

        if predicted_name in true_names:
            correct_classifications += 1
        else:
            incorrect_classifications.append(
                f"  - FAILED: '{filename}' was predicted as '{predicted_name}' but should contain {true_names}"
            )

    total_faces = len(clustered_data)
    accuracy = (correct_classifications / total_faces) * 100 if total_faces > 0 else 0

    report.append(f"- Cluster to Name Mapping: {cluster_to_name_map}")
    report.append(f"- Accuracy: {correct_classifications}/{total_faces} ({accuracy:.2f}%) faces correctly classified.")
    report.append("- Mismatches:")
    if incorrect_classifications:
        report.extend(incorrect_classifications)
    else:
        report.append("  - None! Great job!")
    report.append("--------------------------------------------------")

    return "\n".join(report)

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
    summary += stats.get("accuracy_report", "") # <-- ADD THIS LINE
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
        run_stats["accuracy_report"] = generate_accuracy_report(clustered_data)
    else:
        run_stats["people_found"] = 0
        run_stats["unknown_faces"] = 0
        run_stats["accuracy_report"] = "No faces processed."


    end_time = time.time()
    run_stats["execution_time"] = end_time - start_time

    log_run_summary(run_stats)