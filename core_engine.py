import os
import shutil
import logging
import cv2
from tqdm import tqdm
from collections import Counter

import face_recognition
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics.cluster import homogeneity_score, completeness_score, v_measure_score
import database

# In core_engine.py
import database  # <-- Make sure this is imported at the top


def process_images(source_dir, resize_width, detector_model, existing_paths=set()):
    """
    Scans a directory for images, skipping any paths provided in existing_paths.
    """
    logging.info(f"Discovering new images (skipping {len(existing_paths)} already processed)...")
    new_image_paths = []
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                full_path = os.path.join(root, file)
                if full_path not in existing_paths: # <-- The key change is here
                    new_image_paths.append(full_path)

    if not new_image_paths:
        logging.warning(f"No new images found in '{source_dir}' to process.")
        return None, 0, 0

    logging.info(f"Starting processing for {len(new_image_paths)} new images...")
    all_face_data = []
    total_faces_found = 0

    for image_path in tqdm(new_image_paths, desc="Processing New Images"):
        # ... (the rest of the loop remains exactly the same as before) ...
        try:
            image = cv2.imread(image_path)
            if image is None:
                tqdm.write(f"Warning: Skipping corrupted image: {os.path.basename(image_path)}")
                continue
            (h, w) = image.shape[:2]
            if w > resize_width:
                r = float(resize_width) / w
                dim = (resize_width, int(h * r))
                image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            locations = face_recognition.face_locations(rgb_image, model=detector_model)
            total_faces_found += len(locations)
            encodings = face_recognition.face_encodings(rgb_image, locations)
            for encoding in encodings:
                all_face_data.append({'image_path': image_path, 'encoding': encoding})
        except Exception as e:
            tqdm.write(f"ERROR: Could not process {os.path.basename(image_path)}. Error: {e}")

    logging.info(f"Completed processing. Found {total_faces_found} new faces in {len(new_image_paths)} images.")
    return all_face_data, len(new_image_paths), total_faces_found


def cluster_faces(all_face_data, eps, min_samples):
    """
    Groups face encodings into clusters using the DBSCAN algorithm.

    Args:
        all_face_data (list): The master list of face data from process_images.
        eps (float): The maximum distance between two samples for one to be
                     considered as in the neighborhood of the other.
        min_samples (int): The number of samples in a neighborhood for a point
                           to be considered as a core point.

    Returns:
        tuple: A tuple containing (updated_face_data, num_people, num_unknowns).
               Returns (None, 0, 0) if no data is provided.
    """
    # 1. Early exit if there are no faces to cluster.
    if not all_face_data:
        logging.warning("No faces were detected to cluster.")
        return None, 0, 0

    logging.info("Starting face clustering...")

    # 2. Data Preparation: Extract all 128-d face encodings into a NumPy array.
    # This is the format required by scikit-learn's clustering algorithms.
    encodings = np.array([data['encoding'] for data in all_face_data])

    # 3. Initialize and run the DBSCAN clustering algorithm.
    # `eps` and `min_samples` are the two key parameters that control the clustering.
    logging.info(f"Running DBSCAN with eps={eps} and min_samples={min_samples}...")
    clt = DBSCAN(metric="euclidean", eps=eps, min_samples=min_samples)
    # The .fit() method performs the actual grouping on the encoding data.
    clt.fit(encodings)

    # 4. Calculate summary statistics from the clustering results.
    # The algorithm returns a list of labels (clt.labels_).
    # Each unique label (e.g., 0, 1, 2) represents a unique person.
    # The special label '-1' is assigned by DBSCAN to noisy samples (outliers/unknowns).
    unique_labels = set(clt.labels_)
    num_people = len(unique_labels) - (1 if -1 in unique_labels else 0)
    num_unknowns = list(clt.labels_).count(-1)
    logging.info(f"Found {num_people} unique people (clusters) and {num_unknowns} unknown faces.")

    # 5. Map Results: Add the assigned cluster ID back to each face's data dictionary.
    # This links each individual face to the person-group it belongs to.
    for i, label in enumerate(clt.labels_):
        all_face_data[i]['cluster_id'] = int(label)

    # 6. Return the enriched data and the calculated statistics.
    return all_face_data, num_people, num_unknowns


def organize_files(all_face_data, output_dir, folder_prefix, unknowns_folder):
    """
    Copies the original image files into new, organized folders based on the
    assigned cluster ID for each face.

    Args:
        all_face_data (list): The master list of face data, now with cluster IDs.
        output_dir (str): The root directory where sorted folders will be created.
        folder_prefix (str): The prefix for the named person folders (e.g., "Person_").
        unknowns_folder (str): The name for the folder containing unknown faces.
    """
    # 1. Early exit if there is no data to process.
    if not all_face_data:
        logging.info("No data to organize.")
        return

    logging.info("Organizing files into folders...")

    # 2. Optimization: Keep track of which files have already been copied to which folders.
    # This prevents copying the same file multiple times if it contains, for example,
    # two faces that were both correctly identified as "Person_1".
    copied_files = set()

    # 3. Main Loop: Iterate through every face that was detected.
    for data in all_face_data:
        cluster_id = data['cluster_id']
        image_path = data['image_path']

        # 4. Determine Destination: Decide the folder name based on the cluster ID.
        # If the ID is -1, it's an "Unknown"; otherwise, construct the person's folder name.
        if cluster_id == -1:
            folder_name = unknowns_folder
        else:
            folder_name = f"{folder_prefix}{cluster_id + 1}"

        destination_folder = os.path.join(output_dir, folder_name)

        # 5. Create Directory: Ensure the destination folder exists.
        # `exist_ok=True` prevents an error if the folder has already been created.
        os.makedirs(destination_folder, exist_ok=True)

        # 6. Copy File (if necessary): Check if this exact file has already been
        # copied to this exact folder before performing the copy operation.
        copy_key = (image_path, destination_folder)
        if copy_key not in copied_files:
            shutil.copy(image_path, destination_folder)
            copied_files.add(copy_key)

    logging.info("File organization complete!")


def calculate_clustering_metrics(all_face_data):
    """
    Calculates a comprehensive clustering report. It computes standard metrics
    (V-Measure, etc.) using only single-person photos for accuracy and provides a
    separate, informational summary for faces found in group photos.

    Args:
        all_face_data (list): The master list of face data with cluster IDs.

    Returns:
        str: A formatted string containing the full evaluation report.
    """
    if not all_face_data:
        logging.warning("Cannot calculate metrics: No face data provided.")
        return "\n    No face data to calculate metrics."

    # 1. Data Separation: Divide the data into two lists.
    # This is crucial for a clean evaluation. We will only score the algorithm
    # on its performance with single-person photos.
    single_person_data = []
    group_photo_data = []
    for data in all_face_data:
        # We assume group photos are in a folder starting with '_' (e.g., "_group_photos").
        folder_name = os.path.basename(os.path.dirname(data['image_path']))
        if folder_name.startswith('_'):
            group_photo_data.append(data)
        else:
            single_person_data.append(data)

    # 2. Ground Truth Preparation (for single photos only).
    # We can only calculate metrics if we have single-person photos to evaluate against.
    if not single_person_data:
        return "\n    No single-person photos found to calculate meaningful metrics."

    # `true_labels` are the correct answers (the folder names).
    true_labels = [os.path.basename(os.path.dirname(data['image_path'])) for data in single_person_data]
    # `predicted_labels` are the algorithm's guesses (the cluster IDs).
    predicted_labels = [data['cluster_id'] for data in single_person_data]

    # Metrics are only useful if there are at least two different people to compare.
    if len(set(true_labels)) < 2:
        return "\n    Metrics not calculated: Need at least two source folders of single people."

    # 3. Calculate Core Metrics using scikit-learn.
    homogeneity = homogeneity_score(true_labels, predicted_labels)
    completeness = completeness_score(true_labels, predicted_labels)
    v_measure = v_measure_score(true_labels, predicted_labels)

    # 4. Create Cluster-to-Name Mapping.
    # This uses a "voting" system to assign the most common true name to each cluster ID.
    cluster_to_name_map = {}
    for cluster_id in set(predicted_labels):
        if cluster_id == -1: continue
        names_in_cluster = [true_labels[i] for i, pred_label in enumerate(predicted_labels) if pred_label == cluster_id]
        if names_in_cluster:
            # Find the most frequent name in this cluster and declare it the "winner".
            winner_name = Counter(names_in_cluster).most_common(1)[0][0]
            cluster_to_name_map[cluster_id] = winner_name

    # 5. Identify Specific Mismatches (for single photos only).
    # Compare the true name to the predicted name for each face.
    mismatches = []
    for i, data in enumerate(single_person_data):
        true_name = true_labels[i]
        cluster_id = predicted_labels[i]
        predicted_name = cluster_to_name_map.get(cluster_id, "Unknown")
        if predicted_name != true_name:
            filename = os.path.basename(data['image_path'])
            mismatches.append(f"  - FAILED: '{filename}' (True: {true_name}) was predicted as '{predicted_name}'")

    # 6. Summarize Group Photo Results (informational only).
    # This shows how faces from group photos were classified, without marking them as "failed".
    group_photo_summary = []
    if group_photo_data:
        group_predictions = [cluster_to_name_map.get(data['cluster_id'], "Unknown") for data in group_photo_data]
        group_counts = Counter(group_predictions)
        group_photo_summary.append("\n- Group Photo Analysis:")
        for name, count in group_counts.items():
            group_photo_summary.append(f"  - Identified {count} face(s) as '{name}'")

    # 7. Assemble the Final Report String.
    report_lines = [
        "\n",
        "----------------- CLUSTERING METRICS (Single Photos Only) -----------------",
        f"- V-Measure   : {v_measure:.4f} (The overall balanced score)",
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