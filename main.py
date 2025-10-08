import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import time
import yaml
import logging
import shutil
import core_engine
import database

# --- Load Configuration ---
try:
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    active_model_name = config['active_model']
    preset_settings = config['presets'][active_model_name]
    SOURCE_DIR = config['directory_paths']['source']
    OUTPUT_DIR = config['directory_paths']['output']
    LOG_FILE = config['directory_paths']['log_file']
    FOLDER_PREFIX = config['output_settings']['folder_prefix']
    UNKNOWNS_FOLDER = config['output_settings']['unknowns_folder']
except Exception as e:
    print(f"CRITICAL ERROR loading config.yaml: {e}")
    exit()

# --- Helper Functions for this Runner Script ---

def init_logging(log_filepath):
    """Sets up the logging to write to a file and the console."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filepath, mode='a'),
            logging.StreamHandler()
        ]
    )


def log_run_summary(stats):
    """Formats and logs the final summary of the run."""

    resize_line = f"- Resize Width        : {stats['resize_width']} px" if stats['resize_width'] != 'N/A' else "- Resize Width        : N/A (Internal to Model)"

    summary = f"""
    -------------------- RUN SUMMARY --------------------
    - Timestamp           : {stats['timestamp']}
    - Preset Name         : {stats['preset_name']}
    - Detector Model      : {stats['detector_model']}
    - Embedding Model     : {stats['embedding_model']}
    - DBSCAN eps          : {stats['eps_value']}
    {resize_line} 
    - Images Processed    : {stats['images_processed']} (new)
    - Faces Detected      : {stats['faces_detected']} (new)
    - People Found        : {stats['people_found']} (total)
    - Unknowns / Outliers : {stats['unknown_faces']} face(s) (total)
    - Total Execution Time: {stats['execution_time']:.2f} seconds
    -----------------------------------------------------
    """
    summary += stats.get("clustering_report", "")
    logging.info(summary)


# --- Main Application Logic ---

def main():
    """Main function to orchestrate the entire face sorting process."""
    init_logging(LOG_FILE)
    database.init_db()
    start_time = time.time()

    run_stats = {
        "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
        "preset_name": active_model_name,
        "detector_model": preset_settings.get('detector') or preset_settings.get('model'),
        "embedding_model": preset_settings.get('embedding_model', 'dlib'),
        "eps_value": preset_settings['clustering']['eps'],
        "resize_width": preset_settings.get('resize_width', 'N/A')
    }

    logging.info(f"-- Starting New Run --")
    logging.info(f"Using preset: '{active_model_name}'")

    # 1. Load existing data for the active model
    existing_faces = database.get_faces_by_model(active_model_name)
    processed_paths = {face['image_path'] for face in existing_faces}

    # 2. Process only new images
    # --- NEW: Call the correct function based on the library in the preset ---
    # --- Call the correct function based on the library in the preset ---
    if preset_settings['library'] == 'dlib':
        new_faces, img_count, face_count = core_engine.process_images_dlib(
            SOURCE_DIR, preset_settings, existing_paths=processed_paths
        )
    elif preset_settings['library'] == 'deepface':
        new_faces, img_count, face_count = core_engine.process_images_deepface(
            SOURCE_DIR, preset_settings, existing_paths=processed_paths
        )
    else:
        new_faces, img_count, face_count = None, 0, 0
        logging.error(f"Unknown library specified in preset: {preset_settings['library']}")

    # 3. Combine old and new data for a complete view
    all_face_data = existing_faces + (new_faces or [])

    # 4. Save only the newly processed faces back to the database
    if new_faces:
        database.add_faces(new_faces, active_model_name)

    # Update stats with ONLY the work done in this specific run
    run_stats["images_processed"] = img_count
    run_stats["faces_detected"] = face_count

    # Cluster, sort, and report on the FULL (combined) dataset
    if all_face_data:
        # Clear output directory before organizing files
        if os.path.exists(OUTPUT_DIR):
            shutil.rmtree(OUTPUT_DIR)
        os.makedirs(OUTPUT_DIR)

        clustered_data, people_count, unknown_count = core_engine.cluster_faces(
            all_face_data, preset_settings
        )
        run_stats["people_found"] = people_count
        run_stats["unknown_faces"] = unknown_count

        core_engine.organize_files(
            clustered_data, OUTPUT_DIR, FOLDER_PREFIX, UNKNOWNS_FOLDER
        )
        run_stats["clustering_report"] = core_engine.calculate_clustering_metrics(clustered_data)
    else:
        run_stats["people_found"] = 0
        run_stats["unknown_faces"] = 0
        run_stats["clustering_report"] = "No faces found in source or database."

    # Finalize and log the summary
    end_time = time.time()
    run_stats["execution_time"] = end_time - start_time
    log_run_summary(run_stats)


# --- Script Entry Point ---
if __name__ == "__main__":
    main()