import time
import yaml
import logging
import os
import shutil

# The only major import should be your own engine!
import core_engine
import database

# --- Configuration Loading ---
# We load the config at the global level so all functions can see the constants.
try:
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Get active preset name and load its settings
    active_model_name = config['active_model']
    preset_settings = config['presets'][active_model_name]

    DETECTOR_MODEL = preset_settings['detector']
    EPS_VALUE = preset_settings['eps']
    RESIZE_WIDTH = preset_settings['resize_width']

    # Load general settings
    SOURCE_DIR = config['directory_paths']['source']
    OUTPUT_DIR = config['directory_paths']['output']
    LOG_FILE = config['directory_paths']['log_file']
    MIN_SAMPLES = config['clustering_settings']['min_samples']
    FOLDER_PREFIX = config['output_settings']['folder_prefix']
    UNKNOWNS_FOLDER = config['output_settings']['unknowns_folder']

except Exception as e:
    print(f"CRITICAL ERROR loading config.yaml. Please check the file. Error: {e}")
    exit()


# --- Helper Functions for this Runner Script ---

def init_logging(log_filepath):
    """Sets up the logging configuration using the path from config."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filepath, mode='a'),
            logging.StreamHandler()
        ]
    )


def log_run_summary(stats):
    """Formats and logs the final run summary."""
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


# --- Main Application Logic ---

def main():
    """Main function to orchestrate the face sorting process."""
    # 1. Initialization
    init_logging(LOG_FILE)
    database.init_db()
    start_time = time.time()

    run_stats = {
        "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
        "detector_model": DETECTOR_MODEL,
        "eps_value": EPS_VALUE,
        "resize_width": RESIZE_WIDTH
    }

    logging.info(f"-- Starting New Run --")
    logging.info(f"Using preset: '{active_model_name}'")

    # 2. Load existing data & process new data
    existing_faces = database.get_faces_by_model(active_model_name)
    processed_paths = {face['image_path'] for face in existing_faces}

    # This call now correctly gets only the NEW images
    new_faces, img_count, face_count = core_engine.process_images(
        SOURCE_DIR, RESIZE_WIDTH, DETECTOR_MODEL, existing_paths=processed_paths
    )

    # 3. Combine old data from the DB with newly processed data
    all_face_data = existing_faces + (new_faces or [])

    # 4. Save only the newly processed faces back to the database
    if new_faces:
        database.add_faces(new_faces, active_model_name)

    # 5. Update stats with ONLY the work done in this specific run
    run_stats["images_processed"] = img_count
    run_stats["faces_detected"] = face_count

    # 6. Cluster, sort, and report on the FULL (combined) dataset
    if all_face_data:
        # Clear output directory just before organizing files
        if os.path.exists(OUTPUT_DIR):
            shutil.rmtree(OUTPUT_DIR)
        os.makedirs(OUTPUT_DIR)

        clustered_data, people_count, unknown_count = core_engine.cluster_faces(
            all_face_data, EPS_VALUE, MIN_SAMPLES
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

    # 7. Finalize and log the summary
    end_time = time.time()
    run_stats["execution_time"] = end_time - start_time
    log_run_summary(run_stats)


# --- Script Entry Point ---
if __name__ == "__main__":
    main()