import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from flask import Flask, request, jsonify, render_template
import uuid
import threading
import shutil

import core_engine
import database
import yaml

# --- Flask App Initialization ---
app = Flask(__name__)
# CHANGE: The upload folder is now inside the 'static' directory
UPLOAD_FOLDER = os.path.join("static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- NEW: Shared dictionary to track job status ---
JOBS = {}

@app.route('/')
def index():
    """Serves the main HTML page from the 'templates' folder."""
    return render_template('index.html')


# --- UPDATED: Background Job Function ---
# In web_server.py

def run_face_processing_job(job_id, image_folder, preset_name):
    """
    This function now saves a simplified, JSON-serializable result.
    """
    print(f"Job {job_id}: Starting background processing for preset '{preset_name}'...")

    try:
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        preset_settings = config['presets'][preset_name]

        def update_progress(current, total):
            JOBS[job_id]['progress'] = int((current / total) * 100)


        # Use the correct engine function based on the preset library
        if preset_settings['library'] == 'dlib':
            face_data, _, _ = core_engine.process_images_dlib(
                image_folder, preset_settings, existing_paths=set(), progress_callback=update_progress
            )
        elif preset_settings['library'] == 'deepface':
            face_data, _, _ = core_engine.process_images_deepface(
                image_folder, preset_settings, existing_paths=set(), progress_callback=update_progress
            )
        else:
            raise ValueError(f"Unknown library in preset: {preset_settings['library']}")

        if face_data:
            clustered_data, _, _ = core_engine.cluster_faces(face_data, preset_settings)

            # --- START OF FIX ---
            # Create a simplified result that is JSON-safe
            # CHANGE: Create a result with public URLs for the images
            simplified_results = []
            for face in clustered_data:
                # Construct the public URL path for the image
                public_path = os.path.join(UPLOAD_FOLDER, job_id, os.path.basename(face["image_path"])).replace("\\",
                                                                                                                "/")
                simplified_results.append({
                    "image_url": public_path,
                    "cluster_id": face["cluster_id"]
                })
            JOBS[job_id]['result'] = simplified_results
        else:
            JOBS[job_id]['result'] = "No faces found."

        JOBS[job_id]['status'] = 'complete'
        print(f"Job {job_id}: Processing complete.")

    except Exception as e:
        JOBS[job_id]['status'] = 'failed'
        JOBS[job_id]['error'] = str(e)
        print(f"Job {job_id}: Processing failed. Error: {e}")

# --- UPDATED: Main /api/process Endpoint ---
@app.route('/api/process', methods=['POST'])
def process_images_endpoint():
    uploaded_files = request.files.getlist('photos')
    if not uploaded_files:
        return jsonify({"error": "No photos provided"}), 400

    job_id = str(uuid.uuid4())
    job_folder = os.path.join(UPLOAD_FOLDER, job_id)
    os.makedirs(job_folder)

    for file in uploaded_files:
        file.save(os.path.join(job_folder, file.filename))

    # NEW: Initialize the job status in the JOBS dictionary
    JOBS[job_id] = {'status': 'processing', 'progress': 0, 'result': None}

    # Start the background thread
    thread = threading.Thread(target=run_face_processing_job, args=(job_id, job_folder, "dlib_hog"))
    thread.start()

    return jsonify({"message": "Processing started.", "job_id": job_id})


# --- NEW: Status Endpoint ---
@app.route('/api/status/<job_id>')
def get_status(job_id):
    """Returns the status and result of a specific job."""
    job = JOBS.get(job_id)
    if job is None:
        return jsonify({"error": "Job not found"}), 404

    # For now, we only have 'processing' or 'complete'/'failed'
    # In a more advanced version, the background job could update a 'progress' percentage
    return jsonify(job)


# --- Main Execution ---
if __name__ == '__main__':
    app.run(debug=True, port=5001)