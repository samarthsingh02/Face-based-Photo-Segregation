import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from flask import Flask, request, jsonify, render_template, send_file
import uuid
import threading
import shutil
import io
import zipfile

import core_engine
import database
import yaml

# --- Flask App Initialization ---
app = Flask(__name__)
database.init_db()

# The upload folder is now inside the 'static' directory
UPLOAD_FOLDER = os.path.join("static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Shared dictionary to track job status ---
JOBS = {}

# Custom exception for cancellation ---
class JobCancelledException(Exception):
    pass

@app.route('/')
def index():
    """Serves the main HTML page from the 'templates' folder."""
    return render_template('index.html')


# --- UPDATED: Background Job Function ---

def run_face_processing_job(job_id, image_folder, preset_name):
    """
    Handles the full background process:
    1. Detects faces in new images.
    2. Recognizes known people by comparing against the database.
    3. Clusters any remaining, unidentified faces.
    4. Saves all new faces to the database.
    5. Formats the results for the frontend.
    """
    print(f"Job {job_id}: Starting background processing for preset '{preset_name}'...")
    try:
        # Check for immediate cancellation ---
        if JOBS[job_id].get('status') == 'cancelling':
            JOBS[job_id]['status'] = 'cancelled'
            print(f"Job {job_id}: Cancelled before starting.")
            return

        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        preset_settings = config['presets'][preset_name]
        threshold = preset_settings['clustering']['eps']

        def update_progress(current, total, stage=""):
            # Check for cancellation signal
            if JOBS[job_id].get('status') == 'cancelling':
                raise JobCancelledException(f"Job {job_id} cancelled during {stage}.")

            # Progress is now split into two stages: detection (80%) and clustering (20%)
            base_progress = 80 if stage == "detection" else 100
            stage_progress = int((current / total) * base_progress) if total > 0 else base_progress
            JOBS[job_id]['progress'] = stage_progress

        # --- STAGE 1: Detect all faces in the uploaded images ---
        library = preset_settings['library']
        if library == 'dlib':
            newly_detected_faces, _, _ = core_engine.process_images_dlib(
                image_folder, preset_settings, existing_paths=set(), progress_callback=lambda c, t: update_progress(c, t, "detection")
            )
        elif library == 'deepface':
            newly_detected_faces, _, _ = core_engine.process_images_deepface(
                image_folder, preset_settings, existing_paths=set(), progress_callback=lambda c, t: update_progress(c, t, "detection")
            )
        else:
            raise ValueError(f"Unknown library in preset: {library}")

        if not newly_detected_faces:
            JOBS[job_id].update({'status': 'complete', 'result': "No new faces found."})
            print(f"Job {job_id}: No faces found.")
            return

        # --- STAGE 2: Recognize, Cluster, and Save ---
        JOBS[job_id]['progress'] = 85 # Update progress bar for next stage

        # 2a. Get known faces from the database to compare against
        named_faces_from_db = database.get_named_faces_by_model(preset_name)

        # 2b. Separate new faces into recognized and unrecognized groups
        identified_faces, unidentified_faces = core_engine.recognize_and_classify(
            newly_detected_faces, named_faces_from_db, threshold
        )

        # 2c. Cluster only the unidentified faces
        clustered_unidentified = []
        if unidentified_faces:
            # Check for cancellation before clustering ---
            if JOBS[job_id].get('status') == 'cancelling':
                raise JobCancelledException(f"Job {job_id} cancelled before clustering.")
            clustered_unidentified, _, _ = core_engine.cluster_faces(unidentified_faces, preset_settings)
        JOBS[job_id]['progress'] = 95 # Update progress before saving

        # 2d. Save ALL new faces to the database and get their new IDs
        all_new_faces = identified_faces + clustered_unidentified
        inserted_ids = database.add_faces_and_get_ids(all_new_faces, preset_name)
        for i, face in enumerate(all_new_faces):
            face['id'] = inserted_ids[i] # Assign the new database ID

        # --- STAGE 3: Prepare results for the frontend ---
        simplified_results = []
        for face in all_new_faces:
            public_path = os.path.join("static", "uploads", job_id, os.path.basename(face["image_path"])).replace("\\",                                                                                               "/")
            result_obj = {
                "face_id": face.get("id"),
                "image_url": public_path,
                # Use the person's name for recognized faces, or the cluster_id for new ones
                "cluster_id": face.get("person_name") or face.get("cluster_id"),
                "is_named": "person_name" in face # Flag for the frontend
            }
            simplified_results.append(result_obj)

        JOBS[job_id]['result'] = simplified_results
        JOBS[job_id]['status'] = 'complete'
        print(f"Job {job_id}: Processing complete.")


    # --- NEW: Catch the cancellation exception ---
    except JobCancelledException as e:
        JOBS[job_id]['status'] = 'cancelled'
        print(str(e))

    except Exception as e:
        JOBS[job_id]['status'] = 'failed'
        JOBS[job_id]['error'] = str(e)
        print(f"Job {job_id}: Processing failed. Error: {e}")

@app.route('/api/name_cluster', methods=['POST'])
def name_cluster():
    data = request.get_json()
    face_ids = data.get('face_ids')
    name = data.get('name')

    if not face_ids or not name:
        return jsonify({"error": "Missing face_ids or name"}), 400

    # 1. Find or create the person in the database
    person_id = database.get_or_create_person(name)

    # 2. Link faces directly
    database.link_faces_to_person(face_ids, person_id)

    return jsonify({"success": True, "message": f"Linked {len(face_ids)} faces to '{name}'."})


# --- Main /api/process Endpoint ---
@app.route('/api/process', methods=['POST'])
def process_images_endpoint():
    uploaded_files = request.files.getlist('photos')
    if not uploaded_files:
        return jsonify({"error": "No photos provided"}), 400

    # ---Get the preset name from the form data ---
    # Default to 'dlib_hog' if for some reason it's not provided
    preset_name = request.form.get('preset', 'dlib_hog')

    job_id = str(uuid.uuid4())
    job_folder = os.path.join(UPLOAD_FOLDER, job_id)
    os.makedirs(job_folder)

    for file in uploaded_files:
        file.save(os.path.join(job_folder, file.filename))

    # Initialize the job status in the JOBS dictionary
    JOBS[job_id] = {'status': 'processing', 'progress': 0, 'result': None}

    # Start the background thread
    # Pass the selected preset_name to the background job ---
    thread = threading.Thread(target=run_face_processing_job, args=(job_id, job_folder, preset_name))
    thread.start()

    return jsonify({"message": "Processing started.", "job_id": job_id})


# --- Status Endpoint ---
@app.route('/api/status/<job_id>')
def get_status(job_id):
    """Returns the status and result of a specific job."""
    job = JOBS.get(job_id)
    if job is None:
        return jsonify({"error": "Job not found"}), 404

    # For now, we only have 'processing' or 'complete'/'failed'
    # In a more advanced version, the background job could update a 'progress' percentage
    return jsonify(job)

# --- Cancel Endpoint ---
@app.route('/api/cancel/<job_id>', methods=['POST'])
def cancel_job(job_id):
    """Requests cancellation of a running job."""
    job = JOBS.get(job_id)
    if job and job['status'] == 'processing':
        JOBS[job_id]['status'] = 'cancelling'
        print(f"Job {job_id}: Cancellation requested.")
        return jsonify({"success": True, "message": "Cancellation requested."})
    return jsonify({"error": "Job not found or already completed."}), 404

# --- NEW: Download Zip Endpoint ---
@app.route('/api/download/<job_id>')
def download_zip(job_id):
    """Generates and sends a zip file of the sorted results."""
    job = JOBS.get(job_id)
    if not job or job['status'] != 'complete':
        return jsonify({"error": "Job not found or not complete"}), 404

    results = job.get('result', [])
    if not results:
        return jsonify({"error": "No results to download"}), 404

    # Create an in-memory zip file
    memory_file = io.BytesIO()
    with zipfile.ZipFile(memory_file, 'w', zipfile.ZIP_DEFLATED) as zf:
        # Keep track of files added to avoid duplicates from multiple faces in one image
        added_files_in_zip = set()

        for face in results:
            try:
                # Replicate frontend logic for folder names
                cluster_id = face.get("cluster_id")
                is_named = face.get("is_named")

                if is_named:
                    folder_name = str(cluster_id)
                elif str(cluster_id) == '-1':
                    folder_name = 'Unknowns'
                else:
                    folder_name = f"Person {int(cluster_id) + 1}"

                # The image_url is the path from the app's root (e.g., "static/uploads/...")
                source_file_path = face.get("image_url")

                # Get just the filename (e.g., "samarth_1.jpg")
                file_name = os.path.basename(source_file_path)

                # This is the full path *inside* the zip file (e.g., "Samarth/samarth_1.jpg")
                arcname = os.path.join(folder_name, file_name)

                # Add the file to the zip if it hasn't been added already
                if arcname not in added_files_in_zip:
                    zf.write(source_file_path, arcname=arcname)
                    added_files_in_zip.add(arcname)

            except Exception as e:
                print(f"Error adding {face.get('image_url')} to zip: {e}")
                # Continue trying to add other files

    # Rewind the file to the beginning
    memory_file.seek(0)

    return send_file(
        memory_file,
        download_name=f'sorted_photos_{job_id}.zip',
        as_attachment=True,
        mimetype='application/zip'
    )

# --- Main Execution ---
if __name__ == '__main__':
    app.run(debug=True, port=5001)