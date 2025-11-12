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
from werkzeug.utils import secure_filename
import hashlib  # <-- NEW: For creating unique filenames
import pathlib  # <-- NEW: For easier path/extension handling

# --- Flask App Initialization ---
app = Flask(__name__)

# --- NEW: Load Config and Set Up Permanent Storage ---
try:
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    GALLERY_STORAGE_PATH = config['directory_paths']['gallery_storage']
    # Ensure the gallery_storage directory exists
    os.makedirs(GALLERY_STORAGE_PATH, exist_ok=True)
    # Also ensure the temporary upload folder exists
    UPLOAD_FOLDER = os.path.join("static", "uploads")
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
except Exception as e:
    print(f"CRITICAL ERROR: Could not load 'gallery_storage' path from config.yaml. Please check your config. {e}")
    exit()

# --- NEW: Initialize Database on startup ---
try:
    print("Initializing database...")
    database.init_db()
    print("Database initialization complete.")
except Exception as e:
    print(f"CRITICAL ERROR: Could not initialize database. Is PostgreSQL running? {e}")
    exit()

# Shared dictionary to track job status
JOBS = {}


# Custom exception for cancellation
class JobCancelledException(Exception):
    pass


@app.route('/')
def index():
    """Serves the main HTML page from the 'templates' folder."""
    return render_template('index.html')


# --- UPDATED: Background Job Function (Major Rewrite) ---

def run_face_processing_job(job_id, temp_job_folder, preset_name):
    """
    Handles the new "persistent gallery" background process:
    1. Iterates through all files in the temporary job folder.
    2. Hashes each file to create a unique, permanent path.
    3. Copies the file to the permanent 'gallery_storage'.
    4. Adds the photo to the 'photos' table in the database.
    5. Calls core_engine to find/save faces for that *new* photo_id.
    6. Deletes the temporary job folder.
    """
    print(f"Job {job_id}: Starting background processing for preset '{preset_name}'...")
    try:
        if JOBS[job_id].get('status') == 'cancelling':
            raise JobCancelledException("Job cancelled before start.")

        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        preset_settings = config['presets'][preset_name]

        # --- 1. Get a list of all files to process ---
        files_to_process = []
        for root, dirs, files in os.walk(temp_job_folder):
            for file in files:
                files_to_process.append(os.path.join(root, file))

        if not files_to_process:
            JOBS[job_id].update({'status': 'complete', 'result': "No valid files found."})
            print(f"Job {job_id}: No files found to process.")
            return

        total_files = len(files_to_process)

        # --- 2. Process files one by one ---
        for i, file_path in enumerate(files_to_process):
            if JOBS[job_id].get('status') == 'cancelling':
                raise JobCancelledException(f"Job {job_id} cancelled during processing.")

            # Update progress
            progress = int(((i + 1) / total_files) * 100)
            JOBS[job_id]['progress'] = progress

            original_filename = os.path.basename(file_path)

            try:
                # --- 3. Create unique path and save to permanent gallery ---
                file_hash = hashlib.sha256(open(file_path, 'rb').read()).hexdigest()
                file_ext = pathlib.Path(file_path).suffix.lower()

                # Use first 2 chars of hash for a subfolder (e.g., gallery_storage/f1/f1d2...)
                # This prevents having 1,000,000 files in a single folder.
                hash_folder = os.path.join(GALLERY_STORAGE_PATH, file_hash[:2])
                os.makedirs(hash_folder, exist_ok=True)

                permanent_path = os.path.join(hash_folder, f"{file_hash}{file_ext}")

                # If file already exists, don't copy it again.
                if not os.path.exists(permanent_path):
                    shutil.copy(file_path, permanent_path)

                # --- 4. Add photo to database ---
                photo_id = database.add_photo(permanent_path, original_filename)

                if photo_id is None:
                    # This happens if the photo was already in the DB (UNIQUE constraint)
                    # We can just skip processing it.
                    print(f"Job {job_id}: Skipping already processed photo: {original_filename}")
                    continue

                # --- 5. Call Core Engine to process this *single* photo ---
                # We will write this function in the next step!
                core_engine.process_single_image(
                    permanent_path,
                    photo_id,
                    preset_name,
                    preset_settings
                )

            except Exception as e:
                print(f"Job {job_id}: FAILED to process file {original_filename}: {e}")
                # Don't stop the whole batch, just skip this file.

        # --- 6. Job Complete ---
        JOBS[job_id]['status'] = 'complete'
        # NEW: Results are no longer stored in memory, they are in the database.
        # We just tell the frontend the job is done.
        JOBS[job_id]['result'] = "Processing complete. Your gallery is updated."
        print(f"Job {job_id}: Processing complete.")

    except JobCancelledException as e:
        JOBS[job_id]['status'] = 'cancelled'
        print(str(e))
    except Exception as e:
        JOBS[job_id]['status'] = 'failed'
        JOBS[job_id]['error'] = str(e)
        print(f"Job {job_id}: Processing FAILED. Error: {e}")
    finally:
        # --- 7. CRITICAL: Clean up the temporary job folder ---
        try:
            if os.path.exists(temp_job_folder):
                shutil.rmtree(temp_job_folder)
                print(f"Job {job_id}: Temporary folder {temp_job_folder} cleaned up.")
        except Exception as e:
            print(f"Job {job_id}: ERROR cleaning up temp folder {temp_job_folder}: {e}")


@app.route('/api/name_cluster', methods=['POST'])
def name_cluster():
    data = request.get_json()
    face_ids = data.get('face_ids')
    name = data.get('name')

    if not face_ids or not name:
        return jsonify({"error": "Missing face_ids or name"}), 400

    # 1. Find or create the person in the database
    person_id = database.get_or_create_person(name)
    if person_id is None:
        return jsonify({"error": "Failed to get or create person entry."}), 500

    # 2. Link faces directly
    database.link_faces_to_person(face_ids, person_id)

    return jsonify({"success": True, "message": f"Linked {len(face_ids)} faces to '{name}'."})


# --- UPDATED: Main /api/process Endpoint ---
@app.route('/api/process', methods=['POST'])
def process_images_endpoint():
    uploaded_files = request.files.getlist('photos')
    if not uploaded_files:
        return jsonify({"error": "No photos provided"}), 400

    preset_name = request.form.get('preset', 'dlib_hog')

    job_id = str(uuid.uuid4())
    temp_job_folder = os.path.join(UPLOAD_FOLDER, job_id)
    os.makedirs(temp_job_folder)

    # --- Logic to handle Zips and Images (Unchanged from before) ---
    processed_files_count = 0
    for file in uploaded_files:
        if not file or not file.filename:
            continue

        filename = secure_filename(file.filename)

        try:
            if filename.lower().endswith('.zip'):
                print(f"Job {job_id}: Extracting images from {filename}...")
                file_stream = io.BytesIO(file.read())
                with zipfile.ZipFile(file_stream, 'r') as zf:
                    for member in zf.infolist():
                        if member.is_dir():
                            continue

                        member_name = os.path.basename(member.filename)
                        if not member_name:
                            continue

                        if member_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                            extract_path = os.path.join(temp_job_folder, member_name)
                            with zf.open(member) as source_file:
                                with open(extract_path, 'wb') as target_file:
                                    shutil.copyfileobj(source_file, target_file)
                            processed_files_count += 1

            elif filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                file.save(os.path.join(temp_job_folder, filename))
                processed_files_count += 1

        except zipfile.BadZipFile:
            print(f"Job {job_id}: Ignoring corrupt zip file: {filename}")
        except Exception as e:
            print(f"Job {job_id}: Error processing file {filename}: {e}")
    # --- End of Zip/Image handling ---

    if processed_files_count == 0:
        shutil.rmtree(temp_job_folder)
        return jsonify({"error": "No valid image files (png, jpg, jpeg) or .zip files were provided."}), 400

    # Initialize the job status in the JOBS dictionary
    JOBS[job_id] = {'status': 'processing', 'progress': 0, 'result': None}

    # Start the background thread
    thread = threading.Thread(target=run_face_processing_job, args=(job_id, temp_job_folder, preset_name))
    thread.start()

    return jsonify({"message": "Processing started.", "job_id": job_id})


@app.route('/api/status/<job_id>')
def get_status(job_id):
    """Returns the status and result of a specific job."""
    job = JOBS.get(job_id)
    if job is None:
        return jsonify({"error": "Job not found"}), 404
    return jsonify(job)


@app.route('/api/cancel/<job_id>', methods=['POST'])
def cancel_job(job_id):
    """Requests cancellation of a running job."""
    job = JOBS.get(job_id)
    if job and job['status'] == 'processing':
        JOBS[job_id]['status'] = 'cancelling'
        print(f"Job {job_id}: Cancellation requested.")
        return jsonify({"success": True, "message": "Cancellation requested."})
    return jsonify({"error": "Job not found or already completed."}), 404


@app.route('/api/download/<job_id>')
def download_zip(job_id):
    """
    Generates and sends a zip file of the sorted results.
    --- THIS FUNCTION MUST BE UPDATED ---
    --- For now, it will NOT work, as results are not in memory. ---
    """
    job = JOBS.get(job_id)
    if not job or job['status'] != 'complete':
        return jsonify({"error": "Job not found or not complete"}), 404

    # TODO: This logic is now broken because results are not in memory.
    # We will fix this later by adding a "download_gallery" function
    # that reads from the database.
    return jsonify({"error": "Download function is not yet updated for new architecture."}), 501


# --- Main Execution ---
if __name__ == '__main__':
    # Note: We run on port 5001 now
    app.run(debug=True, port=5001)