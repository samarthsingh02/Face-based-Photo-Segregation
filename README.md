# Face-Based Photo Segregation

## Summary
Face-Based Photo Segregation is a Python application that automatically analyzes a collection of photos, identifies unique individuals using face recognition, and sorts the photos into folders for each person. It leverages unsupervised clustering and configurable models to deliver fast and accurate results, making photo organization effortless.

## Key Features

- **Automated Clustering**: Uses DBSCAN to group faces without prior training.
- **Configurable Models**: Switch between fast `HOG` and accurate `CNN` models.
- **Performance Tuning**: Helper scripts and a config file for tuning image resize width and clustering `eps`.
- **Detailed Logging**: Generates run summaries and clustering quality reports (V-Measure).
- **Web Server**: Optional web interface for uploading and segregating images.
- **Database Support**: Stores face encodings and clustering results in SQLite.

## Tech Stack

- **Backend**: Python
- **Computer Vision**: `face-recognition`, `dlib`, `opencv-python`
- **Machine Learning**: `scikit-learn`
- **Configuration**: `PyYAML`
- **Web**: Flask (if using `web_server.py`)
- **CLI**: `tqdm`
- **Database**: SQLite

## Setup & Installation

1. Clone the repository:
   ```
   git clone https://github.com/samarthsingh02/Face-based-Photo-Segregation.git
   ```
2. Navigate to the project directory:
   ```
   cd Face-based-Photo-Segregation
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

### CLI

1. Add your photos to the `source_images` directory (subfolders by person recommended).
2. Configure settings in `config.yaml` (e.g., set `active_model`).
3. Run the main script:
   ```
   python main.py
   ```
4. Check the `output_sorted` folder for results.

### Web Server (Optional)

1. Start the web server:
   ```
   python web_server.py
   ```
2. Open your browser and go to `http://localhost:5000`.
3. Upload images and view sorted results in the web interface.

## Output

- **output_sorted/**: Contains folders for each identified person and unknowns.
- **faces.db**: SQLite database storing face encodings and clustering info.
- **run_log.txt**: Execution logs.
- **Plots**: Various clustering and distance plots for analysis.

## Helper Scripts

- `find_eps.py`, `tune_eps.py`: For tuning clustering parameters.
- `cleanup_db.py`: For cleaning up the database.
- `run_cnn_experiment.py`: For running experiments with the CNN model.

