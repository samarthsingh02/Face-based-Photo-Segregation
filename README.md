# Face-Based Photo Segregation

An intelligent web application that automatically organizes your photo collections by detecting and grouping faces. Built with privacy in mind, all processing happens locally on your machine.

## Summary

Face-Based Photo Segregation scans your images, detects faces, and groups them into clusters for each unique person using advanced computer vision and machine learning. It is pre-configured to use the fast and efficient **HOG (Histogram of Oriented Gradients)** model to ensure smooth processing, even with large batches of photos.

## Key Features

- **Bulk Upload Support**: Drag and drop individual images or upload entire **.zip files** to process hundreds of photos at once.
- **Automated Clustering**: Uses unsupervised learning (DBSCAN) to group faces without manual training.
- **Fast Processing**: Optimized with the dlib HOG model for a perfect balance of speed and accuracy.
- **Download Results**: Easily download your organized collection as a `.zip` file.
- **Cluster Naming**: Rename "Person 1" to real names (e.g., "Samarth") directly in the interface.
- **Privacy-First**: All processing happens locally—no cloud uploads.
- **SQLite Database**: Efficiently stores face encodings to recognize people across different upload sessions.

## Tech Stack

- **Backend**: Python, Flask
- **Computer Vision**: `face-recognition` (dlib), `opencv-python`
- **Machine Learning**: `scikit-learn` (DBSCAN clustering)
- **Database**: SQLite
- **Configuration**: PyYAML
- **Frontend**: HTML5, CSS3, Vanilla JavaScript

## Prerequisites

- Python 3.8 or higher
- **Windows Users**: Visual Studio Build Tools with "Desktop development with C++" (required for compiling dlib)

## Setup & Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/samarthsingh02/Face-based-Photo-Segregation.git
   cd Face-based-Photo-Segregation
   ```

2. **Set up a virtual environment (Recommended):**
   ```bash
   python -m venv .venv

   # Windows:
   .venv\Scripts\activate

   # Mac/Linux:
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Web Interface

1. **Start the server:**
   ```bash
   python web_server.py
   ```

2. **Open your browser and navigate to:**
   ```
   http://127.0.0.1:5001
   ```

3. **Upload and process:**
   - Drag & drop photos or a .zip file into the upload zone.
   - Click **Start Processing**.
   - Rename clusters by clicking the pencil icon (✏️).
   - Click **Download .zip** to get your sorted folders.

### Command Line Interface

1. Add photos to the `source_images` folder.

2. Run the processing script:
   ```bash
   python main.py
   ```

3. Check the `output_sorted` folder for organized results.

## Project Structure

```
Face-based-Photo-Segregation/
├── web_server.py           # Flask web application backend
├── main.py                 # CLI entry point
├── core_engine.py          # Face detection, encoding, and clustering logic
├── database.py             # SQLite database interactions
├── config.yaml             # Configuration settings
├── static/                 # CSS, JavaScript, and uploads
├── templates/              # HTML frontend
├── source_images/          # Input photos (CLI mode)
├── output_sorted/          # Organized results
└── faces.db                # SQLite database
```

## Output

- **output_sorted/**: Folders for each identified person and unknowns.
- **faces.db**: SQLite database with face encodings and clustering data.
- **run_log.txt**: Execution logs.

## Configuration

The application uses `config.yaml` to manage settings. By default, it is set to:

- **Model**: `dlib_hog` (Fast, CPU-friendly)
- **Resize Width**: 450px (Optimized for detection speed)

Advanced users can modify `config.yaml` to tweak clustering sensitivity (`eps`) or minimum samples.

## License

This project is open-source. Feel free to modify and distribute it.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
