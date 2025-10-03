# Face-Based Photo Segregation

A Python-based application that automatically analyzes a collection of photos, identifies unique individuals using face recognition, and sorts the photos into folders for each person.

## Key Features

- **Automated Clustering**: Uses DBSCAN to automatically group faces without prior training.
- **Configurable Models**: Easily switch between a fast `HOG` model and a high-accuracy `CNN` model.
- **Performance Tuning**: Includes helper scripts and a detailed configuration file to tune parameters like image resize width and clustering `eps`.
- **Detailed Logging**: Generates a run summary and clustering quality report (V-Measure) for each execution.

## Tech Stack

- **Backend**: Python
- **Computer Vision**: `face-recognition`, `dlib`, `opencv-python`
- **Machine Learning**: `scikit-learn`
- **Configuration**: `PyYAML`
- **CLI**: `tqdm`

## Setup & Installation

1.  Clone the repository:
    `git clone https://github.com/samarthsingh02/Face-based-Photo-Segregation.git`
2.  Navigate to the project directory:
    `cd Face-based-Photo-Segregation`
3.  Install dependencies:
    `pip install -r requirements.txt`

## Usage

1.  Add your photos (organized in subfolders by person) to the `source_images` directory.
2.  Configure your desired settings in `config.yaml` (e.g., set `active_model`).
3.  Run the script from your terminal:
    `python main.py`
4.  Check the `output_sorted` folder for the results.