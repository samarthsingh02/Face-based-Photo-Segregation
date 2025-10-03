# database.py

import sqlite3
import numpy as np
import pickle
import logging

DB_FILE = "faces.db"


def init_db():
    """Initializes the database, adding the new model_preset_name column."""
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute("""
                       CREATE TABLE IF NOT EXISTS faces
                       (
                           id
                           INTEGER
                           PRIMARY
                           KEY
                           AUTOINCREMENT,
                           image_path
                           TEXT
                           NOT
                           NULL,
                           encoding
                           BLOB
                           NOT
                           NULL,
                           model_preset_name
                           TEXT
                           NOT
                           NULL -- <-- NEW COLUMN
                       );
                       """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_image_path ON faces (image_path);")
        conn.commit()
        conn.close()
        logging.info(f"Database '{DB_FILE}' initialized successfully.")
    except Exception as e:
        logging.error(f"Error initializing database: {e}")


def add_faces(face_data_list, model_preset_name):
    """Adds a list of face data to the database, tagged with the model preset name."""
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        faces_to_insert = []
        for face in face_data_list:
            encoding_blob = pickle.dumps(face['encoding'])
            # Add the model preset name to the data being inserted
            faces_to_insert.append((face['image_path'], encoding_blob, model_preset_name))

        cursor.executemany(
            "INSERT INTO faces (image_path, encoding, model_preset_name) VALUES (?, ?, ?)",
            faces_to_insert
        )
        conn.commit()
        conn.close()
        logging.info(f"Successfully added {len(faces_to_insert)} new '{model_preset_name}' faces to the database.")
    except Exception as e:
        logging.error(f"Error adding faces to database: {e}")


def get_faces_by_model(model_preset_name):
    """Retrieves all faces from the database that were created with a specific model preset."""
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()

        # Select only the faces matching the current model
        cursor.execute("SELECT image_path, encoding FROM faces WHERE model_preset_name = ?", (model_preset_name,))

        all_face_data = []
        for row in cursor.fetchall():
            image_path, encoding_blob = row
            # De-serialize the blob back into a NumPy array
            encoding = pickle.loads(encoding_blob)
            all_face_data.append({'image_path': image_path, 'encoding': encoding})

        conn.close()
        logging.info(f"Loaded {len(all_face_data)} existing faces for preset '{model_preset_name}' from database.")
        return all_face_data
    except Exception as e:
        logging.error(f"Error retrieving faces from database: {e}")
        return []