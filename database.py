# database.py

import sqlite3
import numpy as np
import pickle
import logging

DB_FILE = "faces.db"


def init_db():
    """Initializes the database, adding the person_id column and foreign key."""
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()

        # --- NEW: Create people table first if it doesn't exist ---
        cursor.execute("""
                       CREATE TABLE IF NOT EXISTS people
                       (
                           id    INTEGER PRIMARY KEY AUTOINCREMENT,
                           name  TEXT NOT NULL UNIQUE
                       );
                       """)

        # --- UPDATED: Add person_id to faces table ---
        cursor.execute("""
                       CREATE TABLE IF NOT EXISTS faces
                       (
                           id                INTEGER PRIMARY KEY AUTOINCREMENT,
                           image_path        TEXT NOT NULL,
                           encoding          BLOB NOT NULL,
                           model_preset_name TEXT NOT NULL,
                           person_id         INTEGER,
                           FOREIGN KEY (person_id) REFERENCES people (id)
                       );
                       """)

        cursor.execute("CREATE INDEX IF NOT EXISTS idx_image_path ON faces (image_path);")
        conn.commit()
    except Exception as e:
        logging.error(f"Error initializing database: {e}")
    finally:
        if conn:
            conn.close()


def add_faces_and_get_ids(face_data_list, model_preset_name):
    """
    Adds a list of face data to the database and returns their new IDs.
    Some faces may already have a person_id if they were recognized.
    """
    inserted_ids = []
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        for face in face_data_list:
            encoding_blob = pickle.dumps(face['encoding'])
            # Check if a person_id was assigned during recognition
            person_id = face.get('person_id')
            cursor.execute(
                "INSERT INTO faces (image_path, encoding, model_preset_name, person_id) VALUES (?, ?, ?, ?)",
                (face['image_path'], encoding_blob, model_preset_name, person_id)
            )
            inserted_ids.append(cursor.lastrowid)
        conn.commit()
        logging.info(f"Successfully added {len(inserted_ids)} new '{model_preset_name}' faces to the database.")
    except Exception as e:
        logging.error(f"Error adding faces to database: {e}")
    finally:
        if conn:
            conn.close()
    return inserted_ids


def get_named_faces_by_model(model_preset_name):
    """Retrieves all faces from the database that have been assigned to a person."""
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        # Use a JOIN to get face data only for faces linked to a person
        cursor.execute("""
            SELECT f.id, f.encoding, p.name, p.id as person_id
            FROM faces f
            JOIN people p ON f.person_id = p.id
            WHERE f.model_preset_name = ?
        """, (model_preset_name,))

        named_faces = []
        for row in cursor.fetchall():
            face_id, encoding_blob, person_name, person_id = row
            encoding = pickle.loads(encoding_blob)
            named_faces.append({
                'id': face_id,
                'encoding': encoding,
                'person_name': person_name,
                'person_id': person_id
            })
        conn.close()
        logging.info(f"Loaded {len(named_faces)} existing named faces for preset '{model_preset_name}'.")
        return named_faces
    except Exception as e:
        logging.error(f"Error retrieving named faces from database: {e}")
        return []


def get_or_create_person(name):
    """
    Finds a person by name. If they don't exist, a new entry is created.
    Returns the unique ID of the person.
    """
    person_id = None
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM people WHERE name = ?", (name,))
        result = cursor.fetchone()
        if result:
            person_id = result[0]
        else:
            cursor.execute("INSERT INTO people (name) VALUES (?)", (name,))
            person_id = cursor.lastrowid
            logging.info(f"Created new person '{name}' with ID {person_id}.")
        conn.commit()
    except Exception as e:
        logging.error(f"Error in get_or_create_person: {e}")
    finally:
        if conn:
            conn.close()
    return person_id


def link_faces_to_person(face_ids, person_id):
    """Links a list of face IDs to a person ID."""
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        data_to_update = [(person_id, face_id) for face_id in face_ids]
        cursor.executemany("UPDATE faces SET person_id = ? WHERE id = ?", data_to_update)
        conn.commit()
        logging.info(f"Successfully linked {cursor.rowcount} faces to person_id {person_id}.")
    except Exception as e:
        logging.error(f"Error in link_faces_to_person: {e}")
    finally:
        if conn:
            conn.close()