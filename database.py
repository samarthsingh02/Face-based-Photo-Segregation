import psycopg2
import psycopg2.extras
import yaml
import logging
import pickle
import numpy as np
from pgvector.psycopg2 import register_vector

# --- Database Configuration ---
DB_CONFIG = None


def _load_config():
    """Loads database configuration from config.yaml"""
    global DB_CONFIG
    if DB_CONFIG is None:
        try:
            with open('config.yaml', 'r') as f:
                config = yaml.safe_load(f)
            DB_CONFIG = config['postgresql']
        except Exception as e:
            logging.critical(f"FATAL: Could not load postgresql config from config.yaml: {e}")
            raise


def get_db_connection():
    """
    Establishes a new connection to the PostgreSQL database.
    """
    if DB_CONFIG is None:
        _load_config()

    try:
        conn = psycopg2.connect(
            host=DB_CONFIG['host'],
            port=DB_CONFIG['port'],
            user=DB_CONFIG['user'],
            password=DB_CONFIG['password'],
            dbname=DB_CONFIG['dbname']
        )
        # IMPORTANT: Register the vector type
        register_vector(conn)
        return conn
    except Exception as e:
        logging.error(f"Error connecting to PostgreSQL database: {e}")
        return None


def init_db():
    """
    Initializes the database by creating tables.
    This uses the new "Persistent Gallery" schema.
    """
    # Note: dlib/face_recognition uses 128-dim vectors.
    # If you use other models (like VGG-Face), you'll need to change this.
    ENCODING_VECTOR_DIM = 128

    create_extension_sql = "CREATE EXTENSION IF NOT EXISTS vector;"

    create_people_sql = """
                        CREATE TABLE IF NOT EXISTS people \
                        ( \
                            id \
                            SERIAL \
                            PRIMARY \
                            KEY, \
                            name \
                            TEXT \
                            NOT \
                            NULL \
                            UNIQUE
                        ); \
                        """

    create_photos_sql = """
                        CREATE TABLE IF NOT EXISTS photos \
                        ( \
                            id \
                            SERIAL \
                            PRIMARY \
                            KEY, \
                            permanent_path \
                            TEXT \
                            NOT \
                            NULL \
                            UNIQUE, \
                            original_filename \
                            TEXT \
                            NOT \
                            NULL, \
                            uploaded_at \
                            TIMESTAMPTZ \
                            NOT \
                            NULL \
                            DEFAULT ( \
                            NOW \
                        ( \
                        ))
                            );
                        CREATE INDEX IF NOT EXISTS idx_photo_path ON photos (permanent_path); \
                        """

    create_faces_sql = f"""
    CREATE TABLE IF NOT EXISTS faces (
        id                  SERIAL PRIMARY KEY,
        photo_id            INTEGER NOT NULL,
        person_id           INTEGER,
        model_preset_name   TEXT NOT NULL,
        encoding            vector({ENCODING_VECTOR_DIM}),

        FOREIGN KEY (photo_id) REFERENCES photos (id) ON DELETE CASCADE,
        FOREIGN KEY (person_id) REFERENCES people (id) ON DELETE SET NULL
    );
    CREATE INDEX IF NOT EXISTS idx_face_photo_id ON faces (photo_id);
    CREATE INDEX IF NOT EXISTS idx_face_person_id ON faces (person_id);
    """

    conn = None
    try:
        conn = get_db_connection()
        if conn is None:
            raise Exception("Failed to get database connection.")

        with conn.cursor() as cursor:
            logging.info("Initializing database schema...")
            cursor.execute(create_extension_sql)
            logging.info("Ensured 'vector' extension exists.")
            cursor.execute(create_people_sql)
            logging.info("Table 'people' is ready.")
            cursor.execute(create_photos_sql)
            logging.info("Table 'photos' is ready.")
            cursor.execute(create_faces_sql)
            logging.info("Table 'faces' is ready.")

        conn.commit()
        logging.info("Database schema initialization complete.")

    except Exception as e:
        logging.error(f"Error initializing database: {e}")
        if conn:
            conn.rollback()
    finally:
        if conn:
            conn.close()


def add_photo(permanent_path, original_filename):
    """
    Adds a new photo to the database and returns its new ID.
    Returns None if it fails (e.g., photo is already in DB).
    """
    sql = """
          INSERT INTO photos (permanent_path, original_filename)
          VALUES (%s, %s) ON CONFLICT (permanent_path) DO NOTHING
    RETURNING id; \
          """
    conn = None
    photo_id = None
    try:
        conn = get_db_connection()
        with conn.cursor() as cursor:
            cursor.execute(sql, (permanent_path, original_filename))
            # Check if a row was returned
            result = cursor.fetchone()
            if result:
                photo_id = result[0]
        conn.commit()
    except Exception as e:
        logging.error(f"Error adding photo {original_filename} to DB: {e}")
        if conn:
            conn.rollback()
    finally:
        if conn:
            conn.close()
    return photo_id


def add_face(photo_id, model_preset, encoding, person_id=None):
    """
    Adds a single detected face to the database.
    """
    # Convert numpy array to list for pgvector
    encoding_list = np.array(encoding).tolist()

    sql = """
          INSERT INTO faces (photo_id, model_preset_name, encoding, person_id)
          VALUES (%s, %s, %s, %s) RETURNING id; \
          """
    conn = None
    face_id = None
    try:
        conn = get_db_connection()
        with conn.cursor() as cursor:
            cursor.execute(sql, (photo_id, model_preset, encoding_list, person_id))
            face_id = cursor.fetchone()[0]
        conn.commit()
    except Exception as e:
        logging.error(f"Error adding face for photo_id {photo_id}: {e}")
        if conn:
            conn.rollback()
    finally:
        if conn:
            conn.close()
    return face_id


def get_or_create_person(name):
    """
    Finds a person by name. If they don't exist, creates them.
    Returns the person's ID.
    """
    conn = None
    person_id = None
    try:
        conn = get_db_connection()
        with conn.cursor() as cursor:
            # First, try to find the person
            cursor.execute("SELECT id FROM people WHERE name = %s;", (name,))
            result = cursor.fetchone()

            if result:
                person_id = result[0]
            else:
                # If not found, create them
                cursor.execute("INSERT INTO people (name) VALUES (%s) RETURNING id;", (name,))
                person_id = cursor.fetchone()[0]
                logging.info(f"Created new person '{name}' with ID {person_id}.")
        conn.commit()
    except Exception as e:
        logging.error(f"Error in get_or_create_person('{name}'): {e}")
        if conn:
            conn.rollback()
    finally:
        if conn:
            conn.close()
    return person_id


def link_faces_to_person(face_ids, person_id):
    """Links a list of face IDs to a person ID."""
    if not isinstance(face_ids, list) or len(face_ids) == 0:
        return

    sql = "UPDATE faces SET person_id = %s WHERE id IN %s;"
    conn = None
    try:
        conn = get_db_connection()
        with conn.cursor() as cursor:
            # psycopg2 needs a tuple for the IN clause
            face_ids_tuple = tuple(face_ids)
            cursor.execute(sql, (person_id, face_ids_tuple))
            count = cursor.rowcount
        conn.commit()
        logging.info(f"Successfully linked {count} faces to person_id {person_id}.")
    except Exception as e:
        logging.error(f"Error in link_faces_to_person: {e}")
        if conn:
            conn.rollback()
    finally:
        if conn:
            conn.close()


def get_named_faces_by_model(model_preset_name):
    """
    Retrieves all faces from the database that have been assigned to a person.
    This is used for the "recognition" step.
    """
    sql = """
          SELECT f.id, f.encoding, p.name, p.id as person_id
          FROM faces f
                   JOIN people p ON f.person_id = p.id
          WHERE f.model_preset_name = %s; \
          """
    conn = None
    named_faces = []
    try:
        conn = get_db_connection()
        # Use a dictionary cursor to get results as key-value pairs
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cursor:
            cursor.execute(sql, (model_preset_name,))
            for row in cursor.fetchall():
                named_faces.append({
                    'id': row['id'],
                    'encoding': np.array(row['encoding']),  # Convert list back to numpy array
                    'person_name': row['person_name'],
                    'person_id': row['person_id']
                })
        logging.info(f"Loaded {len(named_faces)} existing named faces for preset '{model_preset_name}'.")
    except Exception as e:
        logging.error(f"Error retrieving named faces from database: {e}")
    finally:
        if conn:
            conn.close()
    return named_faces