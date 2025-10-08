import sys
import yaml
import face_recognition
import numpy as np
from deepface import DeepFace


def calculate_distance(encoding1, encoding2, metric="euclidean"):
    """Calculates distance between two embeddings using the specified metric."""
    if metric == "euclidean":
        return np.linalg.norm(encoding1 - encoding2)
    elif metric == "cosine":
        # Cosine distance is 1 - cosine similarity
        dot = np.dot(encoding1, encoding2)
        norm = np.linalg.norm(encoding1) * np.linalg.norm(encoding2)
        return 1 - (dot / norm)
    return -1


def main():
    if len(sys.argv) != 3:
        print("Usage: python investigate_face.py <path_to_good_image> <path_to_problem_image>")
        return

    good_image_path = sys.argv[1]
    problem_image_path = sys.argv[2]

    # --- Load Configuration to use the active preset ---
    try:
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        active_model_name = config['active_model']
        preset = config['presets'][active_model_name]
        print(f"--- Using preset: '{active_model_name}' ---")
    except Exception as e:
        print(f"Error loading config.yaml: {e}")
        return

    # --- Generate Encodings using the correct library ---
    encodings = []
    for path in [good_image_path, problem_image_path]:
        try:
            if preset['library'] == 'dlib':
                image = face_recognition.load_image_file(path)
                location = face_recognition.face_locations(image, model=preset['model'])[0]
                encoding = face_recognition.face_encodings(image, [location])[0]
                encodings.append(encoding)
            elif preset['library'] == 'deepface':
                embedding_obj = DeepFace.represent(
                    img_path=path,
                    model_name=preset['embedding_model'],
                    detector_backend=preset['detector'],
                    enforce_detection=True
                )
                encodings.append(embedding_obj[0]['embedding'])
        except IndexError:
            print(f"Error: Could not find a face in {path}. Skipping.")
            return
        except Exception as e:
            print(f"Error processing {path}: {e}")
            return

    # --- Calculate and Report the Distance ---
    metric = preset['clustering']['metric']
    eps_threshold = preset['clustering']['eps']
    distance = calculate_distance(encodings[0], encodings[1], metric=metric)

    print(f"\nDistance between faces (using '{metric}' metric): {distance:.4f}")
    print(f"Your current 'eps' threshold is: {eps_threshold}")

    if distance > eps_threshold:
        print("\nDiagnosis: The distance is LARGER than your eps threshold.")
        print("This is why the face was classified as 'Unknown.'")
    else:
        print("\nDiagnosis: The distance is WITHIN your eps threshold.")
        print("The issue might be that the person's cluster wasn't formed correctly (e.g., min_samples not met).")


if __name__ == "__main__":
    main()