import os
import pickle
import face_recognition
from iptcinfo3 import IPTCInfo

# Paths
PHOTO_DIR = "../data/photos/"

# Function to load training data
def load_training_data():
    if os.path.exists("../data/encodings.pkl"):
        try:
            with open("../data/encodings.pkl", "rb") as f:
                data = pickle.load(f)
            return data["encodings"], data["names"]
        except Exception as e:
            print(f"Error loading encodings.pkl: {e}")
    else:
        print("encodings.pkl file not found.")
    return [], []

# Function to validate metadata updates
def validate_metadata_updates(photo_dir, known_encodings, known_names):
    supported_extensions = ['.jpg', '.jpeg', '.png']  # Only process these extensions

    for image_name in os.listdir(photo_dir):
        if image_name.startswith(".") or "~" in image_name:  # Skip hidden files and backup files
            continue

        image_path = os.path.join(photo_dir, image_name)

        # Ensure the file has a supported extension (in case of incorrect naming)
        if not any(image_name.lower().endswith(ext) for ext in supported_extensions):
            print(f"Skipping unsupported or incorrectly named file: {image_name}")
            continue

        try:
            # Load image and find face encodings
            image = face_recognition.load_image_file(image_path)
            face_encodings = face_recognition.face_encodings(image)

            # Find matches for all faces in the image
            names_in_image = []
            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.4)
                if True in matches:
                    match_index = matches.index(True)
                    names_in_image.append(known_names[match_index])

            # Update metadata if names are found
            if names_in_image:
                try:
                    info = IPTCInfo(image_path, force=True)
                    info['keywords'] = names_in_image
                    info.save_as(image_path)
                    print(f"Updated metadata for {image_name} with names: {names_in_image}")
                except Exception as e:
                    print(f"Error updating metadata for {image_name}: {e}")
            else:
                print(f"No face matches found in {image_name}.")

        except Exception as e:
            print(f"Error processing {image_path}: {e}")

if __name__ == "__main__":
    print("Loading training data...")
    known_encodings, known_names = load_training_data()

    if known_encodings:
        print("Validating metadata updates...")
        validate_metadata_updates(PHOTO_DIR, known_encodings, known_names)
    else:
        print("No known encodings found. Please ensure training data exists.")
