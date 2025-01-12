import os
import face_recognition
import pickle
from iptcinfo3 import IPTCInfo

# Paths
TRAINING_DATA_PATH = "../data/training_data/"
PHOTO_DIR = "../data/photos/"
CROPPED_FACES_DIR = "../data/cropped_faces/"
ENCODINGS_FILE = "../data/encodings.pkl"

# Function to load training data
def load_training_data():
    if os.path.exists(ENCODINGS_FILE):
        try:
            with open(ENCODINGS_FILE, "rb") as f:
                data = pickle.load(f)
            return data["encodings"], data["names"]
        except Exception as e:
            print(f"Error loading encodings.pkl: {e}")
    else:
        print(f"{ENCODINGS_FILE} file not found.")
    return [], []

# Function to validate metadata updates
def validate_metadata_update(photo_dir, known_encodings, known_names):
    updated_files = []

    for image_name in os.listdir(photo_dir):
        if image_name.startswith("."):  # Ignore hidden files
            continue

        image_path = os.path.join(photo_dir, image_name)
        try:
            image = face_recognition.load_image_file(image_path)
            face_encodings = face_recognition.face_encodings(image)

            # Find matches for all faces in the image
            names_in_image = []
            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.4)
                if True in matches:
                    match_index = matches.index(True)
                    names_in_image.append(known_names[match_index])

            # Check if names were found and update metadata
            if names_in_image:
                info = IPTCInfo(image_path, force=True)
                existing_keywords = info.get('keywords', [])
                updated_keywords = list(set(existing_keywords + names_in_image))
                if updated_keywords != existing_keywords:
                    info['keywords'] = updated_keywords
                    info.save_as(image_path)
                    updated_files.append(image_name)
                    print(f"Updated metadata for {image_name} with names: {updated_keywords}")

        except Exception as e:
            print(f"Error updating metadata for {image_name}: {e}")

    return updated_files

# Function to check if all images in /photos directory are processed and metadata updated
def check_all_files_processed(photo_dir, updated_files):
    unprocessed_files = []
    for image_name in os.listdir(photo_dir):
        if image_name.startswith("."):  # Ignore hidden files
            continue
        if image_name not in updated_files:
            unprocessed_files.append(image_name)

    if unprocessed_files:
        print(f"These files were not processed: {', '.join(unprocessed_files)}")
    else:
        print("All files in the photos directory were processed and metadata updated.")

if __name__ == "__main__":
    print("Loading training data...")
    known_encodings, known_names = load_training_data()

    if not known_encodings:
        print("No known encodings found. Please ensure training data exists.")
    else:
        print("Validating metadata updates...")
        updated_files = validate_metadata_update(PHOTO_DIR, known_encodings, known_names)

        if updated_files:
            print(f"Metadata updated for the following files: {', '.join(updated_files)}")

        # Check if all files in /photos have been processed
        check_all_files_processed(PHOTO_DIR, updated_files)
