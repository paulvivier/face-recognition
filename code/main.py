import os
import face_recognition
import cv2
from pathlib import Path
from shutil import move
import shutil
from iptcinfo3 import IPTCInfo
import pickle

# Paths
TRAINING_DATA_PATH = "../data/training_data/"
PHOTO_DIR = "../data/photos/"
CROPPED_FACES_DIR = "../data/cropped_faces/"
ORIGINAL_DIR = "../data/original/"

# Create necessary directories if they don't exist
os.makedirs(CROPPED_FACES_DIR, exist_ok=True)
os.makedirs(ORIGINAL_DIR, exist_ok=True)

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

# Function to detect and crop unrecognized faces
def extract_unrecognized_faces(known_encodings, known_names):
    unrecognized_faces = []

    for image_name in os.listdir(PHOTO_DIR):
        if image_name.startswith("."):  # Ignore hidden files
            continue

        image_path = os.path.join(PHOTO_DIR, image_name)
        try:
            image = face_recognition.load_image_file(image_path)
            
            # Use CNN or HOG model for face detection
            face_locations = face_recognition.face_locations(image, model='hog')
            face_encodings = face_recognition.face_encodings(image, face_locations)

            for i, face_encoding in enumerate(face_encodings):
                top, right, bottom, left = face_locations[i]

                # Check if the face matches known faces with stricter tolerance
                matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.4)
                if not any(matches):
                    # Crop and save the unrecognized face
                    face_image = image[top:bottom, left:right]
                    face_image_path = os.path.join(CROPPED_FACES_DIR, f"face_{len(unrecognized_faces)}.jpg")
                    cv2.imwrite(face_image_path, cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR))
                    unrecognized_faces.append(face_image_path)

        except Exception as e:
            print(f"Error processing {image_path}: {e}")

    return unrecognized_faces

# Function to label and update training data
def label_and_update_training_data(unrecognized_faces, known_encodings, known_names):
    for face_path in unrecognized_faces:
        # Display the cropped face
        face_image = cv2.imread(face_path)
        cv2.imshow("Unrecognized Face", face_image)
        cv2.waitKey(1)

        # Prompt for a name
        label = input(f"Enter the name for the face in {face_path} (or type 'unknown' to skip): ").strip()
        cv2.destroyAllWindows()

        if label.lower() == "unknown":
            print(f"Skipped face: {face_path}")
            continue

        # Encode the face before moving the file
        try:
            image = face_recognition.load_image_file(face_path)
            face_encoding = face_recognition.face_encodings(image)[0]

            # Add encoding to the known list
            known_encodings.append(face_encoding)
            known_names.append(label)
            print(f"Added '{label}' to known encodings.")
        except IndexError:
            print(f"No face found in {face_path}, skipping.")
            continue

        # Move to training data
        label_dir = os.path.join(TRAINING_DATA_PATH, label)
        os.makedirs(label_dir, exist_ok=True)
        moved_path = os.path.join(label_dir, os.path.basename(face_path))
        move(face_path, moved_path)
        print(f"Moved {face_path} to {moved_path}/")

# Function to retrain the model
def retrain_model(force_retrain=False):
    encodings, names = load_training_data()

    if not force_retrain and encodings:
        print("Loading encodings from file...")
        return encodings, names

    print("Generating new encodings from training data...")
    encodings = []
    names = []
    for person_name in os.listdir(TRAINING_DATA_PATH):
        person_path = os.path.join(TRAINING_DATA_PATH, person_name)
        if not os.path.isdir(person_path) or person_name.startswith("."):
            continue

        for image_name in os.listdir(person_path):
            if image_name.startswith("."):  # Ignore hidden files
                continue

            image_path = os.path.join(person_path, image_name)
            try:
                image = face_recognition.load_image_file(image_path)
                encoding = face_recognition.face_encodings(image)[0]
                encodings.append(encoding)
                names.append(person_name)
            except IndexError:
                print(f"Face not found in {image_path}")
            except Exception as e:
                print(f"Error processing {image_path}: {e}")

    # Save the encodings to a file for future use
    with open("../data/encodings.pkl", "wb") as f:
        pickle.dump({"encodings": encodings, "names": names}, f)

    print("Encodings saved to file.")
    return encodings, names

# Function to update image metadata without overwriting existing names
# Function to update image metadata for known faces
def update_image_metadata(photo_dir, known_encodings, known_names):
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

            # Update metadata if names are found
            if names_in_image:
                # Get existing keywords
                info = IPTCInfo(image_path, force=True)
                existing_keywords = info.get_all('keywords', [])
                
                # Add new names if they're not already present
                updated_keywords = list(set(existing_keywords + names_in_image))  # Avoid duplicates
                
                info['keywords'] = updated_keywords
                info.save_as(image_path)
                print(f"Updated metadata for {image_name} with names: {updated_keywords}")

        except Exception as e:
            print(f"Error updating metadata for {image_name}: {e}")

# Function to clean up cropped_faces directory
def clean_cropped_faces_dir():
    for file_name in os.listdir(CROPPED_FACES_DIR):
        file_path = os.path.join(CROPPED_FACES_DIR, file_name)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
                print(f"Deleted leftover file: {file_path}")
        except Exception as e:
            print(f"Error deleting file {file_path}: {e}")


if __name__ == "__main__":
    print("Loading training data...")
    known_encodings, known_names = retrain_model()

    print("Extracting unrecognized faces...")
    unrecognized_faces = extract_unrecognized_faces(known_encodings, known_names)

    if unrecognized_faces:
        print(f"Found {len(unrecognized_faces)} unrecognized faces.")
        label_and_update_training_data(unrecognized_faces, known_encodings, known_names)

        print("Retraining the model...")
        known_encodings, known_names = retrain_model()
        print("Model updated successfully.")
    else:
        print("No unrecognized faces found.")

    print("Updating metadata for existing images...")
    update_image_metadata(PHOTO_DIR, known_encodings, known_names)

    print("Cleaning up cropped_faces directory...")
    clean_cropped_faces_dir()
