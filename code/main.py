import os
import face_recognition
import cv2
from pathlib import Path
from shutil import move
from iptcinfo3 import IPTCInfo
import pickle

# Paths
TRAINING_DATA_PATH = "../data/training_data/"
PHOTO_DIR = "../data/photos/"
CROPPED_FACES_DIR = "../data/cropped_faces/"
ENCODINGS_FILE = "../data/encodings.pkl"

# Ensure directories exist
os.makedirs(CROPPED_FACES_DIR, exist_ok=True)


# Function to load training data
def load_training_data():
    try:
        if os.path.exists(ENCODINGS_FILE):
            print("Loading encodings from file...")
            with open(ENCODINGS_FILE, "rb") as f:
                data = pickle.load(f)
            return data["encodings"], data["names"]
        else:
            print(f"Encodings file not found at {ENCODINGS_FILE}.")
    except Exception as e:
        print(f"Error loading encodings: {e}")
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
            
            # Detect faces in the image
            face_locations = face_recognition.face_locations(image, model='hog')
            face_encodings = face_recognition.face_encodings(image, face_locations)

            for i, face_encoding in enumerate(face_encodings):
                top, right, bottom, left = face_locations[i]

                # Check if the face matches known faces
                matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.4)
                if not any(matches):
                    # Adjust the crop area with a margin
                    margin = 30  # Adjust margin size as needed
                    top = max(0, top - margin)
                    right = min(image.shape[1], right + margin)
                    bottom = min(image.shape[0], bottom + margin)
                    left = max(0, left - margin)

                    # Crop and save the unrecognized face
                    face_image = image[top:bottom, left:right]
                    face_image_path = os.path.join(CROPPED_FACES_DIR, f"face_{len(unrecognized_faces)}.jpg")
                    cv2.imwrite(face_image_path, cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR))
                    unrecognized_faces.append(face_image_path)

                    print(f"Cropped image dimensions for {face_image_path}: {face_image.shape}")

        except Exception as e:
            print(f"Error processing {image_path}: {e}")

    return unrecognized_faces


# Function to label and update training data
def label_and_update_training_data(unrecognized_faces, known_encodings, known_names):
    for face_path in unrecognized_faces:
        try:
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

            # Reload the cropped image and validate face encoding
            image = face_recognition.load_image_file(face_path)
            face_locations = face_recognition.face_locations(image, model='hog')
            if not face_locations:
                print(f"No face detected in {face_path}, skipping.")
                continue

            face_encoding = face_recognition.face_encodings(image)[0]

            # Add encoding to the known list
            known_encodings.append(face_encoding)
            known_names.append(label)
            print(f"Added '{label}' to known encodings.")

            # Move to training data
            label_dir = os.path.join(TRAINING_DATA_PATH, label)
            os.makedirs(label_dir, exist_ok=True)
            moved_path = os.path.join(label_dir, os.path.basename(face_path))
            move(face_path, moved_path)
            print(f"Moved {face_path} to {moved_path}/")

        except IndexError:
            print(f"No face found in {face_path}, skipping.")
        except Exception as e:
            print(f"Error labeling face {face_path}: {e}")


# Function to retrain the model
def retrain_model():
    print("Retraining the model...")
    encodings, names = [], []
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
                print(f"No face found in {image_path}")
            except Exception as e:
                print(f"Error processing {image_path}: {e}")

    # Save the encodings to a file for future use
    with open(ENCODINGS_FILE, "wb") as f:
        pickle.dump({"encodings": encodings, "names": names}, f)

    print("Encodings saved to file.")
    return encodings, names



# Function to update image metadata
def update_image_metadata(photo_dir, known_encodings, known_names):
    for image_name in os.listdir(photo_dir):
        if image_name.startswith("."):  # Ignore hidden files
            continue

        image_path = os.path.join(photo_dir, image_name)
        try:
            print(f"Updating metadata for {image_name}...")
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
                info = IPTCInfo(image_path, force=True)
                info['keywords'] = list(set(names_in_image))
                info.save_as(image_path)
                print(f"Updated metadata for {image_name} with names: {names_in_image}")

        except Exception as e:
            print(f"Error updating metadata for {image_name}: {e}")

# Function to clean up temporary files
def clean_photos_directory():
    for file_name in os.listdir(PHOTO_DIR):
        file_path = os.path.join(PHOTO_DIR, file_name)
        try:
            # Check for files ending with '~' (temporary files)
            if file_name.endswith('~'):
                os.remove(file_path)
                print(f"Deleted leftover temporary file: {file_path}")
        except Exception as e:
            print(f"Error deleting temporary file {file_path}: {e}")


# Main script execution
if __name__ == "__main__":
    print("Loading training data...")
    known_encodings, known_names = load_training_data()

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

    print("Cleaning up /photos directory")
    clean_photos_directory()

    print("Cleaning up cropped_faces directory...")
    for file_name in os.listdir(CROPPED_FACES_DIR):
        os.remove(os.path.join(CROPPED_FACES_DIR, file_name))
