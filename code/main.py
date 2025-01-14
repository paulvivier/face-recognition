import os
import cv2
import face_recognition
import pickle
from PIL import Image
from PIL import ImageOps
from pathlib import Path
import iptcinfo3
import shutil
import numpy as np

# Directories for photos and training data
photos_dir = "../data/photos"
training_data_dir = "../data/training_data"
cropped_faces_dir = "../data/cropped_faces"
encodings_file = "../data/encodings.pkl"

# Load existing encodings or initialize empty list if none exist
if os.path.exists(encodings_file):
    with open(encodings_file, "rb") as f:
        known_face_encodings, known_face_names = pickle.load(f)
else:
    known_face_encodings, known_face_names = [], []

def validate_known_encodings():
    """Ensure all known encodings are valid numpy arrays."""
    global known_face_encodings, known_face_names
    valid_encodings = []
    valid_names = []

    for encoding, name in zip(known_face_encodings, known_face_names):
        if isinstance(encoding, np.ndarray):
            valid_encodings.append(encoding)
            valid_names.append(name)
        else:
            print(f"Invalid encoding detected for {name}. Skipping.")

    known_face_encodings = valid_encodings
    known_face_names = valid_names

def update_metadata(image_path, names):
    """Update the metadata with the names of the recognized faces."""
    try:
        info = iptcinfo3.IPTCInfo(image_path)
        current_keywords = info.keywords or []
        new_keywords = list(set(current_keywords + names))

        # Only update if names have changed
        if set(current_keywords) != set(new_keywords):
            info["keywords"] = new_keywords
            info.save()
            print(f"Updated metadata for {os.path.basename(image_path)} with names: {new_keywords}")
        else:
            print(f"No update needed for {os.path.basename(image_path)}")
    except Exception as e:
        print(f"Error updating metadata for {os.path.basename(image_path)}: {e}")

def add_face_encoding(face_image, name):
    """Add a new face encoding to the known list, ensuring duplicates are avoided."""
    global known_face_encodings, known_face_names
    try:
        face_encodings = face_recognition.face_encodings(face_image)
        if len(face_encodings) > 0:
            face_encoding = face_encodings[0]
            if name not in known_face_names:
                known_face_encodings.append(face_encoding)
                known_face_names.append(name)
                print(f"Added '{name}' to known encodings.")
            else:
                print(f"'{name}' already in known encodings. Skipping...")
        else:
            print("No encodings found for the provided face image.")
    except Exception as e:
        print(f"Error while adding face encoding for '{name}': {e}")

def process_unrecognized_faces():
    """Find and process unrecognized faces in the photos directory."""
    unrecognized_faces = []

    # Look for unrecognized faces
    for image_file in os.listdir(photos_dir):
        image_path = os.path.join(photos_dir, image_file)
        if image_file.endswith(('.jpg', '.jpeg')):
            print(f"Processing image: {image_file}")
            image = face_recognition.load_image_file(image_path)
            face_locations = face_recognition.face_locations(image)
            print(f"Found {len(face_locations)} face(s) in {image_file}")

            if len(face_locations) > 0:
                for i, face_location in enumerate(face_locations):
                    top, right, bottom, left = face_location
                    cropped_face = image[top:bottom, left:right]
                    cropped_face_path = os.path.join(cropped_faces_dir, f"face_{i}.jpg")
                    cropped_face_image = Image.fromarray(cropped_face)

                    # Save cropped face
                    cropped_face_image.save(cropped_face_path)
                    print(f"Saved unrecognized face to {cropped_face_path}")

                    unrecognized_faces.append(cropped_face_path)
            else:
                print(f"No faces detected in {image_file}")

    return unrecognized_faces

def main():
    print("Loading training data...")

    # Validate encodings before proceeding
    validate_known_encodings()

    unrecognized_faces = process_unrecognized_faces()

    # If there are any unrecognized faces, ask the user to identify them
    if unrecognized_faces:
        for face_image_path in unrecognized_faces:
            if not os.path.exists(face_image_path):
                print(f"Cropped face file {face_image_path} not found. Skipping...")
                continue

            print(f"Enter the name for the face in {face_image_path} (or type 'unknown' to skip): ")
            name = input().strip()

            if name != 'unknown' and name:
                # Add new face encoding to training data
                face_image = face_recognition.load_image_file(face_image_path)
                add_face_encoding(face_image, name)

                # Move the cropped face image to the training data directory
                person_dir = os.path.join(training_data_dir, name)
                os.makedirs(person_dir, exist_ok=True)
                face_filename = os.path.basename(face_image_path)
                shutil.move(face_image_path, os.path.join(person_dir, face_filename))

                print(f"Moved {face_image_path} to {os.path.join(person_dir, face_filename)}")

                # Retrain the model with updated encodings
                print("Retraining the model...")
                with open(encodings_file, "wb") as f:
                    pickle.dump((known_face_encodings, known_face_names), f)
                print("Model updated successfully.")

    # Update metadata for all images
    print("Updating metadata for existing images...")
    for image_file in os.listdir(photos_dir):
        image_path = os.path.join(photos_dir, image_file)
        if image_file.endswith(('.jpg', '.jpeg')):
            image = face_recognition.load_image_file(image_path)
            face_locations = face_recognition.face_locations(image)
            face_names = []

            for face_location in face_locations:
                top, right, bottom, left = face_location
                face_encoding = face_recognition.face_encodings(image, [face_location])[0]

                # Check if this face matches any known encodings
                try:
                    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                    if True in matches:
                        first_match_index = matches.index(True)
                        face_names.append(known_face_names[first_match_index])
                    else:
                        face_names.append("unknown")
                except Exception as e:
                    print(f"Error during face comparison: {e}")

            if face_names:
                update_metadata(image_path, face_names)

    print("Cleaning up cropped_faces directory...")
    for file in os.listdir(cropped_faces_dir):
        file_path = os.path.join(cropped_faces_dir, file)
        if os.path.isfile(file_path):
            os.remove(file_path)
            print(f"Deleted leftover file: {file_path}")

if __name__ == "__main__":
    main()
