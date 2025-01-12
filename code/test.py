import os
from iptcinfo3 import IPTCInfo

# Path to the photos directory
PHOTO_DIR = "../data/photos/"

def list_photos_with_metadata():
    # White list for .jpg and .jpeg files
    valid_extensions = (".jpg", ".jpeg")
    
    print("Listing photos with metadata names:")

    for root, dirs, files in os.walk(PHOTO_DIR):
        for file in files:
            if file.lower().endswith(valid_extensions):
                file_path = os.path.join(root, file)
                print(f"Processing file: {file_path}")

                try:
                    # Open the file and load its metadata
                    info = IPTCInfo(file_path)

                    # Retrieve the keywords from the metadata
                    keywords = info['keywords']
                    print(f"Keywords for {file_path}: {keywords}")
                
                except Exception as e:
                    print(f"Failed to read metadata for {file_path}: {e}")

def check_for_temporary_files():
    # Check for files ending with '~' in the photos directory
    print("Checking for temporary files in /photos directory...")

    for root, dirs, files in os.walk(PHOTO_DIR):
        for file in files:
            if file.endswith("~"):
                print("FAILED: Temporary files in /photos directory found. Cleanup failed.")
                return

    print("No temporary files found. Cleanup successful.")

if __name__ == "__main__":
    list_photos_with_metadata()
    check_for_temporary_files()
