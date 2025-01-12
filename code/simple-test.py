import os
from iptcinfo3 import IPTCInfo

# Paths
PHOTO_DIR = "../data/photos/"

def list_metadata_for_photos():
    """
    List all photos in the /photos directory and print their metadata keywords.
    """
    print("Listing photos with metadata names:\n")

    for image_name in os.listdir(PHOTO_DIR):
        # Only process files with '.jpeg' or '.jpg' extensions
        if image_name.lower().endswith(".jpeg") or image_name.lower().endswith(".jpg"):
            image_path = os.path.join(PHOTO_DIR, image_name)
            print(f"Processing file: {image_name}")  # Debugging step

            # Check if file exists
            if os.path.isfile(image_path):
                try:
                    info = IPTCInfo(image_path, force=True)

                    # Get the 'keywords' tag from metadata
                    keywords = info['keywords']  # This is the correct way to access metadata
                    if keywords:
                        print(f"Keywords for {image_name}: {keywords}")
                    else:
                        print(f"No keywords found for {image_name}")

                except Exception as e:
                    print(f"Failed to read metadata for {image_name}: {e}")
            else:
                print(f"Skipping {image_name}: Not a valid file.")

if __name__ == "__main__":
    try:
        list_metadata_for_photos()
    except Exception as e:
        print(f"Unexpected error: {e}")
