import os
from PIL import Image
import shutil

# Example: convert and resize input images to YOLO-compatible folder
source_folder = 'raw_images/'
target_folder = 'data/train/images/'

os.makedirs(target_folder, exist_ok=True)

for file_name in os.listdir(source_folder):
    if file_name.endswith('.jpg'):
        img = Image.open(os.path.join(source_folder, file_name))
        resized = img.resize((640, 640))
        resized.save(os.path.join(target_folder, file_name))

print("Image preprocessing completed.")
