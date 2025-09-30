import os
import pandas as pd
import shutil

# Define paths
base_path = 'c:/Users/danie/Desktop/capstone project/train_images'
csv_path = 'c:/Users/danie/Desktop/capstone project/train.csv'

# Read the CSV file
train_data = pd.read_csv(csv_path)

# Create folders for labels
for label in ['0', '1', '2', '3', '4']:
    label_folder = os.path.join(base_path, label)
    os.makedirs(label_folder, exist_ok=True)

# Move images to corresponding label folders
for _, row in train_data.iterrows():
    image_name = row['id_code']  # Corrected column name
    label = str(row['diagnosis'])
    source_path = os.path.join(base_path, f"{image_name}.png")  # Assuming images have .png extension
    destination_path = os.path.join(base_path, label, f"{image_name}.png")
    if os.path.exists(source_path):
        shutil.move(source_path, destination_path)

print("Images have been moved to their respective label folders.")