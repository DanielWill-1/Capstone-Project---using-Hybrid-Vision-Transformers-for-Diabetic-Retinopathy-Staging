import os
import csv

# Define the directory containing the images
directory = "C:/Users/danie/Desktop/capstone project/OCT-AND-EYE-FUNDUS-DATASET-main/OCT-AND-EYE-FUNDUS-DATASET-main/OCT/normal_sampled/normal_sampled"

# Define the output CSV file
output_csv = 'image_labels.csv'

# Collect image names and labels
image_data = []
for filename in os.listdir(directory):
    if filename.endswith('.png') or filename.endswith('.jpg'):
        image_name = os.path.splitext(filename)[0]  # Remove file extension
        image_data.append([image_name, 0])  # Append image name and label 0

# Write to CSV file
with open(output_csv, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['ImageName', 'Label'])  # Write header
    writer.writerows(image_data)  # Write image data

print(f"CSV file '{output_csv}' created with {len(image_data)} entries.")