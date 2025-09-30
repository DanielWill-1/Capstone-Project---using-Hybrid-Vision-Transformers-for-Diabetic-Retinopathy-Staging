import os

# Define the path to the train_images folder
train_images_path = 'c:/Users/danie/Desktop/capstone project/train_images'

# Count the number of image files in the folder
image_count = len([file for file in os.listdir(train_images_path) if os.path.isfile(os.path.join(train_images_path, file))])

# Print the result
print(f'Total number of images in the folder: {image_count}')
