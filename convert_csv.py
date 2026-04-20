import pandas as pd
import numpy as np
import cv2
import os

print("🚀 Starting conversion...")

emotions = ['angry','disgust','fear','happy','sad','surprise','neutral']

# Create folders
for folder in ['train', 'test']:
    for emotion in emotions:
        os.makedirs(f'dataset/{folder}/{emotion}', exist_ok=True)

print("📁 Folders created")

# Load CSV
data = pd.read_csv('dataset/fer2013.csv')
print("📄 CSV loaded")

# Convert
for index, row in data.iterrows():
    pixels = np.array(row['pixels'].split(), dtype='uint8')
    image = pixels.reshape(48, 48)

    emotion = emotions[row['emotion']]

    if row['Usage'] == 'Training':
        folder = 'train'
    else:
        folder = 'test'

    file_path = f'dataset/{folder}/{emotion}/{index}.jpg'
    cv2.imwrite(file_path, image)

    if index % 5000 == 0:
        print(f"Processed {index} images")

print("✅ DONE!")