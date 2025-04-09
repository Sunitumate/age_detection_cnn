import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Path to dataset
DATASET_PATH = "../datasets/UTKFace/"

def load_utkface_data():
    images, ages = [], []
    for img_name in tqdm(os.listdir(DATASET_PATH)):
        try:
            age = int(img_name.split("_")[0])  # Extract age from filename
            img_path = os.path.join(DATASET_PATH, img_name)

            img = cv2.imread(img_path)
            img = cv2.resize(img, (128, 128))  # Resize image
            img = img / 255.0  # Normalize image

            images.append(img)
            ages.append(age)
        except:
            continue

    return np.array(images), np.array(ages)

X, y = load_utkface_data()

# Split into training & validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Dataset Loaded: Training: {X_train.shape}, Validation: {X_val.shape}")
