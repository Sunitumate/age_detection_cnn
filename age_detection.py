import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Set dataset path
dataset_path = "UTKFace/"

# Load images and labels
images = []
ages = []

for file in tqdm(os.listdir(dataset_path)):
    if file.endswith(".jpg"):
        try:
            age = int(file.split("_")[0])  # Extract age from filename
            img_path = os.path.join(dataset_path, file)
            img = cv2.imread(img_path)
            img = cv2.resize(img, (128, 128))  # Resize
            img = img / 255.0  # Normalize
            images.append(img)
            ages.append(age)
        except:
            continue

# Convert to numpy arrays
images = np.array(images)
ages = np.array(ages)

# Split dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(images, ages, test_size=0.2, random_state=42)

# Load MobileNetV2 model
base_model = MobileNetV2(input_shape=(128, 128, 3), include_top=False, weights='imagenet')

# Freeze layers
for layer in base_model.layers:
    layer.trainable = False

# Add custom layers
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
x = Dense(64, activation='relu')(x)
x = Dense(1, activation='linear')  # Regression output for age prediction

# Compile model
model = Model(inputs=base_model.input, outputs=x)
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Train model
model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=32, epochs=10)

# Save model
model.save("age_detection_model.h5")
print("Model training complete and saved as age_detection_model.h5")
