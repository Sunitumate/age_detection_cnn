import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Load trained model
model = keras.models.load_model("age_detection_model.h5")

def predict_age(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (128, 128))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    predicted_age = model.predict(img)[0][0]
    print(f"Predicted Age: {int(predicted_age)}")

# Test on a sample image
predict_age("UTKFace/sample_image.jpg")
