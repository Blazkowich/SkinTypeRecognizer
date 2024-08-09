import cv2
import numpy as np
from tensorflow.keras.models import load_model

img_width, img_height = 150, 150
model_path = r'train/skin_type_model.h5'
model = load_model(model_path)


def predict_image(image):
    image = cv2.resize(image, (img_width, img_height))
    image = np.expand_dims(image, axis=0) / 255.0
    prediction = model.predict(image)[0]
    return prediction
