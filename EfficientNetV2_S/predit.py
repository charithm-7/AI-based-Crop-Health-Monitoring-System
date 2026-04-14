import tensorflow as tf
import numpy as np
from keras.preprocessing import image
import os

MODEL_PATH = "best_model.h5"
IMG_SIZE = (224, 224)

# load model
model = tf.keras.models.load_model(MODEL_PATH)

# class labels (auto from train folder order)
CLASS_DIR = "../dataset_split/train"
class_names = sorted(os.listdir(CLASS_DIR))

def predict_image(img_path):
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img = image.img_to_array(img)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img)
    return class_names[np.argmax(pred)]

# example
img_path = "sample_leaf.jpg"
print("Prediction:", predict_image(img_path))
