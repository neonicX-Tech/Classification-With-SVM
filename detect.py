import joblib
import pandas as pd
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Load the saved model
best_svc = joblib.load('best_svc_model.pkl')
Categories = ['ripe', 'unripe']
# Load the parameters
with open('best_params.txt', 'r') as file:
    best_params = eval(file.read())

# Function to preprocess a single image
def preprocess_image(img_path):
    img_array = cv2.imread(img_path)
    newimg = img_array

    if img_array.shape[0] > img_array.shape[1]:
        newimg = 255 * np.ones((img_array.shape[0], img_array.shape[0], 3), dtype=float)
        newimg[:, :img_array.shape[1], :] = img_array
    elif img_array.shape[0] < img_array.shape[1]:
        newimg = 255 * np.ones((img_array.shape[1], img_array.shape[1], 3), dtype=float)
        newimg[:img_array.shape[0], :, :] = img_array
    else:
        newimg = 255 * np.ones((img_array.shape[1], img_array.shape[1], 3), dtype=float)
        newimg = img_array

    img_resized = cv2.resize(newimg, (150, 150))
    flat_data = img_resized.flatten()
    return flat_data

# Example usage: predicting the label of a new image
img_path = '/content/koko-limetti-valkoisella-taustalla56.jpg'
flat_data = preprocess_image(img_path)
prediction = best_svc.predict([flat_data])
print(f'Predicted label: {Categories[prediction[0]]}')