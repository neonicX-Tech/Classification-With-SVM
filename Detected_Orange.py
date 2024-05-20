import os
import cv2
import argparse
from google.colab.patches import cv2_imshow
def generate_hsv_histogram(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue_bins = 8
    saturation_bins = 2
    value_bins = 4
    hist = cv2.calcHist([hsv_image], [0, 1, 2], None, [hue_bins, saturation_bins, value_bins], [0, 180, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist

def generate_histogram_from_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Error: Image '{image_path}' not found!")
        
    return generate_hsv_histogram(image)

def test_svm_with_image(image_path, svm_model):
    try:
        test_descriptors = generate_histogram_from_image(image_path)
        _, predicted_label = svm_model.predict(test_descriptors.reshape(1, -1))
        return int(predicted_label[0, 0])
    except FileNotFoundError as e:
        print(e)
        return None

def main(args):
    svm_model = cv2.ml.SVM_load(args.model_filename)
    
    predicted_label = test_svm_with_image(args.image_path, svm_model)
    if predicted_label is not None:
        labels = ["unripe", "ripe"]
        print("Predicted Label:", labels[predicted_label])
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test an SVM model with a new image for Orange Ripeness Detection")
    parser.add_argument('-i','--image_path', type=str, required=True, help='Path to the test image file')
    parser.add_argument('--model_filename', type=str, default='svm_data_1.dat', help='Filename of the trained SVM model to load')

    args  = parser.parse_args()
    main(args)
