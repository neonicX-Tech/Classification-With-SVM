import os
import cv2
import argparse
from google.colab.patches import cv2_imshow
import numpy as np

def generate_hsv_histogram(image):
    """
    Generates an HSV histogram for the given image.
    
    Parameters:
    - image: Input image in BGR format.
    
    Returns:
    - hist: Flattened and normalized histogram.
    """
    # Convert the image from BGR to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Define the number of bins for hue, saturation, and value
    hue_bins = 8
    saturation_bins = 2
    value_bins = 4
    # Calculate the histogram for the HSV image
    hist = cv2.calcHist([hsv_image], [0, 1, 2], None, [hue_bins, saturation_bins, value_bins], [0, 180, 0, 256, 0, 256])
    # Normalize and flatten the histogram
    hist = cv2.normalize(hist, hist).flatten()
    return hist

def generate_histogram_from_image(image_path):
    """
    Generates an HSV histogram from an image file.
    
    Parameters:
    - image_path: Path to the image file.
    
    Returns:
    - hist: Flattened and normalized histogram.
    """
    # Read the image from the file path
    image = cv2.imread(image_path)
    # Check if the image was successfully read
    if image is None:
        raise FileNotFoundError(f"Error: Image '{image_path}' not found!")
    
    # Generate and return the HSV histogram
    return generate_hsv_histogram(image)

def test_svm_with_image(image_path, svm_model):
    """
    Tests the SVM model with a given image.
    
    Parameters:
    - image_path: Path to the image file.
    - svm_model: Trained SVM model.
    
    Returns:
    - predicted_label: Predicted label for the image (0 or 1).
    """
    try:
        # Generate histogram descriptors from the image
        test_descriptors = generate_histogram_from_image(image_path)
        # Predict the label using the SVM model
        _, predicted_label = svm_model.predict(test_descriptors.reshape(1, -1))
        # Return the predicted label as an integer
        return int(predicted_label[0, 0])
    except FileNotFoundError as e:
        # Handle the case where the image file is not found
        print(e)
        return None

def main(args):
    """
    Main function to test the SVM model with a new image.
    
    Parameters:
    - args: Command-line arguments.
    """
    # Load the trained SVM model from the specified file
    svm_model = cv2.ml.SVM_load(args.model_filename)
    
    # Test the SVM model with the provided image and get the predicted label
    predicted_label = test_svm_with_image(args.image_path, svm_model)
    if predicted_label is not None:
        # Define labels for interpretation (0 corresponds to "unripe" and 1 to "ripe")
        labels = ["unripe", "ripe"]
        # Print the predicted label
        print("Predicted Label:", labels[predicted_label])
    

if __name__ == "__main__":
    # Create an argument parser for the command-line interface
    parser = argparse.ArgumentParser(description="Test an SVM model with a new image for Orange Ripeness Detection")
    # Define the --image_path argument to specify the path to the test image file
    parser.add_argument('-i', '--image_path', type=str, required=True, help='Path to the test image file')
    # Define the --model_filename argument to specify the filename of the trained SVM model
    parser.add_argument('--model_filename', type=str, default='svm_data_1.dat', help='Filename of the trained SVM model to load')

    # Parse the command-line arguments
    args = parser.parse_args()
    # Call the main function with the parsed arguments
    main(args)
