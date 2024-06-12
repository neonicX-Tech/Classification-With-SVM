"""
Recognet-Oranges-With-SVM
Copyright (C) <2024/5/23> <neonicX-Tech>

Orange_Detected is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public
License as published by the Free Software Foundation; either
version 2.1 of the License, or any later version.

This library is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public
License along with this library; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
"""

import os
import cv2
import argparse
import numpy as np


from pathlib import Path

ROOT =  Path.cwd()

def generate_hsv_histogram(image:np.ndarray) -> np.ndarray:
    """
    Generates a flattened and normalized 2D histogram (hue and saturation) 
    for the given image assumed to be in BGR format.

    Parameters:
        image (np.ndarray): Input image in BGR format.

    Returns:
        np.ndarray: Flattened and normalized 3D histogram.
    """

    # Convert the image from BGR to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define the number of bins for hue, saturation, and value
    num_hue_bins = 8
    saturation_range_bins = 2
    value_intensity_bins = 4

    # Calculate the histogram for the HSV image
    hist = cv2.calcHist([hsv_image], [0, 1, 2], None, [num_hue_bins, saturation_range_bins, value_intensity_bins], [0, 180, 0, 256, 0, 256])

    # Normalize and flatten the histogram
    hist = cv2.normalize(hist, hist).flatten()
    return hist

def test_svm_with_image(image_path:str, svm_model:cv2.ml.SVM) -> int:
    """
    Tests the SVM model with a given image.

    Args:
        image_path (str): Path to the image file.
        svm_model (cv2.ml.SVM): Trained SVM model.

    Returns:
        int: Predicted label for the image (0 or 1), or None if an error occurs.
    """
    
    try:
        # Load the image
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Generate histogram features from the image
        hsv_histogram_features = generate_hsv_histogram(image)

        # Predict the label using the SVM model
        _, predicted_label = svm_model.predict(hsv_histogram_features.reshape(1, -1))

        # Return the predicted label as an integer
        return int(predicted_label[0, 0])

    except (FileNotFoundError, Exception) as e:
        print(f"Error processing image: {e}")
        return None


def parse_arguments():
    """Parses command-line arguments for SVM model testing.

    Returns:
        Namespace: An object containing parsed arguments.
    """

    parser = argparse.ArgumentParser(description="Test an SVM model with a new image for Orange Ripeness Detection")
    parser.add_argument(
        '-i', '--image-path', type=str, required=True, help='Path to the test image file'
    )
    parser.add_argument(
        '--model-filename',
        type=str,
        default=os.path.join(ROOT, 'svm_data.dat'), 
        help='Filename of the trained SVM model to load (default: ROOT/svm_data.dat)'
    )
    return parser.parse_args()

def main(args) -> None:
    """
    Main function to test the SVM model with a new image.

    Args:
        args (Namespace): Object containing parsed arguments.
    """

    # Load the trained SVM model from the specified file
    try:
        svm_model = cv2.ml.SVM_load(args.model_filename)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Test the SVM model with the provided image and get the predicted label
    predicted_label = test_svm_with_image(args.image_path, svm_model)

    if predicted_label is not None:
        # Define labels for interpretation (0 corresponds to "unripe" and 1 to "ripe")
        labels = ["unripe", "ripe"]
        # Print the predicted label
        print("Predicted Label:", labels[predicted_label])
    else:
        # Handle cases where prediction fails (e.g., error during processing)
        print("Prediction failed. Check the image or model.")

    

if __name__ == "__main__":
    args = parse_arguments()
    main(args)