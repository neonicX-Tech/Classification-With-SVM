
"""
Recognet-Oranges-With-SVM
Copyright (C) <2024/7/30> <neonicX-Tech>

svm_orange_train is free software; you can redistribute it and/or
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

import joblib
import cv2
import os
import numpy as np
import argparse
from pathlib import Path

ROOT =  Path.cwd()

def draw(image_path:str, label:str ) -> None:
    try:
        # Load the image
        image = cv2.imread(image_path)
        # image = cv2.resize(image, (640, 480))
        if image is None:
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        image = cv2.putText(image, label, (15, 30), 1, 2, (140, 0, 0), 2, 1)
        cv2.imshow('Detected', image)
        if cv2.waitKey() == ord('q'): cv2.destroyAllWindows()
        
        return None

    except (FileNotFoundError, Exception) as e:
        print(f"Error processing image: {e}")
        return None


# Function to preprocess a single image
def preprocess_image(img_path):
    img = cv2.imread(img_path)
    padded_img = img

    if img.shape[0] > img.shape[1]:
        padded_img = 255 * np.ones((img.shape[0], img.shape[0], 3), dtype=float)
        padded_img[:, :img.shape[1], :] = img
    elif img.shape[0] < img.shape[1]:
        padded_img = 255 * np.ones((img.shape[1], img.shape[1], 3), dtype=float)
        padded_img[:img.shape[0], :, :] = img
    else:
        padded_img = 255 * np.ones((img.shape[1], img.shape[1], 3), dtype=float)
        padded_img = img

    resized_img = cv2.resize(padded_img, (150, 150))
    flatted_img = resized_img.flatten()
    return flatted_img

def parse_arguments():
    """Parses command-line arguments for SVM model testing.

    Returns:
        Namespace: An object containing parsed arguments.
    """

    parser = argparse.ArgumentParser(description="Test an SVM model with a new image for Orange Ripeness Detection")
    parser.add_argument(
        '-i', '--image-path', 
        type=str, 
        required=True, 
        help='Path to the test image file'
    )
    parser.add_argument(
        '--view', 
        action='store_true',
        help='show output image'
    )
    parser.add_argument(
        '--model-filename',
        type=str,
        default=os.path.join(ROOT, 'svm_model.dat'), 
        help='Filename of the trained SVM model to load (default: ROOT/best_svc_model.dat)'
    )
    parser.add_argument(
        '--categories',
        metavar='N',
        type=str,
        nargs='+',
        default=['ripe', 'unripe'], 
        help='list of trained model categories with a default of ["ripe", "unripe"].'
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
        model = joblib.load(args.model_filename)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    flatted_img = preprocess_image(args.image_path)
    prediction = model.predict([flatted_img])
    if prediction is not None:
        print(f'Predicted label: {args.categories[prediction[0]]}')
        if args.view:
            image = draw(args.image_path, args.categories[prediction[0]])
    else:
        # Handle cases where prediction fails (e.g., error during processing)
        print("Prediction failed. Check the image or model.")

if __name__ == "__main__":
    args = parse_arguments()
    main(args)