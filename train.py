
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

import pandas as pd
import os
import cv2
import joblib
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm
import argparse
from pathlib import Path

ROOT =  Path.cwd()

def load_and_preprocess_data(folder_path:str, categories:list)-> tuple[np.array, np.array]:
    """
    Loads images from a folder, generates HSV histograms, and assigns labels based on folder name.

    Parameters:
    - folder_path: Path to the main directory containing subfolders of images.

    Returns:
    - features: A NumPy array of flattened all images.
    - targets: A NumPy array of labels corresponding to the images.
    """
    # Initialize empty lists to store histograms and labels
    features, target = [], []
    # Iterate through each folder in the given directory
    for i in categories:
        print(f'loading... category: {i}')
        path = os.path.join(folder_path, i)
        for img in tqdm(os.listdir(path), desc=f'Loading category {i}'):
            img_array = cv2.imread(os.path.join(path, img))
            padded_img = img_array

            if img_array.shape[0] > img_array.shape[1]:
                padded_img = 255 * np.ones((img_array.shape[0], img_array.shape[0], 3), dtype=float)
                padded_img[:, :img_array.shape[1], :] = img_array
            elif img_array.shape[0] < img_array.shape[1]:
                padded_img = 255 * np.ones((img_array.shape[1], img_array.shape[1], 3), dtype=float)
                padded_img[:img_array.shape[0], :, :] = img_array
            else:
                padded_img = 255 * np.ones((img_array.shape[1], img_array.shape[1], 3), dtype=float)
                padded_img = img_array

            resized_img = cv2.resize(padded_img, (150, 150))
            features.append(resized_img.flatten())
            target.append(categories.index(i))
        print(f'loaded category: {i} successfully')

    # Convert the lists of features and targets to NumPy arrays
    features = np.array(features)
    targets = np.array(target)

    return features, targets

def parse_arguments():
    """Parses command-line arguments for SVM model testing.

    Returns:
        Namespace: An object containing parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Train an SVM model for Orange Ripeness Detection")
    parser.add_argument(
        '-d','--dataset-folder',
        type=str,
        default=os.path.join(ROOT, 'dataset'),
        help='Path to the dataset folder'
    )
    parser.add_argument(
        '--kernel',
        type=str,
        default='poly',
        help='Kernel type for SVM'
    )
    parser.add_argument(
        '--C',
        type=float,
        default=0.1,
        help='Regularization parameter'
    )
    parser.add_argument(
        '--gamma',
        type=float,
        default=0.0001,
        help='Kernel coefficient for RBF'
    )
    parser.add_argument(
        '--model-filename',
        type=str,
        default='best_svc_model.dat',
        help='Filename to save the trained SVM model'
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


def main(args):
    """
    Main entry point for the script, handling data loading, training, and evaluation.

    Parameters:
    - args: Parsed arguments from the command line.
    """
    # Generate histograms and corresponding labels from the input folder
    features, target = load_and_preprocess_data(args.dataset_folder, args.categories)

    # Dataframe
    df = pd.DataFrame(features)
    df['Target'] = target
    # Input data
    x = df.iloc[:, :-1]
    # Output data
    y = df.iloc[:, -1]
    x_train, x_test, y_train, y_test = train_test_split(x, 
                                                        y, 
                                                        test_size=0.20, 
                                                        random_state=77, 
                                                        stratify=y
                                                        )
    print('Start training... ')
    print(f'C: {args.C}\ngamma: {args.gamma}\nkernel: {args.kernel}')
    svc = svm.SVC(C=args.C, 
                  gamma=args.gamma, 
                  kernel=args.kernel, 
                  probability=True
                  )
    
    svc.fit(x_train, y_train)
    # Evaluate the model
    y_pred = svc.predict(x_test)
    print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
    print(f'Classification Report:\n{classification_report(y_test, y_pred)}')
    # Save the model
    joblib.dump(svc, 'svm_model.dat')
    print('The model was saved in svc_model.dat')


if __name__ == "__main__":
    args = parse_arguments()
    main(args)


