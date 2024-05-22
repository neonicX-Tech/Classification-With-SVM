import os
import cv2
import numpy as np
import argparse
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

def generate_hsv_histogram(image, hue_bins=8, saturation_bins=2, value_bins=4):
    """
    Generate a normalized HSV histogram for an image.

    Args:
    image (np.array): The input image in BGR format.
    hue_bins (int): Number of bins for the Hue channel.
    saturation_bins (int): Number of bins for the Saturation channel.
    value_bins (int): Number of bins for the Value channel.

    Returns:
    np.array: The flattened and normalized HSV histogram.
    """
    # Convert the image from BGR to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Define the number of bins and the range for each channel
    hist_size = [hue_bins, saturation_bins, value_bins]
    hist_range = [0, 180, 0, 256, 0, 256]

    # Calculate the histogram
    hist = cv2.calcHist([hsv_image], [0, 1, 2], None, hist_size, hist_range)
    
    # Normalize the histogram
    hist = cv2.normalize(hist, hist).flatten()
    
    return hist

def generate_histograms_from_folder(folder_path):
    """
    Generates HSV histograms for all images in a given directory and assigns labels based on the folder name.

    Parameters:
    - folder_path: Path to the main directory containing subfolders of images.

    Returns:
    - histograms: A NumPy array of histograms for all images.
    - labels: A NumPy array of labels corresponding to the histograms.
    """

    # Initialize empty lists to store histograms and labels
    histograms = []
    labels = []

    # Iterate through each folder in the given directory
    for folder in os.listdir(folder_path):
        folder_label = folder
        # Set label to 1 if folder name is "ripe", otherwise set it to 0
        label = 1 if folder_label == "ripe" else 0
        folder_path_full = os.path.join(folder_path, folder)

        # Iterate through each file in the current folder
        for filename in os.listdir(folder_path_full):
            image_path = os.path.join(folder_path_full, filename)
            image = cv2.imread(image_path)

            # Check if the image was successfully read
            if image is None:
                print(f"Error: Image '{filename}' not found!")
                continue

            # Generate the HSV histogram for the image
            hist = generate_hsv_histogram(image)
            # Append the histogram and label to their respective lists
            histograms.append(hist)
            labels.append(label)

    # Convert the lists of histograms and labels to NumPy arrays
    histograms = np.array(histograms)
    labels = np.array(labels)

    # Return the histograms and labels as NumPy arrays
    return histograms, labels


def train_svm(features, labels, C=1.0, gamma=0.1, kernel=cv2.ml.SVM_RBF):
    """
    Trains an SVM (Support Vector Machine) model using the provided features and labels.

    Parameters:
    - features: The training data, a NumPy array where each row is a feature vector.
    - labels: The labels corresponding to the training data, a NumPy array of integers.
    - C: The regularization parameter (default is 1.0). This parameter controls the trade-off between achieving a low training error and a low testing error.
    - gamma: The kernel coefficient for the RBF (Radial Basis Function) kernel (default is 0.1). It defines how far the influence of a single training example reaches.
    - kernel: The type of SVM kernel to be used (default is RBF). Other options include linear, polynomial, and sigmoid kernels.

    Returns:
    - svm: The trained SVM model.
    """
    
    # Create an SVM object using OpenCV's machine learning module
    svm = cv2.ml.SVM_create()
    
    # Set the kernel type for the SVM. In this case, the default is RBF (Radial Basis Function)
    svm.setKernel(kernel)
    
    # Set the SVM type to C-Support Vector Classification, which is the most common type of SVM
    svm.setType(cv2.ml.SVM_C_SVC)
    
    # Set the regularization parameter C, which controls the trade-off between achieving a low training error and a low testing error
    svm.setC(C)
    
    # Set the gamma parameter for the kernel function. For RBF, it defines the influence range of a single training example
    svm.setGamma(gamma)
    
    # Train the SVM with the provided features (training data) and labels (training labels)
    # cv2.ml.ROW_SAMPLE indicates that the samples are represented by rows in the feature matrix
    svm.train(features, cv2.ml.ROW_SAMPLE, labels)
    
    # Return the trained SVM model
    return svm



def evaluate_svm(svm, x_test, y_test, model_filename='svm_data_1.dat'):
    """
    Evaluates the performance of the SVM model on the test dataset and saves the trained model.

    Parameters:
    - svm: The trained SVM model.
    - x_test: The test data features.
    - y_test: The true labels for the test data.
    - model_filename: The filename to save the trained SVM model (default is 'svm_data_1.dat').
    """
    
    # Save the trained SVM model to a file
    svm.save(model_filename)
    
    # Use the SVM model to predict the labels for the test data
    _, y_pred = svm.predict(x_test)
    
    # Convert the predicted labels to integers and flatten the array
    y_pred = y_pred.astype(int).flatten()
    
    # Flatten the true labels array
    y_test = y_test.flatten()

    # Calculate the accuracy of the model
    accuracy = np.mean(y_pred == y_test) * 100.0
    print(f"Accuracy: {accuracy:.2f}%")

    # Print the classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Print the confusion matrix
    print("\nConfusion Matrix:")
    conf_matrix = confusion_matrix(y_test, y_pred)
    print(conf_matrix)



def main(args):
    # Get the input path for the dataset folder from the arguments
    input_path = args.dataset_folder
    
    # Generate histograms and corresponding labels from the input folder
    histograms, extracted_labels = generate_histograms_from_folder(input_path)
    print(histograms.shape)
    # Split the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(histograms, extracted_labels, test_size=0.2, random_state=42)
    print(f"Training set size: {x_train.shape}, Testing set size: {x_test.shape}")
    
    # Train the SVM model using the training data
    svm_model = train_svm(x_train, y_train, C=args.C, gamma=args.gamma, kernel=args.kernel)
    
    # Evaluate the trained SVM model using the testing data
    evaluate_svm(svm_model, x_test, y_test, model_filename=args.model_filename)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an SVM model for Orange Ripeness Detection")
    parser.add_argument('-d','--dataset_folder', default="/content/drive/MyDrive/orange_dataset", type=str, help='Path to the dataset folder')
    parser.add_argument('--kernel', type=str, default=cv2.ml.SVM_RBF, help='Kernel type for SVM')
    parser.add_argument('--C', type=float, default=1.0, help='Regularization parameter')
    parser.add_argument('--gamma', type=float, default=0.1, help='Kernel coefficient for RBF')
    parser.add_argument('--model_filename', type=str, default='svm_data_1.dat', help='Filename to save the trained SVM model')

    args = parser.parse_args()
    main(args)
