
# Orange Ripeness Classifier

This project contains machine learning aimed at classifying images of oranges into ripe and unripe categories using the Support Vector Machine (SVM) algorithm. SVM is a powerful supervised learning algorithm commonly used for classification tasks.
Oranges undergo color changes as they ripen, making it feasible to classify their ripeness based on image analysis. This project utilizes SVM, a supervised machine learning algorithm, to classify oranges as ripe or unripe by training on a dataset of labeled orange images.


*************

## Table of Contents

- [Orange Classifier](#Orange Classifier)
  - [Overview](#Overview)
  - [Features](#features)
    - [Efficiency](#Efficiency)
  - [Example](#example)
  - [Getting Started](#getting-started)
    - [Prerequisites](#prerequisites)
    - [Installation](#installation)
  - [Usage](#usage)
  - [Contributing](#contributing)
  - [License](#license)
  - [About Us](#about-us)


***************
## Overview

The goal of this project is to develop a robust model capable of accurately distinguishing between ripe and unripe oranges based on their visual characteristics in images.
The SVM algorithm is chosen for its effectiveness in binary classification tasks and its ability to handle high-dimensional feature spaces.

**********
## Features

Support Vector Machines (SVMs) offer several benefits for image classification.
Support Vector Machines (SVMs) excel in image classification due to their ability to effectively handle high-dimensional data, prevent overfitting, and generalize well with small to medium-sized datasets.
SVMs are memory-efficient and offer interpretable decision boundaries, making them a powerful choice for image classification tasks.


# Efficiency
Support Vector Machine (SVM) is known for its efficiency in handling high-dimensional feature spaces and its effectiveness in binary classification tasks. When it comes to image classification for detecting ripe and unripe oranges, SVM's efficiency depends on several factors, including the choice of features and the quality of the dataset.
Here's how SVM can efficiently utilize the features mentioned in the Features section for classifying oranges:
- **`Color Features:** SVM can efficiently learn to separate ripe and unripe oranges based on their color distributions. Color histograms provide a compact representation of color information, making them computationally efficient for SVM classification.

- **`Texture Features:`** While texture descriptors like Local Binary Patterns (LBP) or Gray-Level Co-occurrence Matrix (GLCM) statistics can increase the dimensionality of the feature space, SVM is well-suited for handling such high-dimensional data efficiently. SVM can efficiently learn decision boundaries in complex feature spaces, enabling effective utilization of texture features for orange classification.

- **`Shape Features:`** Shape features such as area, perimeter, and circularity are relatively simple and computationally inexpensive to compute. SVM's efficiency in handling high-dimensional feature spaces allows it to incorporate shape features without significant computational overhead.

- **`Size Features:`** Size features like bounding box dimensions or area are straightforward to compute and add minimal computational burden to SVM classification. SVM can efficiently utilize size features to distinguish between ripe and unripe oranges based on their size differences.

- **`Edge Features:`** Features derived from edge detection techniques can enhance SVM's ability to capture detailed structural information about oranges. While edge features may increase the dimensionality of the feature space, SVM's efficiency in handling high-dimensional data allows it to effectively utilize edge features for classification.

- **`Statistical Features:`** Descriptive statistics of pixel intensities provide valuable information about the overall distribution of pixel values in the images. SVM's efficiency in processing numerical data enables it to efficiently incorporate statistical features into the classification process.

- **`Spatial Features:`** Spatial features capturing relationships between pixels or regions in the images can be efficiently processed by SVM. SVM's kernel functions allow it to capture complex spatial relationships in the data, making it effective in utilizing spatial features for classification.

******
# Description

This code is for training an image classification model using Support Vector Machines (SVMs) to classify images of ripe and unripe oranges. 

1. It loads images of ripe and unripe oranges from specified directories, resizes them to a standard size, flattens the pixel values into a 1D array, and assigns labels (0 for ripe oranges, 1 for unripe oranges).

2. The data is split into training and testing sets.

3. GridSearchCV is used to find the best hyperparameters for the SVM model by performing a grid search over a predefined parameter grid.

4. The best model is trained using the training data.

5. The model's accuracy is evaluated using the testing data, and a classification report is generated to assess its performance.

6. The best model is saved for future use.

7. Finally, the saved model is loaded, and an example image is provided to demonstrate how the model predicts whether the given orange is ripe or unripe, along with the probability scores for each class.

****
Getting Started

# Example
our dtatset beging with under suppoertd light and train it othis below show some example of that 



***********
# Getting Started

This code is for training a Support Vector Machine (SVM) model to classify images of ripe and unripe oranges. Here's a breakdown of what it does:

Data Preparation:
It loads images of ripe and unripe oranges from specified directories.
Each image is resized to a standard size of 150x150 pixels and flattened into a 1D array.
The pixel values are stored in flat_data_arr, and corresponding labels (0 for ripe, 1 for unripe) are stored in target_arr.

Data Splitting:
The data is split into training and testing sets using train_test_split() function from sklearn.model_selection.

Parameter Grid Definition:
A parameter grid is defined for the SVM model using different values of regularization parameter (C), kernel coefficient (gamma), and kernel type (rbf or poly).

Model Training:
A SVM classifier with probability estimation enabled (probability=True) is created.
GridSearchCV is used to find the best combination of hyperparameters by exhaustively searching over the parameter grid.

Model Evaluation:
The best model is evaluated using the testing data.
Accuracy and a classification report showing precision, recall, and F1-score for each class are printed.

Model Saving and Loading:
The best model is saved using joblib.dump() for future use.
The saved model is loaded back into memory using joblib.load().

Prediction:
An example image is read and resized for prediction.
The model predicts whether the given orange is ripe or unripe, along with the probability scores for each class.
