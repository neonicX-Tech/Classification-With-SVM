# Orange Ripeness Classifier

This project contains machine learning aimed at classifying images of oranges into ripe and unripe categories using the Support Vector Machine (SVM) algorithm. SVM is a powerful supervised learning algorithm commonly used for classification tasks.
Oranges undergo color changes as they ripen, making it feasible to classify their ripeness based on image analysis. This project utilizes SVM, a supervised machine learning algorithm, to classify oranges as ripe or unripe by training on a dataset of labeled orange images.
🍊🍈🍊🍈🍊🍈

*************

## Table of Contents

- [Orange Ripeness Classifier](#orange-ripeness-classifier)
  - [Overview](#overview)
  - [Features](#features)
    - [Efficiency](#efficiency)
  - [Example](#example)
  - [Usage](#Usage)
  - [Installation](#installation)
  - [Contributing](#contributing)
  - [License](#license)
  - [About Us](#about-us)

***************
## Overview

The goal of this project is to develop a robust model capable of accurately distinguishing between ripe and unripe oranges based on their visual characteristics in images.
The SVM algorithm is chosen for its effectiveness in binary classification tasks and its ability to handle high-dimensional feature spaces.📈📉
🍊🍈🍊
**********
## Features
🌎🚀
Support Vector Machines (SVMs) offer several benefits for image classification.
Support Vector Machines (SVMs) excel in image classification due to their ability to effectively handle high-dimensional data, prevent overfitting, and generalize well with small to medium-sized datasets.
SVMs are memory-efficient and offer interpretable decision boundaries, making them a powerful choice for image classification tasks.


# Efficiency
Support Vector Machine (SVM) is known for its efficiency in handling high-dimensional feature spaces and its effectiveness in binary classification tasks. When it comes to image classification for detecting ripe and unripe oranges, SVM's efficiency depends on several factors, including the choice of features and the quality of the dataset.

**Here's how SVM can efficiently utilize the features mentioned in the Features section for classifying oranges:**
- **`Color Features:`** SVM can efficiently learn to separate ripe and unripe oranges based on their color distributions. Color histograms provide a compact representation of color information, making them computationally efficient for SVM classification.

- **`Texture Features:`** While texture descriptors like Local Binary Patterns (LBP) or Gray-Level Co-occurrence Matrix (GLCM) statistics can increase the dimensionality of the feature space, SVM is well-suited for handling such high-dimensional data efficiently. SVM can efficiently learn decision boundaries in complex feature spaces, enabling effective utilization of texture features for orange classification.

- **`Shape Features:`** Shape features such as area, perimeter, and circularity are relatively simple and computationally inexpensive to compute. SVM's efficiency in handling high-dimensional feature spaces allows it to incorporate shape features without significant computational overhead.

- **`Size Features:`** Size features like bounding box dimensions or area are straightforward to compute and add minimal computational burden to SVM classification. SVM can efficiently utilize size features to distinguish between ripe and unripe oranges based on their size differences.

- **`Edge Features:`** Features derived from edge detection techniques can enhance SVM's ability to capture detailed structural information about oranges. While edge features may increase the dimensionality of the feature space, SVM's efficiency in handling high-dimensional data allows it to effectively utilize edge features for classification.

- **`Statistical Features:`** Descriptive statistics of pixel intensities provide valuable information about the overall distribution of pixel values in the images. SVM's efficiency in processing numerical data enables it to efficiently incorporate statistical features into the classification process.

- **`Spatial Features:`** Spatial features capturing relationships between pixels or regions in the images can be efficiently processed by SVM. SVM's kernel functions allow it to capture complex spatial relationships in the data, making it effective in utilizing spatial features for classification.

******
# Description
📊📈📉📊

For the classification of oranges, we use the scikit-learn library to create and train an SVM model. The project is designed to classify oranges as ripe or unripe based on extracted features.

The project involves the following steps:
1. **raining the SVM Model:** Using the scikit-learn library, we train an SVM model with the following parameters:
 - C: 0.1
 - gamma: 0.0001
 - kernel: 'poly'
2. **Evaluating Model Performance:** We evaluate the performance of the trained SVM model to ensure its accuracy and effectiveness in classifying the oranges.
3. **Testing on New Images:** The trained model is then tested on new images to validate its generalization capability.

******
# Example
our dataset begins with images captured under varying lighting conditions. Below are some examples of the images used in our dataset.

| Category | Image Example |
|----------|---------------|
| Ripe     | ![Ripe Orange](https://github.com/neonicX-Tech/Recognet-With-SVM/blob/svm-opencv/image_result/3.jpg)|
| Unripe   | ![Unripe Orange](https://github.com/neonicX-Tech/Recognet-With-SVM/blob/svm-opencv/image_result/5.jpg)|

**********
# Usage

1. Prerequisites:

- Python 3
- pip
- virtualenv

2. Clone the Project:
  Use the following command to clone the project repository from GitHub:

  `git clone https://github.com/neonicX-Tech/Classification-With-SVM.git -b svm-scikit-learn`

3. Set Up Development Environment:
  Activate a virtual environment to isolate project dependencies. Here's how to do it:

  ```bash
  cd Classification-With-SVM  # Navigate to the project directory
  virtualenv venv  # Create a virtual environment named venv
  source venv/bin/activate  # Activate the virtual environment in linux
  venv\Scripts\activate # Activate the virtual environment in windows
  ```
  Once activated, install the required dependencies listed in the project's `requirements.txt` file:

  `pip install -r requirements.txt`

4. Train the SVM Model:
  Train the SVM model on your dataset of orange images.
  
  `python train.py `

   Or replace `<dataset_folder>` with the actual path to the folder containing your training images:
   
   `python train.py --dataset-folder <dataset_folder>`

5. Detect Objects in Images:
  Use the trained model to detect oranges in a new image. Replace `<image_path>` with the path to the image you want to analyze:

  `python detect.py -i <image_path>`

  By default, this script only outputs the class names of objects detected in the image. To visualize the image with these detections overlaid, use the `--view` flag:

  `python detect.py -i <image_path> --view `

********
## Contributing

📌✉ Contact our LinkedIn: [https://www.linkedin.com/company/neonicx/about/]


*******
## License
🚀
[![License](https://img.shields.io/badge/License-GNU%20LGPL%20v2.1-blue.svg)](LICENSE)🚀

Recognet-Oranges-With-SVM is licensed under the GNU Lesser General Public License v2.1. See the [![License](https://img.shields.io/badge/License-GNU%20LGPL%20v2.1-blue.svg)](LICENSE) file for details.
For the closed-source version of DATASET-COMPRESSION-LIBRARY or commercial purposes, please contact us: contact (at) neonicx (dot) com.

*******

## About Us

📧📍 At neonicX, We redefine innovation in digital system design and hardware development. Our journey covers engineering from A to Z, providing complex end-to-end development solutions with our full-stack team.​​​​​​​
[https://www.linkedin.com/company/neonicx/about/]
