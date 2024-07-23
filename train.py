import pandas as pd
import os
import cv2
from skimage.transform import resize
from skimage.io import imread
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm
from google.colab.patches import cv2_imshow

Categories = ['ripe', 'unripe']
flat_data_arr = []  # input array
target_arr = []  # output array
datadir = 'dataset/'

for i in Categories:
    print(f'loading... category : {i}')
    path = os.path.join(datadir, i)
    for img in tqdm(os.listdir(path), desc=f'Loading category {i}'):
        img_array = cv2.imread(os.path.join(path, img))
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
        flat_data_arr.append(img_resized.flatten())
        target_arr.append(Categories.index(i))
    print(f'loaded category:{i} successfully')

flat_data = np.array(flat_data_arr)
target = np.array(target_arr)

# Dataframe
df = pd.DataFrame(flat_data)
df['Target'] = target

# Input data
x = df.iloc[:, :-1]
# Output data
y = df.iloc[:, -1]

# Splitting the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=77, stratify=y)

# Defining the parameters grid for manual GridSearchCV
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [0.0001, 0.001, 0.1, 1],
    'kernel': ['rbf', 'poly']
}

best_score = 0
best_params = None

# Manually perform grid search with progress bar
total_params = len(param_grid['C']) * len(param_grid['gamma']) * len(param_grid['kernel'])
progress = tqdm(total=total_params, desc='Training model')

for C in param_grid['C']:
    for gamma in param_grid['gamma']:
        for kernel in param_grid['kernel']:
            svc = svm.SVC(C=C, gamma=gamma, kernel=kernel, probability=True)
            svc.fit(x_train, y_train)
            score = svc.score(x_test, y_test)
            if score > best_score:
                best_score = score
                best_params = {'C': C, 'gamma': gamma, 'kernel': kernel}
            progress.update(1)

progress.close()

# Train the best model with the best parameters found
best_svc = svm.SVC(**best_params, probability=True)
best_svc.fit(x_train, y_train)

# Evaluate the model
y_pred = best_svc.predict(x_test)
print(f'Best Parameters: {best_params}')
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
print(f'Classification Report:\n{classification_report(y_test, y_pred)}')