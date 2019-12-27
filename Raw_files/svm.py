# Support Vector Machines


import os
import joblib
import numpy as np
import pandas as pd

# Loading saved models

X = joblib.load('C:/Users/SOURAV R S/Desktop/Emotion-Classification-Ravdess/features/Xsmall.joblib')
y = joblib.load('C:/Users/SOURAV R S/Desktop/Emotion-Classification-Ravdess/features/ysmall.joblib')

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

markers = ('x', 's', 'o', '^', 'v', '*','.', ',')
colors = ('red', 'blue', 'green', 'grey', 'yellow', 'orange', 'pink','purple')
cmap = ListedColormap(colors[:len(np.unique(y_test))])
for idx, cl in enumerate(np.unique(y)):
    plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],c=cmap(idx), 
                marker=markers[idx], label =cl)

standard_scaler = StandardScaler()
standard_scaler.fit(X_train)
X_train_standard = standard_scaler.transform(X_train)
X_test_standard = standard_scaler.transform(X_test)
os.system('cls')
print('\n-----SVM-----\n')
# print('The first five rows after standardisation look like this:\n')
# print(pd.DataFrame(X_train_standard,y_train).head())


SVM = SVC(kernel='rbf', random_state=0, gamma=.10, C=1.0)
SVM.fit(X_train_standard, y_train)
print('Accuracy of our SVM model on the training data is {:.2f} out of 1'.format(SVM.score(X_train_standard, y_train)))
print('Accuracy of our SVM model on the test data is {:.2f} out of 1'.format(SVM.score(X_test_standard, y_test)))