# Random Forest Classifier


import os
import joblib

# Loading saved models

X = joblib.load('C:/Users/SOURAV R S/Desktop/Emotion-Classification-Ravdess/features/Xsmall.joblib')
y = joblib.load('C:/Users/SOURAV R S/Desktop/Emotion-Classification-Ravdess/features/ysmall.joblib')


print('-----Random Forest Classifier-----')
print('-----This wil take some time please be patient-----')
# To make a first attempt in accomplishing this classification task I chose a decision tree:

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn.ensemble import RandomForestClassifier

rforest = RandomForestClassifier(criterion="gini", max_depth=10, max_features="log2", 
                                 max_leaf_nodes = 100, min_samples_leaf = 3, min_samples_split = 20, 
                                 n_estimators= 22000, random_state= 5, verbose=1)

rforest.fit(X_train, y_train)
predictions = rforest.predict(X_test)
print(classification_report(y_test,predictions))
