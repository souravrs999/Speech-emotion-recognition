# Descision Tree Classifier


import os
import joblib

# Loading saved models

X = joblib.load('C:/Users/SOURAV R S/Desktop/Emotion-Classification-Ravdess/features/Xsmall.joblib')
y = joblib.load('C:/Users/SOURAV R S/Desktop/Emotion-Classification-Ravdess/features/ysmall.joblib')


print('-----Decision Tree Classifier-----')
# To make a first attempt in accomplishing this classification task I chose a decision tree:

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.tree import DecisionTreeClassifier

dtree = DecisionTreeClassifier()
dtree.fit(X_train, y_train)
predictions = dtree.predict(X_test)


# Let's go with our classification report.
# 
print('''\na quick reminder of the classes we are trying to predict:

emotions = {
    "neutral": "0",
    "calm": "1",
    "happy": "2",
    "sad": "3",
    "angry": "4", 
    "fearful": "5", 
    "disgust": "6", 
    "surprised": "7"
}''')

from sklearn.metrics import classification_report,confusion_matrix
from sklearn.preprocessing import LabelEncoder
print(classification_report(y_test,predictions))