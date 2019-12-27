# MLP Classifier


import os
import joblib

# Loading saved models

X = joblib.load('C:/Users/SOURAV R S/Desktop/Emotion-Classification-Ravdess/features/Xsmall.joblib')
y = joblib.load('C:/Users/SOURAV R S/Desktop/Emotion-Classification-Ravdess/features/ysmall.joblib')



print('\n-----MLP Classifier-----\n')

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

print("[+] Number of training samples:", X_train.shape[0])

print("\n[+] Number of testing samples:", X_test.shape[0])

print("'\n[+] Number of features:", X_train.shape[1])

params = {
    'alpha': 0.01,
    'batch_size': 16,
    'epsilon': 1e-08, 
    'hidden_layer_sizes': (300,), 
    'learning_rate': 'adaptive', 
    'max_iter': 1000,
    'verbose':0
}

model = MLPClassifier(**params)

print("\n[*] Training the model...\n")
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# calculating the accuracy
accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)

print("Accuracy: {:.2f}%".format(accuracy*100))
