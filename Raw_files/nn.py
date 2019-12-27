# Neural network


# Let's build our neural network!
# To do so, we need to expand the dimensions of our array, adding a third one using the numpy "expand_dims" feature.
import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
import time

start_time = time.time()

X = joblib.load('C:/Users/SOURAV R S/Desktop/Emotion-Classification-Ravdess/features/X.joblib')
y = joblib.load('C:/Users/SOURAV R S/Desktop/Emotion-Classification-Ravdess/features/y.joblib')

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

x_traincnn = np.expand_dims(X_train, axis=2)
x_testcnn = np.expand_dims(X_test, axis=2)

from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
lb = LabelEncoder()

y_traincnn = np_utils.to_categorical(lb.fit_transform(y_train))
y_testcnn = np_utils.to_categorical(lb.fit_transform(y_test))

x_traincnn.shape, x_testcnn.shape, y_traincnn.shape, y_testcnn.shape

import keras
import numpy as np
import matplotlib.pyplot as plt
# import tensorflow as tf
from keras_tqdm import TQDMCallback
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.utils import to_categorical
from keras.layers import Input, Flatten, Dropout, Activation ,BatchNormalization
from keras.layers import Conv1D, MaxPooling1D
from keras.models import Model
from keras.callbacks import ModelCheckpoint

model = Sequential()

model.add(Conv1D(128, 5,padding='same',
                 input_shape=(x_traincnn.shape[1],x_traincnn.shape[2])))
model.add(Activation('relu'))
model.add(Dropout(0.1))
model.add(MaxPooling1D(pool_size=(8)))
model.add(Conv1D(128, 5,padding='same',))
model.add(Activation('relu'))
model.add(Dropout(0.1))
model.add(Flatten())
model.add(Dense(y_traincnn.shape[1]))
model.add(Activation('softmax'))
opt = keras.optimizers.rmsprop(lr=0.00001, rho=0.9, decay=1e-6)

# Altrenate Model

# model = Sequential()
# model.add(Conv1D(256, 5,padding='same', input_shape=(X_train.shape[1],1)))
# model.add(Activation('relu'))
# model.add(Conv1D(128, 5,padding='same'))
# model.add(Activation('relu'))
# model.add(Dropout(0.1))
# model.add(MaxPooling1D(pool_size=(8)))
# model.add(Conv1D(128, 5,padding='same',))
# model.add(Activation('relu'))
# model.add(Conv1D(128, 5,padding='same',))
# model.add(Activation('relu'))
# model.add(Conv1D(128, 5,padding='same',))
# model.add(Activation('relu'))
# model.add(Dropout(0.2))
# model.add(Conv1D(128, 5,padding='same',))
# model.add(Activation('relu'))
# model.add(Flatten())
# model.add(Dense(8))
# model.add(Activation('softmax'))
# opt = keras.optimizers.rmsprop(lr=0.00001, decay=1e-6)

os.system('cls')
model.summary()

model.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['accuracy'])

cnnhistory=model.fit(x_traincnn, y_traincnn, batch_size=32, epochs=1000, validation_data=(x_testcnn, y_testcnn),verbose=0, callbacks=[TQDMCallback()])

# Let's plot the loss:

plt.plot(cnnhistory.history['loss'])
plt.plot(cnnhistory.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# And now let's plot the accuracy:

plt.plot(cnnhistory.history['accuracy'])
plt.plot(cnnhistory.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# Let's now create a classification report to review the f1-score of the model per class.
# To do so, we have to:
# - Create a variable predictions that will contain the model.predict_classes outcome
# - Convert our y_test (array of strings with our classes) to an array of int called new_Ytest, otherwise it will not be comparable to the predictions by the classification report.

predictions = model.predict_classes(x_testcnn)
predictions
np.argmax(y_testcnn, axis=1)

# Okay, now we can display the classification report:

from sklearn.metrics import classification_report
report = classification_report(np.argmax(y_testcnn, axis=1), predictions)
print(report)

# And now, the confusion matrix: it will show us the misclassified samples

from sklearn.metrics import confusion_matrix
matrix = confusion_matrix(np.argmax(y_testcnn, axis=1), predictions)
print (matrix)

print('\n0 = neutral, 1 = calm, 2 = happy, 3 = sad, 4 = angry, 5 = fearful, 6 = disgust, 7 = surprised\n')

# # Save the model

model_name = 'Emotion_Voice_Detection_CNNModel.h5'
save_dir = 'C:/Users/SOURAV R S/Desktop/Emotion-Classification-Ravdess/model/'
# Save model and weights
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)


# # Reloading the model to test it

loaded_model = keras.models.load_model('C:/Users/SOURAV R S/Desktop/Emotion-Classification-Ravdess/model/Emotion_Voice_Detection_CNNModel.h5')
loaded_model.summary()


# # Checking the accuracy of the loaded model

loss, acc = loaded_model.evaluate(x_testcnn, y_testcnn)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))

print("--- Total time taken: %s min ---" % ((time.time() - start_time)/60))
