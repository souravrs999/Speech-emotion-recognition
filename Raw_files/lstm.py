# LSTM


import os
import joblib
import keras
import numpy as np
from keras_tqdm import TQDMCallback
from keras.layers import LSTM
from keras.models import Model
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D #, AveragePooling1D
from keras.layers import Flatten, Dropout, Activation # Input, 
from keras.layers import Dense #, Embedding
import matplotlib.pyplot as plt
import time

# Loading saved models

start_time = time.time()
X = joblib.load('C:/Users/SOURAV R S/Desktop/Emotion-Classification-Ravdess/features/X.joblib')
y = joblib.load('C:/Users/SOURAV R S/Desktop/Emotion-Classification-Ravdess/features/y.joblib')

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

x_trainlstm =np.expand_dims(X_train, axis=2)
x_testlstm= np.expand_dims(X_test, axis=2)
x_trainlstm.shape

hidden_units = x_trainlstm.shape[1]*2

hidden_units

hidden_units*3//4

# One-Hot Encoding
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from keras_tqdm import TQDMNotebookCallback

lb = LabelEncoder()

y_trainlstm = np_utils.to_categorical(lb.fit_transform(y_train))
y_testlstm = np_utils.to_categorical(lb.fit_transform(y_test))

model = Sequential()

model.add(LSTM(hidden_units,return_sequences=True, input_shape=(x_trainlstm.shape[1],1)))
model.add(Dropout(0.2))

model.add(LSTM(hidden_units, return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM((hidden_units*3)//4))
model.add(Dropout(0.2))


model.add(Dense(y_trainlstm.shape[1]))

model.add(Activation('softmax'))
opt = keras.optimizers.rmsprop(lr=0.00001, decay=1e-6)

os.system('cls')

print('\n-----LSTM-----\n')
print('\n...this will take a long time please be patient...\n')

model.summary()

model.compile(loss='categorical_crossentropy', optimizer=opt,metrics=['accuracy'])

lstmhistory=model.fit(x_trainlstm, y_trainlstm, batch_size=16, epochs=50, validation_data=(x_testlstm, y_testlstm),verbose=0, callbacks=[TQDMCallback()])

plt.plot(lstmhistory.history['loss'])
plt.plot(lstmhistory.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(lstmhistory.history['accuracy'])
plt.plot(lstmhistory.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

model_name = 'Emotion_Voice_Detection_LSTMModel.h5'
save_dir = 'C:/Users/SOURAV R S/Desktop/Emotion-Classification-Ravdess/model/'
# Save model and weights
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('\nSaved trained model at %s ' % model_path)

loaded_model = keras.models.load_model('C:/Users/SOURAV R S/Desktop/Emotion-Classification-Ravdess/model/Emotion_Voice_Detection_LSTMModel.h5')
loaded_model.summary()

loss, acc = loaded_model.evaluate(x_testlstm, y_testlstm)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))
print("--- Total time taken: %s min ---" % ((time.time() - start_time)/60))
