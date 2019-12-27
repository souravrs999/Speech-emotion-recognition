#!/usr/bin/env python
# coding: utf-8
# !pip install librosa
import librosa

from tqdm import tqdm_gui, tqdm
from librosa import display
import numpy as np

data, sampling_rate = librosa.load('C:/Users/SOURAV R S/Desktop/Emotion-Classification-Ravdess/Ravdess/Actor_01/03-01-01-01-01-01-01.wav')

# get_ipython().run_line_magic('pylab', 'inline')
import matplotlib.pyplot as plt
import os
import pandas as pd
import glob 
plt.figure(figsize=(12, 4))
librosa.display.waveplot(data, sr=sampling_rate)
plt.show()


# # Load all files
# We will create our numpy array extracting Mel-frequency cepstral coefficients (MFCCs), Chroma_STFT, MEL Spectrogram, Spectral Contrast, Tonnetz, Zero Crossing Rate, while the classes to predict will be extracted from the name of the file (see the introductory section of this notebook to see the naming convention of the files of this dataset).

import time
print('---Loading the wav files and extracting features from it please be patient---')

path = 'C:/Users/SOURAV R S/Desktop/Emotion-Classification-Ravdess/Ravdess/'
lst = []

start_time = time.time()

for subdir, dirs, files in tqdm(os.walk(path)):
    for file in files:
        try:
        #Load librosa array, obtain mfcss, store the file and the mcss information in a new array
            X, sample_rate = librosa.load(os.path.join(subdir,file), res_type='kaiser_fast')
            stft = np.abs(librosa.stft(X))
            mfccs = np.mean(librosa.feature.mfcc(X, sr=sample_rate, n_mfcc=40).T,axis=0)
            chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
            mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
            contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)
            tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X),sr=sample_rate).T,axis=0)
            zerocr = np.mean(librosa.feature.zero_crossing_rate(X).T,axis=0)
        # The instruction below converts the labels (from 1 to 8) to a series from 0 to 7
        # This is because our predictor needs to start from 0 otherwise it will try to predict also 0.
            file = int(file[7:8]) - 1
            # This should be used only if you are using the bigger dataset 
            # file = file[:3]
            feature_list = np.concatenate((mfccs, chroma, mel, contrast, tonnetz, zerocr))
            arr = feature_list,file
            lst.append(arr)
      # If the file is not valid, skip it
        except ValueError:
            continue

print("--- Data loaded. Loading time: %s seconds ---" % (time.time() - start_time))

print('---processing the Data and saving them into joblib files, (it has already been split into X and y no further modification required)---')
# Creating X and y: zip makes a list of all the first elements, and a list of all the second elements.
X, y = zip(*lst)
# X = lst[1000:4]
# y = lst[:-1]

y[:5]


import numpy as np
X = np.asarray(X)
y = np.asarray(y)


X.shape, y.shape

# Saving joblib files to not load them again with the loop above

import joblib

X_name = 'Xsmall.joblib'
y_name = 'ysmall.joblib'
save_dir = 'C:/Users/SOURAV R S/Desktop/Emotion-Classification-Ravdess/'

savedX = joblib.dump(X, os.path.join(save_dir, X_name))
savedy = joblib.dump(y, os.path.join(save_dir, y_name))
print(f'joblib files have been saved to{save_dir}')
