import os 
import sys
import joblib
import time
import random
from tqdm import tqdm
import librosa
from librosa import feature, display
import numpy as np
from pandas.plotting import scatter_matrix
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier
import keras
from keras.utils import np_utils
from keras_tqdm import TQDMCallback
from keras.layers import LSTM
from keras.models import Model
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D #, AveragePooling1D
from keras.layers import Flatten, Dropout, Activation # Input, 
from keras.layers import Dense #, Embedding

def logo():
	clr = random.choice(('color 0C', 'color 0A', 'color 0E', 'color 0F', 'color 0B', 'color0d'))
	os.system(clr)
	os.system('cls')
	print('[*][*][*] SPEECH EMOTION DETECTION[*][*][*]')
	print('--- written in python ---\n\n')
	print('''
				                                      
	            oo88888888boo
	            `""88888888888bo
	                ""888; `"Y888o
	                   Y88;    "Y88.
	                    "88.     `88b.    ,
	                     `Yb      `888.   :8b
	                       Yb    , `888.   88b
	                        Y.    ` `"88.  Y8"8.
	                         Y. `. b ``8b  :8 Y8.
	           ,oooooooo      Yo :.`b`b`8;  8  8b
	    ,ood8P""""""""88888oo  8b Y.:b:b:8;,8 ,:8.
	,od88888bo  ` ,o.   """888o`8b`8 Y.8.88d8 : 8;
	"""""""""""8oo`,. 'oo.   ""888b8b:8db8888 d :8 :;
	          d8888boP , "Y88o. ""Y8888888888 8 d8.88
	        o""""888888o``o'"88bood8888888888:8,;8888
	      o"    ""8888888o."88888888" oooo `888d888:8
	     d'    ,o8888888P88od88P""' ,d8888; Y888888:8
	   ,8'     ood88888,dP"88       :888888 :88888;d;
	   8'  ,o88""8888`8P   `"       888888P  8888d;8;
	  d;,o8P" ,d888'oP'             "8888"    d88888boo
	  8,88'  ,8888'88                 `' ,o8; "" Y88888888oooo.
	 :88'   ,8888'dP,88o    :;          d88P    oooo88d888888888bo.
	 `"    ,8888;,;:88888.  d8.        :8P'   ""'          :8888888888ooo
	      ,88888 8,88. :88; 88;     ood"                    88888888888P"Y;
	     oP d88;d;d888od"Y8  8;     ""                      :8P""Y88PYb8 :
	    :P'd888`8'8b  ""Y88. 8'                             `"   `8"  YP 8
	   ,P .:Y8Pd8,8Y88o.  :;                                     `"  o8 d;
	  ,8'','8888;:8o   """Y8           ooood88888oooo.       o         dP
	  8P  ,d88'8;:8"888oooo8;       ,o888888888888888888boo  `Y8oo.   dP
	 :8bd88888 8':8ooo.  ""Yb     odP""          """888888888bo8P""'o8"
	 """""8888 8 :8888888o. 8oooo888oooooooooo.       Y8888888888oo8"
	     d8888 Y :bo     `""""888P"""         ""Ybo.    `"8888888""
	    ,8`Y88.: :8"Y88oooooooo88.                `Ybo     Y8"
	    dP'd88;:; 8o        `""Y8b                  `"b.   dP
	    88`8:8;:; 88888booooood888.                   `8.  8'
	   :8P:'Y88:b 8P            `8b                    `8d8'
	   88 ',88888 Y8888ooooooP""""Yb                    `"
	  ,8; o8888bY;8Yb '         ooo88b                       Author: Sourav R S (darkstalker)
	  :8o8":;888'8;88bo,od8` '`'`'  Ybo                      github: https://github.com/souravrs999/Speech-emotion-detection
	  d8"  d;888bP;o'`        ,.:o:'`"P o                    Mail:souravraveendran6@gmail.com,souravraveendran6@outlook.com
	  "'   8'8888d8b,;odP8;dP'`    o:;`'8 :o       '         @if you use this tool please cite me at  https://github.com/souravrs999/Speech-emotion-detection
	       8 :8P8'88o`8P''    ooo'   ,oo" d8.dboo
	      ,8 :`d88b,88od8888P"'   oo""  ,'" dP"88888
	      :P  88888;8b 888;   oo8"'   ,P' ,8' d'88"8
	      d;,dY88888.Y. Y8888""    odP' ,d" ,d'dP ,P
	      8bP' Y8Y888d8o `Y8;  ood8P' ,dP  o8':P  :;
	     ,P"   :8YY;88b"b  Y8888P"  o'"  o8P ,P   8 
	           `8d:`888b`bo `8b  ,o8"  ,dP' ,P   :;
	            8;:dP88888Yb  Y888;   d8;  ,P    8
	            8;:8 :8888b88. `Y8boo8P'  ,P    :;
	            8b8' `88:;Y88"b. `Y888   ,P     8
	            88'   Y88':88b."8o `"8b.oP     8'
	            "'    :8Y :88888o"8o  :88o.  ,8'
	                   8: 88;8Y88b88bod8"Y8oo8P
	                   8.d':b8`8:P`"8888o. :8P
	                   88'  Yd 88'   `"88888"
	                  :8'   `8 dP       """'
	                  `'     8o8
	                         88'

                    ''')


class emdec:

	os.chdir('C:/Users/SOURAV R S/Desktop/Emotion-Classification-Ravdess/')

	def cross_val():
		
		start_time = time.time()

		ch = str(input('\n[-]Are you sure you want to run 10Fold cross validation (y/n):').upper())
		if ch == 'Y':

			try:
				print('\n[-][-][-] Starting 10Fold Cross Validation [-][-][-]')
				
				# Loading saved models
				X = joblib.load('./features/Xsmall.joblib')
				y = joblib.load('./features/ysmall.joblib')

				X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

				# Spot Check Algorithms
				models = []
				models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
				models.append(('LDA', LinearDiscriminantAnalysis()))
				models.append(('KNN', KNeighborsClassifier()))
				models.append(('CART', DecisionTreeClassifier()))
				models.append(('NB', GaussianNB()))
				models.append(('SVM', SVC(gamma='auto')))

				print('\n--- This process is going to take some time and depend purely on your hardware so grab a cup of coffee sit back and relax ---\n')
				print(models)
				# evaluate each model in turn
				results = []
				names = []
				for name, model in tqdm(models):
					try:
						kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
						cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy', verbose=0)
						results.append(cv_results)
						names.append(name)
						print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))
					except Exception as e:
						print(f'Error:{e}')

				# Compare Algorithms
				plt.boxplot(results, labels=names)
				plt.title('Algorithm Comparison')
				plt.show()
				
			except Exception as e:
				print(f'Error:{e}')
		else:
			sys.exit(1)
		print("--- Total time taken: %s min ---" % ((time.time() - start_time)/60))

	
	def rand_for():

		start_time = time.time()

		ch = str(input('\n[-]Do you want to run Random Forest Classifier(y/n):').upper())
		if ch == 'Y':

			try:

				# Loading saved models
				X = joblib.load('./features/Xsmall.joblib')
				y = joblib.load('./features/ysmall.joblib')


				print('\n[-][-][-]Random Forest Classifier[-][-][-]')
				print('\n-----This wil take some time please be patient-----\n')
				# To make a first attempt in accomplishing this classification task I chose a decision tree:

				X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

				rforest = RandomForestClassifier(criterion="gini", max_depth=10, max_features="log2", 
												max_leaf_nodes = 100, min_samples_leaf = 3, min_samples_split = 20, 
												n_estimators= 22000, random_state= 5, verbose=1)

				rforest.fit(X_train, y_train)
				predictions = rforest.predict(X_test)
				print(classification_report(y_test,predictions))

			except Exception as e:
				print(f'Error:{e}')
		else:
			sys.exit(1)
		print("\n--- Total time taken: %s min ---" % ((time.time() - start_time)/60))
	
	def desc_tree():
		start_time = time.time()

		ch = str(input('\n[-]Do you want to run Decision Classifier(y/n):').upper())
		if ch == 'Y':
			try:
			
				# Loading saved models
				X = joblib.load('./features/Xsmall.joblib')
				y = joblib.load('./features/ysmall.joblib')

				print('[-][-][-]Decision Tree Classifier[-][-][-]')

				X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

				from sklearn.tree import DecisionTreeClassifier

				dtree = DecisionTreeClassifier()
				dtree.fit(X_train, y_train)
				predictions = dtree.predict(X_test)


				# Let's go with our classification report.
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

				print(classification_report(y_test,predictions))
			except Exception as e:
				print(f'Error:{e}')
		else:
			sys.exit(1)
		print("\n--- Total time taken: %s min ---" % ((time.time() - start_time)/60))

	def svm():
		start_time = time.time()

		ch = str(input('\n[-]Do you want to run Support Vector Machine(y/n):').upper())
		if ch == 'Y':
			try:
				print('\n[-][-][-]Support Vector Machine[-][-][-]\n')
				# Loading saved models
				X = joblib.load('./features/Xsmall.joblib')
				y = joblib.load('./features/ysmall.joblib')

				X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

				markers = ('x', 's', 'o', '^', 'v', '*','.', ',')
				colors = ('red', 'blue', 'green', 'grey', 'yellow', 'orange', 'pink','purple')
				cmap = ListedColormap(colors[:len(np.unique(y_test))])
				for idx, cl in enumerate(np.unique(y)):
					plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],c=cmap(idx), 
								marker=markers[idx], label =cl)
					plt.show()
				# plt.show()

				standard_scaler = StandardScaler()
				standard_scaler.fit(X_train)
				X_train_standard = standard_scaler.transform(X_train)
				X_test_standard = standard_scaler.transform(X_test)
				# os.system('cls')
				print('\n-----SVM-----\n')
				# print('The first five rows after standardisation look like this:\n')
				# print(pd.DataFrame(X_train_standard,y_train).head())

				SVM = SVC(kernel='rbf', random_state=0, gamma=.10, C=1.0)
				SVM.fit(X_train_standard, y_train)
				print('Accuracy of our SVM model on the training data is {:.2f} out of 1'.format(SVM.score(X_train_standard, y_train)))
				print('\nAccuracy of our SVM model on the test data is {:.2f} out of 1'.format(SVM.score(X_test_standard, y_test)))
			except Exception as e:
				print(f'Error:{e}')
			
		else:
			sys.exit(1)
		print("\n--- Total time taken: %s min ---" % ((time.time() - start_time)/60))

	def mlp():

		start_time = time.time()
		ch = str(input('\n[-]Do you want to run Multi Layer Perceptron(y/n):').upper())
		if ch == 'Y':
			try:

				print('\n[-][-][-]Multi Layer Perceptron[-][-][-]\n')
				# Loading saved models
				X = joblib.load('./features/Xsmall.joblib')
				y = joblib.load('./features/ysmall.joblib')

				print('\n-----MLP Classifier-----\n')

				from sklearn.model_selection import train_test_split
				X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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

				print(f'\nDefault parameters:{params}')

				model = MLPClassifier(**params)

				print("\n[*] Training the model...\n")
				model.fit(X_train, y_train)
				y_pred = model.predict(X_test)

				# calculating the accuracy
				accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)

				print("Accuracy: {:.2f}%".format(accuracy*100))
			except Exception as e:
				print(f'Error:{e}')
		else:
			sys.exit(1)
		print("\n--- Total time taken: %s min ---" % ((time.time() - start_time)/60))

	def lstm():

		start_time = time.time()
		ch = str(input('\n[-]Do you want to run Long Short Term Memory(y/n):').upper())
		if ch == 'Y':
			start_time = time.time()
			# Loading saved models
			X = joblib.load('./features/Xsmall.joblib')
			y = joblib.load('./features/ysmall.joblib')
			lb = LabelEncoder()

			X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
			x_trainlstm =np.expand_dims(X_train, axis=2)
			x_testlstm= np.expand_dims(X_test, axis=2)
			y_trainlstm = np_utils.to_categorical(lb.fit_transform(y_train))
			y_testlstm = np_utils.to_categorical(lb.fit_transform(y_test))

			ch = str(input('\n[-]Do you want to retrain the model(Y)(will take long time..) or use the pretrained model(n):').upper())
			if ch == 'Y':

				try:
					print('\n[-][-][-]Long Short Term Memory[-][-][-]\n')
					
					x_trainlstm.shape

					hidden_units = x_trainlstm.shape[1]*2

					hidden_units

					hidden_units*3//4

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
					opt = keras.optimizers.rmsprop(lr=0.00005, decay=1e-6)

					# os.system('cls')
					print('\n-----LSTM-----\n')
					print('\n--- This process is going to take some time and depend purely on your hardware so grab a cup of coffee sit back and relax ---\n')

					model.summary()

					model.compile(loss='categorical_crossentropy', optimizer=opt,metrics=['accuracy'])

					lstmhistory=model.fit(x_trainlstm, y_trainlstm, batch_size=32, epochs=100, validation_data=(x_testlstm, y_testlstm),verbose=0, callbacks=[TQDMCallback()])

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
					save_dir = './model/'
					# Save model and weights
					if not os.path.isdir(save_dir):
						os.makedirs(save_dir)
					model_path = os.path.join(save_dir, model_name)
					model.save(model_path)
					print('\nSaved trained model at %s ' % model_path)

					loaded_model = keras.models.load_model('./model/Emotion_Voice_Detection_LSTMModel.h5')
					loaded_model.summary()

					loss, acc = loaded_model.evaluate(x_testlstm, y_testlstm)
					print("Restored model, accuracy: {:5.2f}%".format(100*acc))
					# print("--- Total time taken: %s min ---" % ((time.time() - start_time)/60))
				except Exception as e:
					print(f'Error:{e}')
			else:
				try:
					print('\nUsing the pre trained model...\n')
					loaded_model = keras.models.load_model('./model/Emotion_Voice_Detection_LSTMModel.h5')
					loaded_model.summary()

					loss, acc = loaded_model.evaluate(x_testlstm, y_testlstm)
					print("Restored model, accuracy: {:5.2f}%".format(100*acc))
					# print("--- Total time taken: %s min ---" % ((time.time() - start_time)/60))
				except Exception as e:
					print(f'Error:{e}')
		else:
			sys.exit(1)
		print("\n--- Total time taken: %s min ---" % ((time.time() - start_time)/60))

	def cnn():

		start_time = time.time()
		ch = str(input('\n[-]Do you want to run Convolutional Neural Network(y/n):').upper())
		if ch == 'Y':
			# Loading saved models
			X = joblib.load('./features/Xsmall.joblib')
			y = joblib.load('./features/ysmall.joblib')
			lb = LabelEncoder()

			X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
			x_traincnn =np.expand_dims(X_train, axis=2)
			x_testcnn= np.expand_dims(X_test, axis=2)
			y_traincnn = np_utils.to_categorical(lb.fit_transform(y_train))
			y_testcnn = np_utils.to_categorical(lb.fit_transform(y_test))

			ch = str(input('\n[-]Do you want to retrain the model(Y)(will take some time..) or use the pretrained model(n):').upper())
			if ch == 'Y':

				try:
					print('\n[-][-][-]Convolutional Neural Network[-][-][-]\n')
					
					model = Sequential()

					# model.add(Conv1D(128, 5,padding='same',input_shape=(x_traincnn.shape[1],x_traincnn.shape[2])))
					# model.add(Activation('relu'))
					# model.add(Dropout(0.1))
					# model.add(MaxPooling1D(pool_size=(8)))
					# model.add(Conv1D(128, 5,padding='same',))
					# model.add(Activation('relu'))
					# model.add(Dropout(0.1))
					# model.add(Flatten())
					# model.add(Dense(y_traincnn.shape[1]))
					# model.add(Activation('softmax'))
					# opt = keras.optimizers.rmsprop(lr=0.00001, rho=0.9, decay=1e-6)

					# Altrenate Model

					# model = Sequential()
					model.add(Conv1D(256, 5,padding='same', input_shape=(X_train.shape[1],1)))
					model.add(Activation('relu'))
					model.add(Conv1D(128, 5,padding='same'))
					model.add(Activation('relu'))
					model.add(Dropout(0.1))
					model.add(MaxPooling1D(pool_size=(8)))
					model.add(Conv1D(128, 5,padding='same',))
					model.add(Activation('relu'))
					model.add(Conv1D(128, 5,padding='same',))
					model.add(Activation('relu'))
					model.add(Conv1D(128, 5,padding='same',))
					model.add(Activation('relu'))
					model.add(Dropout(0.2))
					model.add(Conv1D(128, 5,padding='same',))
					model.add(Activation('relu'))
					model.add(Flatten())
					model.add(Dense(y_traincnn.shape[1]))
					model.add(Activation('softmax'))
					opt = keras.optimizers.rmsprop(lr=0.00001, decay=1e-6)

					# os.system('cls')
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
					save_dir = './model/'
					# Save model and weights
					if not os.path.isdir(save_dir):
						os.makedirs(save_dir)
					model_path = os.path.join(save_dir, model_name)
					model.save(model_path)
					print('Saved trained model at %s ' % model_path)


					# # Reloading the model to test it

					loaded_model = keras.models.load_model('./model/Emotion_Voice_Detection_CNNModel.h5')
					loaded_model.summary()


					# # Checking the accuracy of the loaded model

					loss, acc = loaded_model.evaluate(x_testcnn, y_testcnn)
					print("Restored model, accuracy: {:5.2f}%".format(100*acc))

					print("--- Total time taken: %s min ---" % ((time.time() - start_time)/60))
				except Exception as e:
					print(f'Error:{e}')
			else:
				try:
					print('\nUsing the pre trained model...\n')
					# # Reloading the model to test it

					loaded_model = keras.models.load_model('./model/Emotion_Voice_Detection_CNNModel.h5')
					loaded_model.summary()

					# # Checking the accuracy of the loaded model

					loss, acc = loaded_model.evaluate(x_testcnn, y_testcnn)
					print("Restored model, accuracy: {:5.2f}%".format(100*acc))

				except Exception as e:
					print(f'Error:{e}')
		else:
			sys.exit(1)
		print("\n--- Total time taken: %s min ---" % ((time.time() - start_time)/60))

	def datapre():
		start_time = time.time()
		ch = str(input('\n[-]Do you want to run Data preprocessing(y/n):').upper())
		if ch == 'Y':
			try:
				data, sampling_rate = librosa.load('./Ravdess/Actor_01/03-01-01-01-01-01-01.wav')
				plt.figure(figsize=(12, 4))
				librosa.display.waveplot(data, sr=sampling_rate)
				plt.show()

				print('\n---Loading the wav files and extracting features from it please be patient---\n')

				path = './Ravdess/'
				lst = []

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

				# print("--- Data loaded. Loading time: %s seconds ---" % (time.time() - start_time))

				print('---processing the Data and saving them into joblib files, (it has already been split into X and y no further modification required)---')
				# Creating X and y: zip makes a list of all the first elements, and a list of all the second elements.
				X, y = zip(*lst)
				# X = lst[1000:4]
				# y = lst[:-1]

				y[:5]
				X = np.asarray(X)
				y = np.asarray(y)
				X.shape, y.shape

				# Saving joblib files to not load them again with the loop above

				X_name = 'Xsmall.joblib'
				y_name = 'ysmall.joblib'
				save_dir = './features/'

				savedX = joblib.dump(X, os.path.join(save_dir, X_name))
				savedy = joblib.dump(y, os.path.join(save_dir, y_name))
				print(f'joblib files have been saved to{save_dir}')
			except  Exception as e:
				print(f'Error:{e}')
		else:
			sys.exit(1)
		print("\n--- Total time taken: %s min ---" % ((time.time() - start_time)/60))

	def __init__(self):
		try:
			print('\n\n\n[-]Starting program .........\n')
			print('[-]-----Main Menu-----\n\n[-][1]10Fold Cross Validation\n\n[-][2]Random Forest Classifier\n\n[-][3]Decision Tree Classifier\n\n[-][4]Support Vector Machine\n\n[-][5]Multi Layer Perceptron\n\n[-][6]Long Short-term Memory\n\n[-][7]Convolutional Neural Network\n\n[-][8]Run data preprocessing\n')
			self.mode = int(input('\n[-]Enter your choice:'))
			print(f'\n[-]You chose option:[ {self.mode} ]\n')
			print(f'\t\t\t--------------------------------------------------------------------------------------------------------------------------------------------------------')
			if int(self.mode) > int(8) or int(self.mode) < int(1):
				print('\n[-]Error: Invalid option please select one from above')
				sys.exit(1)
		except Exception as e:
			print(f'\n[-]Error {e}')
		try:

			if self.mode == int(1):
				emdec.cross_val()
			elif self.mode == int(2):
				emdec.rand_for()
			elif self.mode == int(3):
				emdec.desc_tree()
			elif self.mode == int(4):
				emdec.svm()
			elif self.mode == int(5):
				emdec.mlp()
			elif self.mode == int(6):
				emdec.lstm()
			elif self.mode == int(7):
				emdec.cnn()
			elif self.mode == int(8):
				emdec.datapre()
		except Exception as e:
			print(f'[-]Error {e}')


if __name__ == '__main__':
	logo()
	emdec = emdec()
