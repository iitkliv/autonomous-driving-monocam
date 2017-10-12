import numpy as np
import keras
import matplotlib.pyplot as plt
import csv
import cv2
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.models import *
from keras.layers import *
from tqdm import tqdm
from multiprocessing import Pool

#resized image dimension in training
img_rows = 16
img_cols = 32

#data path
data_folder = '../../data/udacity_sim_data/'

#batch size and epoch
batch_size=128
nb_epoch=20

def image_preprocessing(img):
	"""preproccesing training data to keep only S channel in HSV color space, and resize to 16X32"""

	resized = cv2.resize((cv2.cvtColor(img, cv2.COLOR_RGB2HSV))[:,:,1],(img_cols,img_rows))
	return resized

def extract(i):
	X_ = []
	y_ = []

	img_path = logs[i][0]
	img_path = data_folder+'IMG'+(img_path.split('IMG')[1]).strip()
	img = plt.imread(img_path)
	X_.append(image_preprocessing(img))
	y_.append([float(logs[i][3]), (float(logs[i][6])-15)/15.0])

	img_path = logs[i][1]
	img_path = data_folder+'IMG'+(img_path.split('IMG')[1]).strip()
	img = plt.imread(img_path)
	X_.append(image_preprocessing(img))
	y_.append([float(logs[i][3]) + delta_, (float(logs[i][6])-15)/15.0])

	img_path = logs[i][2]
	img_path = data_folder+'IMG'+(img_path.split('IMG')[1]).strip()
	img = plt.imread(img_path)
	X_.append(image_preprocessing(img))
	y_.append([float(logs[i][3]) - delta_, (float(logs[i][6])-15)/15.0])

	return X_, y_

def load_data(X,y,data_folder,delta=0.08):
	"""function to load training data"""

	global logs
	global delta_

	delta_ = delta

	log_path = data_folder + 'driving_log.csv'
	logs = []
	
	# load logs
	with open(log_path,'rt') as f:
		reader = csv.reader(f)
		for line in reader:
			logs.append(line)
		log_labels = logs.pop(0)

	pool = Pool(processes=6)

	for X_, y_ in tqdm(pool.imap_unordered(extract, range(len(logs))), total=len(logs)):
		X.extend(X_)
		y.extend(y_)
	pool.close()
	# load center camera image
	# for i in tqdm(range(len(logs)), total=len(logs)):
	# 	img_path = logs[i][0]
	# 	img_path = data_folder+'IMG'+(img_path.split('IMG')[1]).strip()
	# 	img = plt.imread(img_path)
	# 	X.append(image_preprocessing(img))
	# 	y.append(float(logs[i][3]))

	# # load left camera image
	# for i in tqdm(range(len(logs)), total=len(logs)):
	# 	img_path = logs[i][1]
	# 	img_path = data_folder+'IMG'+(img_path.split('IMG')[1]).strip()
	# 	img = plt.imread(img_path)
	# 	X.append(image_preprocessing(img))
	# 	y.append(float(logs[i][3]) + delta)

	# # load right camera image
	# for i in tqdm(range(len(logs)), total=len(logs)):
	# 	img_path = logs[i][2]
	# 	img_path = data_folder+'IMG'+(img_path.split('IMG')[1]).strip()
	# 	img = plt.imread(img_path)
	# 	X.append(image_preprocessing(img))
	# 	y.append(float(logs[i][3]) - delta)



if __name__ == '__main__':

	#load data

	print("loading data...")

	data={}
	data['features'] = []
	data['labels'] = []

	load_data(data['features'], data['labels'],data_folder,0.3)

	X_train = np.array(data['features']).astype('float32')
	y_train = np.array(data['labels']).astype('float32')
	y_train = y_train.reshape([-1, 2])

    

	# horizonal reflection to agument the data
	X_train = np.append(X_train,X_train[:,:,::-1],axis=0)
	y_train = np.append(y_train, np.matmul(y_train, np.array([[-1, 0], [0, 1]])),axis=0)

	# split train and validation
	X_train, y_train = shuffle(X_train, y_train)
	X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, random_state=0, test_size=0.1)

	# reshape to have correct dimension
	X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
	X_val = X_val.reshape(X_val.shape[0], img_rows, img_cols, 1)


	# build model

	print("building model...")

	model = Sequential([
			Lambda(lambda x: x/127.5 - 1.,input_shape=(img_rows,img_cols,1)),
			Conv2D(2, 3, 3, border_mode='valid', input_shape=(img_rows,img_cols,1), activation='relu'),
			MaxPooling2D((4,4),(4,4),'valid'),
			Dropout(0.25),
			Flatten(),
			Dense(2)
		])

	model.summary()


	# training

	print("training model...")

	model.compile(loss='mean_squared_error',optimizer='adam')
	history = model.fit(X_train, y_train,batch_size=batch_size, nb_epoch=nb_epoch,verbose=1, validation_data=(X_val, y_val))


	# save model

	print('Saving model...')

	model_json = model.to_json()
	with open("model_speed.json", "w") as json_file:
		json_file.write(model_json)

	model.save_weights("model_speed.h5")
	print("Model Saved.")
