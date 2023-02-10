import os
import numpy as np
import scipy as sc
import pandas as pd
import tensorflow as tf

grades_df = pd.read_csv("../data/eeg_grades.csv")

from scipy.io import loadmat

file_ext = ".mat"
data_basepath = "../data/MAT_format/"
data_files = list(grades_df['file_ID'])	

data_qtfd = dict()
for fname in data_files:
	data_mat = loadmat(data_basepath + fname + '.mat')
	data_qtfd[fname] = np.array(data_mat['qtfd_log'])

print("loaded mat data")

train_test_grades_df = grades_df.dropna()
train_test_grades_df = train_test_grades_df.set_index('file_ID')
train_test_files = list(train_test_grades_df.index)

chop_f = True
if chop_f:
	qtfd_shape = (256,112)
else:
	qtfd_shape = list(data_qtfd.values())[1].shape[0:2]  #(256, 128)


def get_qtfd(fname, ch, seg):
	qtfd = data_qtfd[fname]
	if chop_f:
		return qtfd[:, 7:119, ch, seg]
	else:
		return qtfd[:, :, ch, seg]


nsegm = list(data_qtfd.values())[1].shape[3]
nch = list(data_qtfd.values())[1].shape[2]
nfiles = len(train_test_grades_df.index)
n_total_inputs = nsegm*nch*nfiles

# train_test_x = np.empty( (n_total_inputs, qtfd_shape[0], qtfd_shape[1]) )
# train_test_y = np.empty( (n_total_inputs, 1) )

# print('shape input train_test_x: ', train_test_x.shape)
# print('shape target train_test_y: ', train_test_y.shape)

file_ID = []
baby_ID = []
epoch = []
segment = []
channel = []
qtfd = []
grade = []

grades_df = grades_df.set_index('file_ID')
filenames = list(grades_df.index)
for i, fname in enumerate(filenames):
	# print(fname)
	for j in range(nsegm):
		for k in range(nch):
			n = i*(nsegm*nch) + j*nch + k;
			#print(n)
			file_ID.append(fname)
			baby_ID.append(grades_df['baby_ID'].loc[fname])
			epoch.append(grades_df['epoch_number'].loc[fname])
			segment.append(j)
			channel.append(k)
			qtfd.append(get_qtfd(fname, k, j))
			grade.append(grades_df['grade'].loc[fname])


data_dict = {'file_ID': file_ID, 'baby_ID': baby_ID, 'epoch': epoch, 'segment': segment, 'channel': channel, 'qTFD': qtfd, 'grade': grade}
data = pd.DataFrame(data_dict)
data_train_test = data.dropna().copy()

print("created data dataframe")

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import Activation, Dense, Flatten, BatchNormalization, Conv2D, MaxPool2D, MaxPooling2D, AveragePooling2D, Dropout,GlobalAveragePooling2D, MaxPooling1D 
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy



def create_convnet():
    # layers as specified in the paper
    input_shape = tf.keras.Input(shape=(256, 112, 1))

    tower_1 = Conv2D(10, (8, 1), padding='same', activation='relu')(input_shape)
    tower_1 = MaxPooling2D((4, 4), strides=(2, 2), padding='same')(tower_1)

    tower_2 = Conv2D(10, (1, 8), padding='same', activation='relu')(input_shape)
    tower_2 = MaxPooling2D((4, 4), strides=(2, 2), padding='same')(tower_2)

    tower_3 = Conv2D(10, (8, 8), padding='same', activation='relu')(input_shape)
    tower_3 = MaxPooling2D((4, 4), strides=(2, 2), padding='same')(tower_3)

    merged = keras.layers.concatenate([tower_1, tower_2, tower_3], axis=3)

    layer1 = Conv2D(60, (4,4), padding='same', 
                    activation ='relu',strides =(2,2))(merged)
    
    layer2a = MaxPooling2D((2,2), padding ='same', strides =(2,2))(layer1)
    layer2b = BatchNormalization()(layer2a)
    layer3 = Conv2D(60, (2,2), padding='same' )(layer2b)

    layer4 = MaxPooling2D( (2,2), padding='same', strides =(2,2))(layer3)
    layer5 = GlobalAveragePooling2D()(layer4)

    out = Dense(60, activation='relu')(layer5)
    out = Dense(4, activation='softmax')(out)

    model = tf.keras.Model(input_shape, out)
    #plot_model(model, to_file=img_path)
    return model

model = create_convnet()
model.summary()

#defining the learning rate step
def scheduler(epoch, lr):
  n =np.floor((epoch-1)/5)
  return lr*(0.8)**n
opt = tf.keras.optimizers.SGD(momentum =0.9, nesterov =True)
callback_lr = tf.keras.callbacks.LearningRateScheduler(scheduler, verbose=1)

# model.compile(opt,loss='categorical_crossentropy')

def mpredict(x):
  return model.predict(x)

y = data_train_test['grade']
X = data_train_test['qTFD']

subjects = data_train_test['baby_ID']


from sklearn.model_selection import LeaveOneGroupOut
logo = LeaveOneGroupOut()
logo.get_n_splits(X,y,subjects)

# LOSO Cross Validation
fold_no = 1
for train_index, test_index in logo.split(X,y,subjects):
	# separate train and test based on group
	X_train, X_test = X.iloc[train_index], X.iloc[test_index]
	y_train, y_test = y.iloc[train_index], y.iloc[test_index]

	# convert X to ndarrays of (n, 256, 112) shape
	# and y to (n, 1)
	X_train = np.reshape(list(X_train), (len(X_train), 256, 112))
	X_test = np.reshape(list(X_test), (len(X_test), 256, 112))
	# y_train = np.reshape(list(y_train), (len(y_train), 1)).astype(np.int32)
	# y_train = list(y_train - 1)
	# y_test = list(y_test)
	y_train = tf.keras.utils.to_categorical(y_train - 1, num_classes=4)
	y_test = tf.keras.utils.to_categorical(y_test, num_classes=4)

	# create model architecture
	model = create_convnet()
	opt = tf.keras.optimizers.SGD(momentum =0.9, nesterov =True)
	callback_lr = tf.keras.callbacks.LearningRateScheduler(scheduler)

	# compile model
	model.compile(opt,loss='categorical_crossentropy')

	# print
	print('------------------------------------------------------------------------') 
	print(f'Training for fold {fold_no} ...')

	# fit model
	history = model.fit(x=X_train, y=y_train, 
		epochs=30, batch_size=128, callbacks=[callback_lr])

	# get metrics
	scores = model.evaluate(x=X_test, y=y_test, verbose=0)
	print(f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%')

	fold_no += 1
	break