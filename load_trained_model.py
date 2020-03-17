#!/usr/bin/env python
# coding: utf-8

import numpy as np
import tensorflow as tf
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
import keras
from keras import optimizers
from keras import regularizers
from keras import losses
from keras import layers
from keras.layers import Input, Add, Dropout, concatenate, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, SeparableConv2D, MaxPooling2D, MaxPooling2D, GlobalMaxPooling2D
from keras.models import Model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
from keras.initializers import glorot_uniform
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
import keras.backend as K
K.set_image_data_format('channels_last')
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from matplotlib.pyplot import imshow
from sklearn.metrics import confusion_matrix
import os
import keras.backend.tensorflow_backend as KTF

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def feature_normalize(dataset):
    mu = np.mean(dataset)
    sigma = np.std(dataset)
    return (dataset - mu) / sigma

def one_hot(labels):
    classes = np.unique(labels)
    n_classes = classes.size
    one_hot_labels = np.zeros(labels.shape + (n_classes,))
    for c in classes:
        one_hot_labels[labels == c, c-1] = 1
    return one_hot_labels

acc_x = np.loadtxt(u'C:/Users/dongm/Documents/Project HAR/UCI HAR Dataset/train/Inertial Signals/body_acc_x_train.txt', dtype=float)
acc_y = np.loadtxt(u'C:/Users/dongm/Documents/Project HAR/UCI HAR Dataset/train/Inertial Signals/body_acc_y_train.txt', dtype=float)
acc_z = np.loadtxt(u'C:/Users/dongm/Documents/Project HAR/UCI HAR Dataset/train/Inertial Signals/body_acc_z_train.txt', dtype=float)
body_gyro_x = np.loadtxt(u'C:/Users/dongm/Documents/Project HAR/UCI HAR Dataset/train/Inertial Signals/body_gyro_x_train.txt', dtype=float)
body_gyro_y = np.loadtxt(u'C:/Users/dongm/Documents/Project HAR/UCI HAR Dataset/train/Inertial Signals/body_gyro_y_train.txt', dtype=float)
body_gyro_z = np.loadtxt(u'C:/Users/dongm/Documents/Project HAR/UCI HAR Dataset/train/Inertial Signals/body_gyro_z_train.txt', dtype=float)
total_acc_x = np.loadtxt(u'C:/Users/dongm/Documents/Project HAR/UCI HAR Dataset/train/Inertial Signals/total_acc_x_train.txt', dtype=float)
total_acc_y = np.loadtxt(u'C:/Users/dongm/Documents/Project HAR/UCI HAR Dataset/train/Inertial Signals/total_acc_y_train.txt', dtype=float)
total_acc_z = np.loadtxt(u'C:/Users/dongm/Documents/Project HAR/UCI HAR Dataset/train/Inertial Signals/total_acc_z_train.txt', dtype=float)
train_x = np.dstack([total_acc_x, total_acc_y, total_acc_z, acc_x, acc_y, acc_z, body_gyro_x, body_gyro_y, body_gyro_z])
train_x = train_x.reshape(len(train_x), 1, 128, 9)
train_x[:,0,:,0] = train_x[:,0,:,0] * 9.80665
train_x[:,0,:,1] = train_x[:,0,:,1] * 9.80665
train_x[:,0,:,2] = train_x[:,0,:,2] * 9.80665
train_x[:,0,:,3] = train_x[:,0,:,3] * 9.80665
train_x[:,0,:,4] = train_x[:,0,:,4] * 9.80665
train_x[:,0,:,5] = train_x[:,0,:,5] * 9.80665

labels = np.loadtxt(u'C:/Users/dongm/Documents/Project HAR/UCI HAR Dataset/train/y_train.txt', dtype=int)
train_y = one_hot(labels)
acc_x = np.loadtxt(u'C:/Users/dongm/Documents/Project HAR/UCI HAR Dataset/test/Inertial Signals/body_acc_x_test.txt', dtype=float)
acc_y = np.loadtxt(u'C:/Users/dongm/Documents/Project HAR/UCI HAR Dataset/test/Inertial Signals/body_acc_y_test.txt', dtype=float)
acc_z = np.loadtxt(u'C:/Users/dongm/Documents/Project HAR/UCI HAR Dataset/test/Inertial Signals/body_acc_z_test.txt', dtype=float)
body_gyro_x = np.loadtxt('C:/Users/dongm/Documents/Project HAR/UCI HAR Dataset/test/Inertial Signals/body_gyro_x_test.txt', dtype=float)
body_gyro_y = np.loadtxt('C:/Users/dongm/Documents/Project HAR/UCI HAR Dataset/test/Inertial Signals/body_gyro_y_test.txt', dtype=float)
body_gyro_z = np.loadtxt('C:/Users/dongm/Documents/Project HAR/UCI HAR Dataset/test/Inertial Signals/body_gyro_z_test.txt', dtype=float)
total_acc_x = np.loadtxt('C:/Users/dongm/Documents/Project HAR/UCI HAR Dataset/test/Inertial Signals/total_acc_x_test.txt', dtype=float)
total_acc_y = np.loadtxt('C:/Users/dongm/Documents/Project HAR/UCI HAR Dataset/test/Inertial Signals/total_acc_y_test.txt', dtype=float)
total_acc_z = np.loadtxt('C:/Users/dongm/Documents/Project HAR/UCI HAR Dataset/test/Inertial Signals/total_acc_z_test.txt', dtype=float)
test_x = np.dstack([total_acc_x, total_acc_y, total_acc_z, acc_x, acc_y, acc_z, body_gyro_x, body_gyro_y, body_gyro_z])
test_x = test_x.reshape(len(test_x), 1, 128, 9)
test_x[:,0,:,0] = test_x[:,0,:,0] * 9.80665
test_x[:,0,:,1] = test_x[:,0,:,1] * 9.80665
test_x[:,0,:,2] = test_x[:,0,:,2] * 9.80665
test_x[:,0,:,3] = test_x[:,0,:,3] * 9.80665
test_x[:,0,:,4] = test_x[:,0,:,4] * 9.80665
test_x[:,0,:,5] = test_x[:,0,:,5] * 9.80665
labels = np.loadtxt(u'C:/Users/dongm/Documents/Project HAR/UCI HAR Dataset/test/y_test.txt', dtype=int)
test_y = one_hot(labels)

train_x[:,0,:,0] = feature_normalize(train_x[:,0,:,0])
train_x[:,0,:,1] = feature_normalize(train_x[:,0,:,1])
train_x[:,0,:,2] = feature_normalize(train_x[:,0,:,2])
train_x[:,0,:,3] = feature_normalize(train_x[:,0,:,3])
train_x[:,0,:,4] = feature_normalize(train_x[:,0,:,4])
train_x[:,0,:,5] = feature_normalize(train_x[:,0,:,5])
train_x[:,0,:,6] = feature_normalize(train_x[:,0,:,6])
train_x[:,0,:,7] = feature_normalize(train_x[:,0,:,7])
train_x[:,0,:,8] = feature_normalize(train_x[:,0,:,8])

test_x[:,0,:,0] = feature_normalize(test_x[:,0,:,0])
test_x[:,0,:,1] = feature_normalize(test_x[:,0,:,1])
test_x[:,0,:,2] = feature_normalize(test_x[:,0,:,2])
test_x[:,0,:,3] = feature_normalize(test_x[:,0,:,3])
test_x[:,0,:,4] = feature_normalize(test_x[:,0,:,4])
test_x[:,0,:,5] = feature_normalize(test_x[:,0,:,5])
test_x[:,0,:,6] = feature_normalize(test_x[:,0,:,6])
test_x[:,0,:,7] = feature_normalize(test_x[:,0,:,7])
test_x[:,0,:,8] = feature_normalize(test_x[:,0,:,8])

train_aux = np.loadtxt(u'C:/Users/dongm/Documents/Project HAR/UCI HAR Dataset/train/X_train.txt', dtype=float) #
test_aux = np.loadtxt(u'C:/Users/dongm/Documents/Project HAR/UCI HAR Dataset/test/X_test.txt', dtype=float) #

train_aux = feature_normalize(train_aux)
test_aux = feature_normalize(test_aux)

batch_size = 128
num_channel = 6
num_class = 6

filter_stride = 1
pool_size = 11

training_epochs = 9
total_batchs = train_x.shape[0] // batch_size

def inception_net(input_shape=(1,128,9),classes=6):
    X_input = Input(input_shape)
    X_aux_input = Input((561,))

    X = SeparableConv2D(filters=64, kernel_size=(1,1), strides=(filter_stride,1), padding='same', depth_multiplier=1, activation='relu',
        depthwise_initializer=glorot_uniform(seed=921212), pointwise_initializer=glorot_uniform(seed=921212), bias_initializer='zeros')(X_input)
    X = MaxPooling2D((1,pool_size), strides=(3,1))(X)
    X = SeparableConv2D(filters=64, kernel_size=(1,1), strides=(filter_stride,1), padding='same', depth_multiplier=1, activation='relu',
        depthwise_initializer=glorot_uniform(seed=921212), pointwise_initializer=glorot_uniform(seed=921212), bias_initializer='zeros')(X)
    X = MaxPooling2D((1,pool_size), strides=(3,1))(X)
    X = SeparableConv2D(filters=64, kernel_size=(1,1), strides=(filter_stride,1), padding='same', depth_multiplier=1, activation='relu',
        depthwise_initializer=glorot_uniform(seed=921212), pointwise_initializer=glorot_uniform(seed=921212), bias_initializer='zeros')(X)
    X = MaxPooling2D((1,pool_size), strides=(3,1))(X)
    X = SeparableConv2D(filters=64, kernel_size=(1,1), strides=(filter_stride,1), padding='same', depth_multiplier=1, activation='relu',
        depthwise_initializer=glorot_uniform(seed=921212), pointwise_initializer=glorot_uniform(seed=921212), bias_initializer='zeros')(X)
    X = MaxPooling2D((1,pool_size), strides=(3,1))(X)

    X1 = SeparableConv2D(filters=64, kernel_size=(5,1), strides=(filter_stride,1), padding='same', depth_multiplier=1, activation='relu',
        depthwise_initializer=glorot_uniform(seed=921212), pointwise_initializer=glorot_uniform(seed=921212), bias_initializer='zeros')(X_input)
    X1 = MaxPooling2D((1,pool_size), strides=(3,1))(X1)
    X1 = SeparableConv2D(filters=64, kernel_size=(5,1), strides=(filter_stride,1), padding='same', depth_multiplier=1, activation='relu',
        depthwise_initializer=glorot_uniform(seed=921212), pointwise_initializer=glorot_uniform(seed=921212), bias_initializer='zeros')(X1)
    X1 = MaxPooling2D((1,pool_size), strides=(3,1))(X1)
    X1 = SeparableConv2D(filters=64, kernel_size=(5,1), strides=(filter_stride,1), padding='same', depth_multiplier=1, activation='relu',
        depthwise_initializer=glorot_uniform(seed=921212), pointwise_initializer=glorot_uniform(seed=921212), bias_initializer='zeros')(X1)
    X1 = MaxPooling2D((1,pool_size), strides=(3,1))(X1)
    X1 = SeparableConv2D(filters=64, kernel_size=(5,1), strides=(filter_stride,1), padding='same', depth_multiplier=1, activation='relu',
        depthwise_initializer=glorot_uniform(seed=921212), pointwise_initializer=glorot_uniform(seed=921212), bias_initializer='zeros')(X1)
    X1 = MaxPooling2D((1,pool_size), strides=(3,1))(X1)

    X2 = SeparableConv2D(filters=64, kernel_size=(9,1), strides=(filter_stride,1), padding='same', depth_multiplier=1, activation='relu',
        depthwise_initializer=glorot_uniform(seed=921212), pointwise_initializer=glorot_uniform(seed=921212), bias_initializer='zeros')(X_input)
    X2 = MaxPooling2D((1,pool_size), strides=(3,1))(X2)
    X2 = SeparableConv2D(filters=64, kernel_size=(9,1), strides=(filter_stride,1), padding='same', depth_multiplier=1, activation='relu',
        depthwise_initializer=glorot_uniform(seed=921212), pointwise_initializer=glorot_uniform(seed=921212), bias_initializer='zeros')(X2)
    X2 = MaxPooling2D((1,pool_size), strides=(3,1))(X2)
    X2 = SeparableConv2D(filters=64, kernel_size=(9,1), strides=(filter_stride,1), padding='same', depth_multiplier=1, activation='relu',
        depthwise_initializer=glorot_uniform(seed=921212), pointwise_initializer=glorot_uniform(seed=921212), bias_initializer='zeros')(X2)
    X2 = MaxPooling2D((1,pool_size), strides=(3,1))(X2)
    X2 = SeparableConv2D(filters=64, kernel_size=(9,1), strides=(filter_stride,1), padding='same', depth_multiplier=1, activation='relu',
        depthwise_initializer=glorot_uniform(seed=921212), pointwise_initializer=glorot_uniform(seed=921212), bias_initializer='zeros')(X2)
    X2 = MaxPooling2D((1,pool_size), strides=(3,1))(X2)

    X3 = SeparableConv2D(filters=64, kernel_size=(13,1), strides=(filter_stride,1), padding='same', depth_multiplier=1, activation='relu',
        depthwise_initializer=glorot_uniform(seed=921212), pointwise_initializer=glorot_uniform(seed=921212), bias_initializer='zeros')(X_input)
    X3 = MaxPooling2D((1,pool_size), strides=(3,1))(X3)
    X3 = SeparableConv2D(filters=64, kernel_size=(13,1), strides=(filter_stride,1), padding='same', depth_multiplier=1, activation='relu',
        depthwise_initializer=glorot_uniform(seed=921212), pointwise_initializer=glorot_uniform(seed=921212), bias_initializer='zeros')(X3)
    X3 = MaxPooling2D((1,pool_size), strides=(3,1))(X3)
    X3 = SeparableConv2D(filters=64, kernel_size=(13,1), strides=(filter_stride,1), padding='same', depth_multiplier=1, activation='relu',
        depthwise_initializer=glorot_uniform(seed=921212), pointwise_initializer=glorot_uniform(seed=921212), bias_initializer='zeros')(X3)
    X3 = MaxPooling2D((1,pool_size), strides=(3,1))(X3)
    X3 = SeparableConv2D(filters=64, kernel_size=(13,1), strides=(filter_stride,1), padding='same', depth_multiplier=1, activation='relu',
        depthwise_initializer=glorot_uniform(seed=921212), pointwise_initializer=glorot_uniform(seed=921212), bias_initializer='zeros')(X3)
    X3 = MaxPooling2D((1,pool_size), strides=(3,1))(X3)

    X4 = SeparableConv2D(filters=64, kernel_size=(17,1), strides=(filter_stride,1), padding='same', depth_multiplier=1, activation='relu',
        depthwise_initializer=glorot_uniform(seed=921212), pointwise_initializer=glorot_uniform(seed=921212), bias_initializer='zeros')(X_input)
    X4 = MaxPooling2D((1,pool_size), strides=(3,1))(X4)
    X4 = SeparableConv2D(filters=64, kernel_size=(17,1), strides=(filter_stride,1), padding='same', depth_multiplier=1, activation='relu',
        depthwise_initializer=glorot_uniform(seed=921212), pointwise_initializer=glorot_uniform(seed=921212), bias_initializer='zeros')(X4)
    X4 = MaxPooling2D((1,pool_size), strides=(3,1))(X4)
    X4 = SeparableConv2D(filters=64, kernel_size=(17,1), strides=(filter_stride,1), padding='same', depth_multiplier=1, activation='relu',
        depthwise_initializer=glorot_uniform(seed=921212), pointwise_initializer=glorot_uniform(seed=921212), bias_initializer='zeros')(X4)
    X4 = MaxPooling2D((1,pool_size), strides=(3,1))(X4)
    X4 = SeparableConv2D(filters=64, kernel_size=(17,1), strides=(filter_stride,1), padding='same', depth_multiplier=1, activation='relu',
        depthwise_initializer=glorot_uniform(seed=921212), pointwise_initializer=glorot_uniform(seed=921212), bias_initializer='zeros')(X4)
    X4 = MaxPooling2D((1,pool_size), strides=(3,1))(X4)

    X5 = SeparableConv2D(filters=64, kernel_size=(21,1), strides=(filter_stride,1), padding='same', depth_multiplier=1, activation='relu',
        depthwise_initializer=glorot_uniform(seed=921212), pointwise_initializer=glorot_uniform(seed=921212), bias_initializer='zeros')(X_input)
    X5 = MaxPooling2D((1,pool_size), strides=(3,1))(X5)
    X5 = SeparableConv2D(filters=64, kernel_size=(21,1), strides=(filter_stride,1), padding='same', depth_multiplier=1, activation='relu',
        depthwise_initializer=glorot_uniform(seed=921212), pointwise_initializer=glorot_uniform(seed=921212), bias_initializer='zeros')(X5)
    X5 = MaxPooling2D((1,pool_size), strides=(3,1))(X5)
    X5 = SeparableConv2D(filters=64, kernel_size=(21,1), strides=(filter_stride,1), padding='same', depth_multiplier=1, activation='relu',
        depthwise_initializer=glorot_uniform(seed=921212), pointwise_initializer=glorot_uniform(seed=921212), bias_initializer='zeros')(X5)
    X5 = MaxPooling2D((1,pool_size), strides=(3,1))(X5)
    X5 = SeparableConv2D(filters=64, kernel_size=(21,1), strides=(filter_stride,1), padding='same', depth_multiplier=1, activation='relu',
        depthwise_initializer=glorot_uniform(seed=921212), pointwise_initializer=glorot_uniform(seed=921212), bias_initializer='zeros')(X5)
    X5 = MaxPooling2D((1,pool_size), strides=(3,1))(X5)

    X6 = SeparableConv2D(filters=64, kernel_size=(25,1), strides=(filter_stride,1), padding='same', depth_multiplier=1, activation='relu',
        depthwise_initializer=glorot_uniform(seed=921212), pointwise_initializer=glorot_uniform(seed=921212), bias_initializer='zeros')(X_input)
    X6 = MaxPooling2D((1,pool_size), strides=(3,1))(X6)
    X6 = SeparableConv2D(filters=64, kernel_size=(25,1), strides=(filter_stride,1), padding='same', depth_multiplier=1, activation='relu',
        depthwise_initializer=glorot_uniform(seed=921212), pointwise_initializer=glorot_uniform(seed=921212), bias_initializer='zeros')(X6)
    X6 = MaxPooling2D((1,pool_size), strides=(3,1))(X6)
    X6 = SeparableConv2D(filters=64, kernel_size=(25,1), strides=(filter_stride,1), padding='same', depth_multiplier=1, activation='relu',
        depthwise_initializer=glorot_uniform(seed=921212), pointwise_initializer=glorot_uniform(seed=921212), bias_initializer='zeros')(X6)
    X6 = MaxPooling2D((1,pool_size), strides=(3,1))(X6)
    X6 = SeparableConv2D(filters=64, kernel_size=(25,1), strides=(filter_stride,1), padding='same', depth_multiplier=1, activation='relu',
        depthwise_initializer=glorot_uniform(seed=921212), pointwise_initializer=glorot_uniform(seed=921212), bias_initializer='zeros')(X6)
    X6 = MaxPooling2D((1,pool_size), strides=(3,1))(X6)

    X7 = SeparableConv2D(filters=64, kernel_size=(29,1), strides=(filter_stride,1), padding='same', depth_multiplier=1, activation='relu',
        depthwise_initializer=glorot_uniform(seed=921212), pointwise_initializer=glorot_uniform(seed=921212), bias_initializer='zeros')(X_input)
    X7 = MaxPooling2D((1,pool_size), strides=(3,1))(X7)
    X7 = SeparableConv2D(filters=64, kernel_size=(29,1), strides=(filter_stride,1), padding='same', depth_multiplier=1, activation='relu',
        depthwise_initializer=glorot_uniform(seed=921212), pointwise_initializer=glorot_uniform(seed=921212), bias_initializer='zeros')(X7)
    X7 = MaxPooling2D((1,pool_size), strides=(3,1))(X7)
    X7 = SeparableConv2D(filters=64, kernel_size=(29,1), strides=(filter_stride,1), padding='same', depth_multiplier=1, activation='relu',
        depthwise_initializer=glorot_uniform(seed=921212), pointwise_initializer=glorot_uniform(seed=921212), bias_initializer='zeros')(X7)
    X7 = MaxPooling2D((1,pool_size), strides=(3,1))(X7)
    X7 = SeparableConv2D(filters=64, kernel_size=(29,1), strides=(filter_stride,1), padding='same', depth_multiplier=1, activation='relu',
        depthwise_initializer=glorot_uniform(seed=921212), pointwise_initializer=glorot_uniform(seed=921212), bias_initializer='zeros')(X7)
    X7 = MaxPooling2D((1,pool_size), strides=(3,1))(X7)

    X8 = SeparableConv2D(filters=64, kernel_size=(33,1), strides=(filter_stride,1), padding='same', depth_multiplier=1, activation='relu',
        depthwise_initializer=glorot_uniform(seed=921212), pointwise_initializer=glorot_uniform(seed=921212), bias_initializer='zeros')(X_input)
    X8 = MaxPooling2D((1,pool_size), strides=(3,1))(X8)
    X8 = SeparableConv2D(filters=64, kernel_size=(33,1), strides=(filter_stride,1), padding='same', depth_multiplier=1, activation='relu',
        depthwise_initializer=glorot_uniform(seed=921212), pointwise_initializer=glorot_uniform(seed=921212), bias_initializer='zeros')(X8)
    X8 = MaxPooling2D((1,pool_size), strides=(3,1))(X8)
    X8 = SeparableConv2D(filters=64, kernel_size=(33,1), strides=(filter_stride,1), padding='same', depth_multiplier=1, activation='relu',
        depthwise_initializer=glorot_uniform(seed=921212), pointwise_initializer=glorot_uniform(seed=921212), bias_initializer='zeros')(X8)
    X8 = MaxPooling2D((1,pool_size), strides=(3,1))(X8)
    X8 = SeparableConv2D(filters=64, kernel_size=(33,1), strides=(filter_stride,1), padding='same', depth_multiplier=1, activation='relu',
        depthwise_initializer=glorot_uniform(seed=921212), pointwise_initializer=glorot_uniform(seed=921212), bias_initializer='zeros')(X8)
    X8 = MaxPooling2D((1,pool_size), strides=(3,1))(X8)

    X_collection = concatenate([X, X1, X2, X3, X4, X5, X6, X7, X8], axis=3)

    X_collection = Flatten()(X_collection)
    X_collection = concatenate([X_collection, X_aux_input])
    X_collection = Dense(4096, activation='relu', kernel_initializer=glorot_uniform(seed=921212))(X_collection)
    X_collection = Dense(128, activation='tanh', kernel_initializer=glorot_uniform(seed=921212))(X_collection)

    X_collection = Dense(classes, activation='softmax', kernel_initializer=glorot_uniform(seed=921212))(X_collection)
    model = Model(inputs=[X_input, X_aux_input], output=X_collection, name="inception_net")
    return model


class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch':[], 'epoch':[]}
        self.accuracy = {'batch':[], 'epoch':[]}
        self.val_loss = {'batch':[], 'epoch':[]}
        self.val_acc = {'batch':[], 'epoch':[]}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_acc'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))
        checkpoint = ModelCheckpoint(filepath='./model/best_weights.h5',monitor='val_acc',mode='auto' ,save_best_only='True')

    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        plt.figure()
        # acc
        plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')
        # loss
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        if loss_type == 'epoch':
            # val_acc
            plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
            # val_loss
            plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('acc-loss')
        plt.legend(loc="upper right")
        plt.show()

from keras.models import load_model
model = load_model('./model.hdf5')
loss, accuracy = model.evaluate([test_x, test_aux], test_y, batch_size=batch_size, verbose=0)
print('\ninitial loaded model loss',loss)
print('initial loaded modelaccuracy',accuracy)
model = inception_net(input_shape=(1,128,9), classes=6)
adam = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(loss="categorical_crossentropy", optimizer=adam, metrics=["accuracy"])
model.fit([train_x, train_aux], train_y, epochs=2, batch_size=128,validation_data=([test_x, test_aux], test_y), verbose=2, shuffle=True, callbacks = None)
loss, acc = model.evaluate([test_x, test_aux], test_y, batch_size=batch_size, verbose=0)
print('test_loss',loss,'test_accuracy',acc)

model.save_model('./model_updated.hdf5')
from keras.models import model_from_json
json_string = model.to_json()
model = model_from_json(json_string)

print(json_string)
