# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Activation, Dropout
from tensorflow.keras.losses import sparse_categorical_crossentropy

from keras.utils import np_utils
import tensorflow.keras
import pickle
import numpy as np
# import matplotlib.pyplot as plt
import random
from math import sqrt
import math
import time


'''
 Remove previous weights, bias, inputs, etc.. 
'''
from tensorflow.python.framework import ops
ops.reset_default_graph()

def neural_network_configuration():
    # Note the input shape is the desired size of the image 32x32 with 3 bytes color
    '''
	  model = tf.keras.models.Sequential([
    # Note the input shape is the desired size of the image 32x32 with 3 bytes color
    # This is the first convolution
    tf.keras.layers.Conv2D(10, (3,3), activation='relu', input_shape=(32, 32, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The second convolution
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The third convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The fourth convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # # The fifth convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # Flatten the results to feed into a DNN
    tf.keras.layers.Flatten(),
    # 512 neuron hidden layer
    tf.keras.layers.Dense(128, activation='relu'),
    # Only 1 output neuron. It will contain a value from 0-1 where 0 for 1 class ('dandelions') and 1 for the other ('grass')
    tf.keras.layers.Dense(10, activation='sigmoid')
    ])
	'''
	
    '''
	# Model sequential
    model = Sequential()
    # 1st hidden layer (we also need to tell the input dimension)
    # 10 neurons, but you can change to play a bit
    model.add(Dense(10, input_dim=1, activation='sigmoid'))
     
    model.add(Dense(5, activation='sigmoid'))
    ## 2nd hidden layer - YOU MAY TEST THIS
    #model.add(Dense(10, activation='sigmoid'))
    # Output layer
    #model.add(Dense(1, activation='sigmoid'))
    model.add(Dense(1, activation='tanh'))
    '''
    model = tf.keras.Sequential([
    #tf.keras.layers.Flatten(input_shape=(1,)),
    #tf.keras.layers.Dense(32, activation='relu'),
    #tf.keras.layers.Dense(1),
    # tf.keras.layers.Conv2D((32,3), padding='same', activation='relu'),
    #tf.keras.layers.MaxPooling2D(),
    #tf.keras.layers.Flatten(),
    #tf.keras.layers.Dense(128, activation='sigmoid'),
    #tf.keras.layers.Flatten(),
    # tf.keras.layers.Flatten(),
    #tf.keras.layers.Dense(10)
    
    ])
    input_shape = 100,3072,1
    no_classes = 10
    # img_width, img_height, img_num_channels = 32, 32, 3
    loss_function = sparse_categorical_crossentropy
    optimizer = 'Adam'
   
    
    # Learning rate has huge effect
    tf.keras.optimizers.SGD(lr=0.1)
    #model.compile(optimizer='sgd', loss='mse', metrics=['mse'])
    
    model.compile(optimizer='sgd',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
    
    return model

# normalize pixels from 0..255 to between 0..1 
def normalize_pixel_values(data_array):
	# convert from integers to floats
	img_array = data_array.astype('float32')	
	# normalize to range 0-1
	img_array = img_array / 255.0
	# return normalized image array
	return img_array
    
def load_cifar10_data_batch(batch_id):
    with open('./cifardata/data_batch_' + str(batch_id), mode='rb') as file:
        # note the encoding type is 'latin1'
        batch = pickle.load(file, encoding='latin1')
        
    features = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
    labels = batch['labels']        
    return features, labels

def load_cifar10_test_batch():
    with open('./cifardata/test_batch', mode='rb') as file:
        # note the encoding type is 'latin1'
        batch = pickle.load(file, encoding='latin1')
        
    features = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
    labels = batch['labels']        
    return features, labels

def load_label_names():
    return ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def one_hot_encoding(x):
    num_classes = 10
    encoded = np_utils.to_categorical(x, num_classes)
    return encoded

def runCNNtrain():  
    
    # Initialize the variables
    train = dict()
    test = dict()
    trainX = []
    trainY = []
    testX = []

    samples_start = 0
    samples = 100
    model = neural_network_configuration()
    
    testXx, testY = load_cifar10_test_batch()
    
    testY = one_hot_encoding(testY)[samples_start:samples]
    testXx = normalize_pixel_values(testXx)
    
 
    # Loop over all batches
    n_batches = 1
    for batch_i in range(1, n_batches + 1):
        print("Runnning batch: ", batch_i)
        trainXx, trainY = load_cifar10_data_batch(batch_i)
        
    
        
        trainX = trainXx[samples_start:samples]#*3072].reshape(-1, 3072)
        testX = testXx[samples_start:samples]#*3072].reshape(-1, 3072)
        
        # Works only on dicts
        #trainX = random.sample(trainX, 500)
        #testX = random.sample(testX, 500)        
        
        # one hot encode data
        Y = one_hot_encoding(trainY)
        Y = Y[samples_start:samples]
            
        for Xindex in range((10)):            
            print("I: ", Xindex)
            X = normalize_pixel_values(trainX)            
            X = X.reshape(samples,3072)
            
            test = testX.reshape(samples,3072)      
            
            # fit model
            history = model.fit(X[Xindex], test[Xindex], epochs=5, verbose=1)                                                           
           
            # evaluate model
           # _, acc = model.evaluate(X[Xindex], testX[Xindex], verbose=1)
            
            print("Eval: ", 0)
            
        model.summary()
        
        cifarmodel = model
        cifarProbModel = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
        predictions = []
        
        for Xindex in range(len(X)):
            predictions = cifarProbModel.predict(X[Xindex])
            classPrediction = np.argmax(predictions)
            
            print(np.unravel_index(np.argmax((predictions), axis=0), predictions.shape)[1])
            print(predictions, " predictions size: ", len(predictions))
            print("Class for target: ", testY[Xindex], " --> ", classPrediction)
            
runCNNtrain()