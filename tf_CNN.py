"""
This is an introduction to CNN's using Tensorflow 
I am using the MNIST dataset since its relatively small and uses less resources for traning
"""

import tensorflow as tf
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten
from keras.models import Model
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


def CNN():
	inputs = Input(shape=(28,28,1))
	#Layer 1
	X = Conv2D(32,(5,5),padding='same',activation = 'relu',name="Conv1")(inputs)
	X = MaxPooling2D(name="Max1")(X)

	#Layer 2
	X = Conv2D(64,(3,3),padding='same', activation = 'relu',name="Conv2")(X)
	X = MaxPooling2D(name="Max2")(X)

	#Layer 3
	X = Flatten()(X)
	pred_y = Dense(10,activation='softmax',name="Dense1")(X)

	model = Model(inputs = inputs,outputs = pred_y)

	return model

mnist_model = CNN()

mnist_model.compile(optimizer='Adam',loss = 'categorical_crossentropy',metrics=['accuracy'])

train = mnist.train.images
train_reshape = train.reshape(-1,28,28,1)
labels = mnist.train.labels






