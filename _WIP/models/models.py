'''****************************************************************************
 * models.py: TODO
 ******************************************************************************
 * v0.1 - 01.03.2019
 *
 * Copyright (c) 2019 Tobias Schlosser (tobias@tobias-schlosser.net)
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 ****************************************************************************'''


import tensorflow as tf

from tensorflow.keras        import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPool2D

from layers.layers  import HConv2D, HMaxPool2D, SConv2D, SMaxPool2D
from models.resnets import model_ResNet_v1, model_ResNet_v2


def convert_model_parameters(kernel_size, pool_size):
	if type(kernel_size) is not tuple:
		kernel_size = tuple(kernel_size)

	if type(pool_size) is not tuple:
		pool_size = tuple(pool_size)

	if len(kernel_size) == 1:
		kernel_size *= 2

	if len(pool_size) == 1:
		pool_size *= 2

	return kernel_size, pool_size


def model_CNN(input_shape, classes, kernel_size, pool_size):
	kernel_size, pool_size = convert_model_parameters(kernel_size, pool_size)

	model = Sequential()

	model.add(Conv2D(filters=32, kernel_size=kernel_size, padding='SAME', activation=tf.nn.relu, input_shape=input_shape))
	model.add(MaxPool2D(pool_size=pool_size, padding='SAME'))
	model.add(Dropout(rate=0.25))
	model.add(Conv2D(filters=64, kernel_size=kernel_size, padding='SAME', activation=tf.nn.relu))
	model.add(MaxPool2D(pool_size=pool_size, padding='SAME'))
	model.add(Dropout(rate=0.25))
	model.add(Flatten())
	model.add(Dense(units=128, activation=tf.nn.relu))
	model.add(Dropout(rate=0.5))
	model.add(Dense(units=classes, activation=tf.nn.softmax))

	return model


def model_CNN_SConv2D(input_shape, classes, kernel_size, pool_size):
	kernel_size, pool_size = convert_model_parameters(kernel_size, pool_size)

	model = Sequential()

	model.add(SConv2D(filters=32, kernel_size=kernel_size, padding='SAME', activation=tf.nn.relu, input_shape=input_shape))
	model.add(MaxPool2D(pool_size=pool_size, padding='SAME'))
	model.add(Dropout(rate=0.25))
	model.add(SConv2D(filters=64, kernel_size=kernel_size, padding='SAME', activation=tf.nn.relu))
	model.add(MaxPool2D(pool_size=pool_size, padding='SAME'))
	model.add(Dropout(rate=0.25))
	model.add(Flatten())
	model.add(Dense(units=128, activation=tf.nn.relu))
	model.add(Dropout(rate=0.5))
	model.add(Dense(units=classes, activation=tf.nn.softmax))

	return model


def model_CNN_SMaxPool2D(input_shape, classes, kernel_size, pool_size):
	kernel_size, pool_size = convert_model_parameters(kernel_size, pool_size)

	model = Sequential()

	model.add(Conv2D(filters=32, kernel_size=kernel_size, padding='SAME', activation=tf.nn.relu, input_shape=input_shape))
	model.add(SMaxPool2D(pool_size=pool_size, padding='SAME'))
	model.add(Dropout(rate=0.25))
	model.add(Conv2D(filters=64, kernel_size=kernel_size, padding='SAME', activation=tf.nn.relu))
	model.add(SMaxPool2D(pool_size=pool_size, padding='SAME'))
	model.add(Dropout(rate=0.25))
	model.add(Flatten())
	model.add(Dense(units=128, activation=tf.nn.relu))
	model.add(Dropout(rate=0.5))
	model.add(Dense(units=classes, activation=tf.nn.softmax))

	return model


def model_SCNN(input_shape, classes, kernel_size, pool_size):
	kernel_size, pool_size = convert_model_parameters(kernel_size, pool_size)

	model = Sequential()

	model.add(SConv2D(filters=32, kernel_size=kernel_size, padding='SAME', activation=tf.nn.relu, input_shape=input_shape))
	model.add(SMaxPool2D(pool_size=pool_size, padding='SAME'))
	model.add(Dropout(rate=0.25))
	model.add(SConv2D(filters=64, kernel_size=kernel_size, padding='SAME', activation=tf.nn.relu))
	model.add(SMaxPool2D(pool_size=pool_size, padding='SAME'))
	model.add(Dropout(rate=0.25))
	model.add(Flatten())
	model.add(Dense(units=128, activation=tf.nn.relu))
	model.add(Dropout(rate=0.5))
	model.add(Dense(units=classes, activation=tf.nn.softmax))

	return model


def model_CNN_HConv2D(input_shape, classes, kernel_size, pool_size):
	kernel_size, pool_size = convert_model_parameters(kernel_size, pool_size)

	model = Sequential()

	model.add(HConv2D(filters=32, kernel_size=kernel_size, padding='SAME', activation=tf.nn.relu, input_shape=input_shape))
	model.add(MaxPool2D(pool_size=pool_size, padding='SAME'))
	model.add(Dropout(rate=0.25))
	model.add(HConv2D(filters=64, kernel_size=kernel_size, padding='SAME', activation=tf.nn.relu))
	model.add(MaxPool2D(pool_size=pool_size, padding='SAME'))
	model.add(Dropout(rate=0.25))
	model.add(Flatten())
	model.add(Dense(units=128, activation=tf.nn.relu))
	model.add(Dropout(rate=0.5))
	model.add(Dense(units=classes, activation=tf.nn.softmax))

	return model


def model_CNN_HMaxPool2D(input_shape, classes, kernel_size, pool_size):
	kernel_size, pool_size = convert_model_parameters(kernel_size, pool_size)

	model = Sequential()

	model.add(Conv2D(filters=32, kernel_size=kernel_size, padding='SAME', activation=tf.nn.relu, input_shape=input_shape))
	model.add(HMaxPool2D(pool_size=pool_size, padding='SAME'))
	model.add(Dropout(rate=0.25))
	model.add(Conv2D(filters=64, kernel_size=kernel_size, padding='SAME', activation=tf.nn.relu))
	model.add(HMaxPool2D(pool_size=pool_size, padding='SAME'))
	model.add(Dropout(rate=0.25))
	model.add(Flatten())
	model.add(Dense(units=128, activation=tf.nn.relu))
	model.add(Dropout(rate=0.5))
	model.add(Dense(units=classes, activation=tf.nn.softmax))

	return model


def model_HCNN(input_shape, classes, kernel_size, pool_size):
	kernel_size, pool_size = convert_model_parameters(kernel_size, pool_size)

	model = Sequential()

	model.add(HConv2D(filters=32, kernel_size=kernel_size, padding='SAME', activation=tf.nn.relu, input_shape=input_shape))
	model.add(HMaxPool2D(pool_size=pool_size, padding='SAME'))
	model.add(Dropout(rate=0.25))
	model.add(HConv2D(filters=64, kernel_size=kernel_size, padding='SAME', activation=tf.nn.relu))
	model.add(HMaxPool2D(pool_size=pool_size, padding='SAME'))
	model.add(Dropout(rate=0.25))
	model.add(Flatten())
	model.add(Dense(units=128, activation=tf.nn.relu))
	model.add(Dropout(rate=0.5))
	model.add(Dense(units=classes, activation=tf.nn.softmax))

	return model


