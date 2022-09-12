'''****************************************************************************
 * models.py: Square and Hexagonal Models Header
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


################################################################################
# Imports
################################################################################

import tensorflow as tf

from tensorflow.keras        import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPool2D

from layers.layers import HConv2D, HMaxPool2D, HSampling2D, SConv2D, SMaxPool2D, SSampling2D

from models.resnets        import *
from models.contrib.models import *




################################################################################
# Conv2D and MaxPool2D tests
################################################################################


def model_CNN(input_shape, classes, kernel_size, pool_size):
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




################################################################################
# Sampling2D tests
################################################################################


def model_CNN_SSampling2D_test(input_shape, classes, kernel_size, pool_size):
	model = Sequential()

	model.add(Conv2D(filters=32, kernel_size=kernel_size, padding='SAME', activation=tf.nn.relu, input_shape=input_shape))
	model.add(SSampling2D(size_factor = (0.5, 0.5), interpolation = 'bilinear'))
	model.add(Dropout(rate=0.25))
	model.add(Conv2D(filters=64, kernel_size=kernel_size, padding='SAME', activation=tf.nn.relu))
	model.add(SSampling2D(size_factor = (0.5, 0.5), interpolation = 'bilinear'))
	model.add(Dropout(rate=0.25))
	model.add(Flatten())
	model.add(Dense(units=128, activation=tf.nn.relu))
	model.add(Dropout(rate=0.5))
	model.add(Dense(units=classes, activation=tf.nn.sigmoid))

	return model


def model_CNN_HSampling2D_square_test(input_shape, classes, kernel_size, pool_size):
	model = Sequential()

	model.add(Conv2D(filters=32, kernel_size=kernel_size, padding='SAME', activation=tf.nn.relu, input_shape=input_shape))
	model.add(HSampling2D(size_factor = (0.5, 0.5), interpolation = 'bilinear', mode = 'square_interpolation'))
	model.add(Dropout(rate=0.25))
	model.add(Conv2D(filters=64, kernel_size=kernel_size, padding='SAME', activation=tf.nn.relu))
	model.add(HSampling2D(size_factor = (0.5, 0.5), interpolation = 'bilinear', mode = 'square_interpolation'))
	model.add(Dropout(rate=0.25))
	model.add(Flatten())
	model.add(Dense(units=128, activation=tf.nn.relu))
	model.add(Dropout(rate=0.5))
	model.add(Dense(units=classes, activation=tf.nn.sigmoid))

	return model


def model_CNN_HSampling2D_hexagonal_test(input_shape, classes, kernel_size, pool_size):
	model = Sequential()

	model.add(Conv2D(filters=32, kernel_size=kernel_size, padding='SAME', activation=tf.nn.relu, input_shape=input_shape))
	model.add(HSampling2D(size_factor = (0.5, 0.5), interpolation = 'bilinear', mode = 'hexagonal_interpolation'))
	model.add(Dropout(rate=0.25))
	model.add(Conv2D(filters=64, kernel_size=kernel_size, padding='SAME', activation=tf.nn.relu))
	model.add(HSampling2D(size_factor = (0.5, 0.5), interpolation = 'bilinear', mode = 'hexagonal_interpolation'))
	model.add(Dropout(rate=0.25))
	model.add(Flatten())
	model.add(Dense(units=128, activation=tf.nn.relu))
	model.add(Dropout(rate=0.5))
	model.add(Dense(units=classes, activation=tf.nn.sigmoid))

	return model




################################################################################
# Multi-label tests
################################################################################


def model_CNN_multilabel_test(input_shape, classes, kernel_size, pool_size):
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
	model.add(Dense(units=classes, activation=tf.nn.sigmoid))

	return model




################################################################################
# Custom models' tests
################################################################################


def model_SCNN_custom_v1_ks2_ps2(input_shape, classes):
	kernel_size = (2, 2)
	pool_size   = (2, 2)

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


def model_SCNN_custom_v1_ks2_ps3(input_shape, classes):
	kernel_size = (2, 2)
	pool_size   = (3, 3)

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


def model_SCNN_custom_v1_ks3_ps2(input_shape, classes):
	kernel_size = (3, 3)
	pool_size   = (2, 2)

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


def model_SCNN_custom_v1_ks3_ps3(input_shape, classes):
	kernel_size = (3, 3)
	pool_size   = (3, 3)

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


def model_HCNN_custom_v1(input_shape, classes):
	kernel_size = (3, 3)
	pool_size   = (3, 3)

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


def model_SCNN_custom_v2_ps2(input_shape, classes):
	kernel_size = (3, 3)
	pool_size   = (2, 2)

	model = Sequential()

	model.add(SConv2D(filters=32, kernel_size=kernel_size, padding='SAME', activation=tf.nn.relu, input_shape=input_shape))
	model.add(SConv2D(filters=32, kernel_size=kernel_size, strides=(2, 2), padding='SAME', activation=tf.nn.relu))
	model.add(Dropout(rate=0.25))
	model.add(SConv2D(filters=64, kernel_size=kernel_size, padding='SAME', activation=tf.nn.relu))
	model.add(SMaxPool2D(pool_size=pool_size, padding='SAME'))
	model.add(Dropout(rate=0.25))
	model.add(Flatten())
	model.add(Dense(units=128, activation=tf.nn.relu))
	model.add(Dropout(rate=0.5))
	model.add(Dense(units=classes, activation=tf.nn.softmax))

	return model


def model_SCNN_custom_v2_ps3(input_shape, classes):
	kernel_size = (3, 3)
	pool_size   = (3, 3)

	model = Sequential()

	model.add(SConv2D(filters=32, kernel_size=kernel_size, padding='SAME', activation=tf.nn.relu, input_shape=input_shape))
	model.add(SConv2D(filters=32, kernel_size=kernel_size, strides=(2, 2), padding='SAME', activation=tf.nn.relu))
	model.add(Dropout(rate=0.25))
	model.add(SConv2D(filters=64, kernel_size=kernel_size, padding='SAME', activation=tf.nn.relu))
	model.add(SMaxPool2D(pool_size=pool_size, padding='SAME'))
	model.add(Dropout(rate=0.25))
	model.add(Flatten())
	model.add(Dense(units=128, activation=tf.nn.relu))
	model.add(Dropout(rate=0.5))
	model.add(Dense(units=classes, activation=tf.nn.softmax))

	return model


def model_HCNN_custom_v2(input_shape, classes):
	kernel_size = (3, 3)
	pool_size   = (3, 3)

	model = Sequential()

	model.add(HConv2D(filters=32, kernel_size=kernel_size, padding='SAME', activation=tf.nn.relu, input_shape=input_shape))
	model.add(HConv2D(filters=32, kernel_size=kernel_size, strides=(2, 2), padding='SAME', activation=tf.nn.relu))
	model.add(Dropout(rate=0.25))
	model.add(HConv2D(filters=64, kernel_size=kernel_size, padding='SAME', activation=tf.nn.relu))
	model.add(HMaxPool2D(pool_size=pool_size, padding='SAME'))
	model.add(Dropout(rate=0.25))
	model.add(Flatten())
	model.add(Dense(units=128, activation=tf.nn.relu))
	model.add(Dropout(rate=0.5))
	model.add(Dense(units=classes, activation=tf.nn.softmax))

	return model

