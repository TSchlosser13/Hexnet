'''****************************************************************************
 * models_keras.py: Keras Models
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


def model_keras_DenseNet121(input_shape, classes, weights=None, **kwargs):
	return tf.keras.applications.DenseNet121(input_shape=input_shape, classes=classes, weights=weights, **kwargs)

def model_keras_DenseNet169(input_shape, classes, weights=None, **kwargs):
	return tf.keras.applications.DenseNet169(input_shape=input_shape, classes=classes, weights=weights, **kwargs)

def model_keras_DenseNet201(input_shape, classes, weights=None, **kwargs):
	return tf.keras.applications.DenseNet201(input_shape=input_shape, classes=classes, weights=weights, **kwargs)

def model_keras_InceptionResNetV2(input_shape, classes, weights=None, **kwargs):
	return tf.keras.applications.InceptionResNetV2(input_shape=input_shape, classes=classes, weights=weights, **kwargs)

def model_keras_InceptionV3(input_shape, classes, weights=None, **kwargs):
	return tf.keras.applications.InceptionV3(input_shape=input_shape, classes=classes, weights=weights, **kwargs)

def model_keras_MobileNet(input_shape, classes, weights=None, **kwargs):
	return tf.keras.applications.MobileNet(input_shape=input_shape, classes=classes, weights=weights, **kwargs)

def model_keras_MobileNetV2(input_shape, classes, weights=None, **kwargs):
	return tf.keras.applications.MobileNetV2(input_shape=input_shape, classes=classes, weights=weights, **kwargs)

def model_keras_NASNetLarge(input_shape, classes, weights=None, **kwargs):
	return tf.keras.applications.NASNetLarge(input_shape=input_shape, classes=classes, weights=weights, **kwargs)

def model_keras_NASNetMobile(input_shape, classes, weights=None, **kwargs):
	return tf.keras.applications.NASNetMobile(input_shape=input_shape, classes=classes, weights=weights, **kwargs)

def model_keras_ResNet50(input_shape, classes, weights=None, **kwargs):
	return tf.keras.applications.ResNet50(input_shape=input_shape, classes=classes, weights=weights, **kwargs)

def model_keras_ResNet50V2(input_shape, classes, weights=None, **kwargs):
	return tf.keras.applications.ResNet50V2(input_shape=input_shape, classes=classes, weights=weights, **kwargs)

def model_keras_ResNet101(input_shape, classes, weights=None, **kwargs):
	return tf.keras.applications.ResNet101(input_shape=input_shape, classes=classes, weights=weights, **kwargs)

def model_keras_ResNet101V2(input_shape, classes, weights=None, **kwargs):
	return tf.keras.applications.ResNet101V2(input_shape=input_shape, classes=classes, weights=weights, **kwargs)

def model_keras_ResNet152(input_shape, classes, weights=None, **kwargs):
	return tf.keras.applications.ResNet152(input_shape=input_shape, classes=classes, weights=weights, **kwargs)

def model_keras_ResNet152V2(input_shape, classes, weights=None, **kwargs):
	return tf.keras.applications.ResNet152V2(input_shape=input_shape, classes=classes, weights=weights, **kwargs)

def model_keras_VGG16(input_shape, classes, weights=None, **kwargs):
	return tf.keras.applications.VGG16(input_shape=input_shape, classes=classes, weights=weights, **kwargs)

def model_keras_VGG19(input_shape, classes, weights=None, **kwargs):
	return tf.keras.applications.VGG19(input_shape=input_shape, classes=classes, weights=weights, **kwargs)

def model_keras_Xception(input_shape, classes, weights=None, **kwargs):
	return tf.keras.applications.Xception(input_shape=input_shape, classes=classes, weights=weights, **kwargs)

