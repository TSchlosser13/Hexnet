'''****************************************************************************
 * visual_inspection.py: Visual Inspection Models
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


from tensorflow.keras        import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPool2D


'''
	A Novel Visual Fault Detection and Classification System for Semiconductor Manufacturing Using Stacked Hybrid Convolutional Neural Networks
	Schlosser, Tobias ; Beuth, Frederik ; Friedrich, Michael ; Kowerko, Danny
'''

def model_CNN_custom_Schlosser2019_good_bad_chips(input_shape, classes):
	model = Sequential()

	model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu', input_shape=input_shape))
	model.add(Conv2D(filters=48, kernel_size=(3, 3), activation='relu'))
	model.add(MaxPool2D(pool_size=(3, 3)))
	model.add(Dropout(rate=0.25))
	model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
	model.add(Conv2D(filters=96, kernel_size=(3, 3), activation='relu'))
	model.add(MaxPool2D(pool_size=(2, 2)))
	model.add(Dropout(rate=0.25))
	model.add(Conv2D(filters=144, kernel_size=(3, 3), activation='relu'))
	model.add(Conv2D(filters=192, kernel_size=(3, 3), activation='relu'))
	model.add(MaxPool2D(pool_size=(2, 2)))
	model.add(Dropout(rate=0.25))
	model.add(Flatten())
	model.add(Dense(units=192, activation='relu'))
	model.add(Dropout(rate=0.5))
	model.add(Dense(units=classes, activation='softmax'))

	return model

def model_CNN_custom_Schlosser2019_in_out_chips(input_shape, classes):
	model = Sequential()

	model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
	model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
	model.add(MaxPool2D(pool_size=(2, 2)))
	model.add(Dropout(rate=0.25))
	model.add(Flatten())
	model.add(Dense(units=128, activation='relu'))
	model.add(Dropout(rate=0.5))
	model.add(Dense(units=classes, activation='softmax'))

	return model

def model_CNN_custom_Schlosser2019_good_bad_streets(input_shape, classes):
	model = Sequential()

	model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu', input_shape=input_shape))
	model.add(Conv2D(filters=48, kernel_size=(3, 3), activation='relu'))
	model.add(MaxPool2D(pool_size=(3, 3)))
	model.add(Dropout(rate=0.25))
	model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
	model.add(Conv2D(filters=96, kernel_size=(3, 3), activation='relu'))
	model.add(MaxPool2D(pool_size=(2, 2)))
	model.add(Dropout(rate=0.25))
	model.add(Conv2D(filters=144, kernel_size=(3, 3), activation='relu'))
	model.add(Conv2D(filters=192, kernel_size=(3, 3), activation='relu'))
	model.add(MaxPool2D(pool_size=(1, 3))) # (2, 2) -> (1, 3)
	model.add(Dropout(rate=0.25))
	model.add(Flatten())
	model.add(Dense(units=192, activation='relu'))
	model.add(Dropout(rate=0.5))
	model.add(Dense(units=classes, activation='softmax'))

	return model




'''
	Convolutional Neural Network for Wafer Surface Defect Classification and the Detection of Unknown Defect Class
	Cheon, Sejune ; Lee, Hankang ; Kim, Chang Ouk ; Lee, Seok Hyung
'''

def model_CNN_custom_Cheon2019(input_shape, classes):
	model = Sequential()

	model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
	model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))
	model.add(MaxPool2D(pool_size=(2, 2)))
	model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
	model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
	model.add(MaxPool2D(pool_size=(2, 2)))
	model.add(Flatten())
	model.add(Dense(units=512, activation='relu'))
	model.add(Dropout(rate=0.5))
	model.add(Dense(units=classes, activation='softmax'))

	return model


'''
	Wafer Map Defect Pattern Classification and Image Retrieval Using Convolutional Neural Network
	Nakazawa, Takeshi ; Kulkarni, Deepak V.
'''

def model_CNN_custom_Nakazawa2018(input_shape, classes):
	model = Sequential()

	model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
	model.add(MaxPool2D(pool_size=(2, 2)))
	model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))
	model.add(MaxPool2D(pool_size=(2, 2)))
	model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
	model.add(MaxPool2D(pool_size=(2, 2)))
	model.add(Flatten())
	model.add(Dense(units=256, activation='sigmoid'))
	model.add(Dropout(rate=0.5))
	model.add(Dense(units=classes, activation='softmax'))

	return model


'''
	Deep Learning for Classification of the Chemical Composition of Particle Defects on Semiconductor Wafers
	O'Leary, Jared ; Sawlani, Kapil ; Mesbah, Ali
'''

def model_CNN_custom_OLeary2020(input_shape, classes):
	model = Sequential()

	model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu', input_shape=input_shape))
	model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'))
	model.add(MaxPool2D(pool_size=(2, 2)))
	model.add(Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu'))
	model.add(Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu'))
	model.add(MaxPool2D(pool_size=(2, 2)))
	model.add(Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu'))
	model.add(Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu'))
	model.add(Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu'))
	model.add(MaxPool2D(pool_size=(2, 2)))
	model.add(Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu'))
	model.add(Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu'))
	model.add(Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu'))
	model.add(MaxPool2D(pool_size=(2, 2)))
	model.add(Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu'))
	model.add(Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu'))
	model.add(Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu'))
	model.add(MaxPool2D(pool_size=(2, 2)))
	model.add(Flatten())
	model.add(Dense(units=4096, activation='tanh'))
	model.add(Dropout(rate=0.5))
	model.add(Dense(units=4096, activation='tanh'))
	model.add(Dropout(rate=0.5))
	model.add(Dense(units=4096, activation='tanh'))
	model.add(Dropout(rate=0.5))
	model.add(Dense(units=classes, activation='softmax'))

	return model


