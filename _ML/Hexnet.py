#!/usr/bin/env python3.7


'''****************************************************************************
 * Hexnet.py: The Hexagonal Image Processing Framework
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
# Default parameters
################################################################################

model              = 'CNN'
load_model         = None
load_weights       = None

dataset            = 'datasets/MNIST/MNIST.h5'
augment_dataset    = None
augmenter          = 'simple'
augmentation_level = 1

tests_dir          = 'tests/tmp'
visualize_model    = None

runs               = 1
epochs             = 1
batch_size         = 32

cnn_kernel_size    = (3, 3)
cnn_pool_size      = (3, 3)

verbosity_level    = 2

transform_s2h      = False
transform_s2s      = False
transform_rad_o    = 1.0
transform_width    = 64
transform_height   = None

disable_tensorflow_warnings = True


################################################################################
# Imports
################################################################################

import argparse
import inspect
import os
import sys

import numpy      as np
import tensorflow as tf

from datetime          import datetime
from matplotlib.pyplot import imsave

from core.Hexnet import Hexnet_init

import datasets.datasets  as datasets
import misc.augmenters    as augmenters
import misc.visualization as visualization
import models.models      as models

from misc.misc import Hexnet_print, print_newline


################################################################################
# Disable TensorFlow warnings
################################################################################

if disable_tensorflow_warnings:
	os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

	import tensorflow.python.util.deprecation as deprecation
	deprecation._PRINT_DEPRECATION_WARNINGS = False


################################################################################
# Run Hexnet
################################################################################

def run(args):

	model_string       = args.model
	load_model         = args.load_model
	load_weights       = args.load_weights
	save_model         = args.save_model
	save_weights       = args.save_weights

	dataset            = args.dataset
	augment_dataset    = args.augment_dataset
	augmenter_string   = args.augmenter
	augmentation_level = args.augmentation_level

	tests_dir          = args.tests_dir
	show_dataset       = args.show_dataset
	visualize_model    = args.visualize_model
	show_results       = args.show_results

	runs               = args.runs
	epochs             = args.epochs
	batch_size         = args.batch_size

	cnn_kernel_size    = args.cnn_kernel_size
	cnn_pool_size      = args.cnn_pool_size

	verbosity_level    = args.verbosity_level

	transform_s2h      = args.transform_s2h
	transform_s2s      = args.transform_s2s
	transform_rad_o    = args.transform_rad_o
	transform_width    = args.transform_width
	transform_height   = args.transform_height


	model = None

	train_classes = []
	train_data    = []
	train_labels  = []
	test_classes  = []
	test_data     = []
	test_labels   = []


	############################################################################
	# Transform the dataset
	############################################################################

	if transform_s2h or transform_s2h is None or transform_s2s or transform_s2s is None:
		Hexnet_init()

		Hexnet_print('Dataset transformation')

		if transform_s2h or transform_s2h is None:
			if transform_s2h is None:
				transform_s2h = f'{dataset}_s2h'

			datasets.transform_dataset(
				dataset         = dataset,
				output_dir      = transform_s2h,
				mode            = 's2h',
				rad_o           = transform_rad_o,
				method          = 0,
				verbosity_level = verbosity_level)

		if transform_s2s or transform_s2s is None:
			if transform_s2s is None:
				transform_s2s = f'{dataset}_s2s'

			datasets.transform_dataset(
				dataset         = dataset,
				output_dir      = transform_s2s,
				mode            = 's2s',
				width           = transform_width,
				height          = transform_height,
				method          = 0,
				verbosity_level = verbosity_level)


	############################################################################
	# Load, augment, and show the dataset
	############################################################################

	((train_classes, train_data, train_labels), (test_classes, test_data, test_labels)) = datasets.load_dataset(
		dataset         = dataset,
		create_h5       = True,
		verbosity_level = verbosity_level)

	print_newline()

	if augment_dataset is not None:
		Hexnet_print('Dataset augmentation')

		augmenter = vars(augmenters)[f'augmenter_{augmenter_string}']

		if 'custom' in augmenter_string:
			augmenter = augmenter()
		else:
			augmenter = augmenter(augmentation_level)

		if 'train' in augment_dataset:
			train_data = augmenter(images=train_data)

		if 'test' in augment_dataset:
			test_data  = augmenter(images=test_data)

		print_newline()

	print_newline()

	if show_dataset:
		datasets.show_dataset_classes(
			train_classes,
			train_data,
			train_labels,
			test_classes,
			test_data,
			test_labels,
			max_images_per_class   =  1,
			max_classes_to_display = 10)


	if model_string is None:
		return 0


	############################################################################
	# Prepare the dataset
	############################################################################

	train_labels  = [int(np.where(train_classes == label)[0]) for label in train_labels]
	train_classes = list(set(train_labels))
	test_labels   = [int(np.where(test_classes == label)[0]) for label in test_labels]
	test_classes  = list(set(test_labels))

	train_test_data_n = 255
	train_data        = train_data / train_test_data_n
	test_data         = test_data  / train_test_data_n

	if 'autoencoder' in model_string:
		min_size_factor = 2**5 # TODO

		if train_data.shape[1] % min_size_factor:
			padding_h = min_size_factor - train_data.shape[1] % min_size_factor
			padding_h = (int(padding_h / 2) + padding_h % 2, int(padding_h / 2))

		if train_data.shape[2] % min_size_factor:
			padding_w = min_size_factor - train_data.shape[2] % min_size_factor
			padding_w = (int(padding_w / 2) + padding_w % 2, int(padding_w / 2))

		pad_width = ((0, 0), padding_h, padding_w, (0, 0))

		train_data = np.pad(train_data, pad_width, mode='constant', constant_values=0)
		test_data  = np.pad(test_data,  pad_width, mode='constant', constant_values=0)


	############################################################################
	# Start a new run
	############################################################################

	for run in range(1, runs + 1):
		run_string = f'run={run}/{runs}'

		dataset   = os.path.basename(dataset)
		timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')

		if 'CNN' in model_string and not 'custom' in model_string:
			run_title = f'{model_string}_{dataset}_{timestamp}_{epochs}-{batch_size}-{cnn_kernel_size[0]}-{cnn_pool_size[0]}'
		else:
			run_title = f'{model_string}_{dataset}_{timestamp}_{epochs}-{batch_size}'

		if runs > 1:
			run_title = f'{run_title}_run{run}'


		########################################################################
		# Initialize / load the model
		########################################################################

		Hexnet_print(f'({run_string}) Model initialization')

		input_shape = train_data.shape[1:4]
		classes     = len(train_classes)

		if load_model is None:
			model = vars(models)[f'model_{model_string}']

			if 'autoencoder' in model_string:
				model = model(input_shape)
			elif 'custom' in model_string:
				model = model(input_shape, classes)
			elif 'CNN' in model_string:
				model = model(input_shape, classes, cnn_kernel_size, cnn_pool_size)
			else:
				model = model(input_shape, classes)
		else:
			model = tf.keras.models.load_model(load_model)

		if load_weights is not None:
			model.load_weights(load_weights)

		print_newline()

		Hexnet_print(f'({run_string}) Model summary')
		model.summary()
		print_newline()


		########################################################################
		# Fit the model
		########################################################################

		if not 'autoencoder' in model_string:
			loss    = 'sparse_categorical_crossentropy'
			metrics = ['accuracy']
		else:
			loss    = 'mse'
			metrics = None

		model.compile(
			optimizer = 'adam',
			loss      = loss,
			metrics   = metrics)

		Hexnet_print(f'({run_string}) Training')

		if not 'autoencoder' in model_string:
			history = model.fit(train_data, train_labels, batch_size, epochs, shuffle=True)
		else:
			history = model.fit(train_data, train_data, batch_size, epochs, shuffle=True)

		print_newline()


		########################################################################
		# Visualize filters and feature maps for training and test results
		########################################################################

		if visualize_model is not None:
			Hexnet_print(f'({run_string}) Visualization')

			os.makedirs(visualize_model, exist_ok=True)

			visualization.visualize_model(
				model,
				test_classes,
				test_data,
				test_labels,
				output_dir           = visualize_model,
				max_images_per_class = 10,
				verbosity_level      = verbosity_level)

			print_newline()

		Hexnet_print(f'({run_string}) History')
		Hexnet_print(f'({run_string}) history.history.keys()={history.history.keys()}')

		if tests_dir or show_results:
			os.makedirs(tests_dir, exist_ok=True)
			visualization.visualize_results(history, run_title, tests_dir, show_results)

		print_newline()

		Hexnet_print(f'({run_string}) Test')

		if not 'autoencoder' in model_string:
			test_loss, test_acc = model.evaluate(test_data, test_labels)
		else:
			test_loss = model.evaluate(test_data, test_data)

		predictions = model.predict(test_data)

		if not 'autoencoder' in model_string:
			predictions_classes = predictions.argmax(axis=-1)
			Hexnet_print(f'({run_string}) test_acc={test_acc}, test_loss={test_loss}')
		else:
			Hexnet_print(f'({run_string}) test_loss={test_loss}')

		if tests_dir is not None:
			tests_dir_predictions = os.path.join(tests_dir, f'{run_title}_predictions')

			if not 'autoencoder' in model_string:
				np.savetxt(f'{tests_dir_predictions}.csv',         predictions,                   delimiter=',')
				np.savetxt(f'{tests_dir_predictions}_classes.csv', predictions_classes, fmt='%i', delimiter=',')
			else:
				image_filename_base = os.path.basename(tests_dir_predictions)
				os.makedirs(tests_dir_predictions, exist_ok=True)

				for image_counter, image in enumerate(predictions):
					image_filename = f'{image_filename_base}_image{image_counter}.png'
					imsave(os.path.join(tests_dir_predictions, image_filename), image)


		########################################################################
		# Save the model
		########################################################################

		if save_model and tests_dir is not None:
			model.save(os.path.join(tests_dir, f'{run_title}_model.h5'))

		if save_weights and tests_dir is not None:
			model.save_weights(os.path.join(tests_dir, f'{run_title}_weights.h5'))


		if run < runs:
			print_newline()
			print_newline()


	return 0


################################################################################
# parse_args
################################################################################

def parse_args():
	parser = argparse.ArgumentParser(description='Hexnet - The Hexagonal Image Processing Framework')


	model_choices = [model[0][len('model_'):] for model in inspect.getmembers(
		models, inspect.isfunction) if model[0].startswith('model_')]

	parser.add_argument(
		'--model',
		nargs   = '?',
		default = model,
		choices = model_choices,
		help    = 'model to train and test: choices are generated from models/models.py '
		          '(providing no argument disables training and testing)')

	parser.add_argument('--load-model',   default = load_model,   help = 'load model from file')
	parser.add_argument('--load-weights', default = load_weights, help = 'load model weights from file')
	parser.add_argument('--save-model',   action  = 'store_true', help = 'save model to file')
	parser.add_argument('--save-weights', action  = 'store_true', help = 'save model weights to file')


	parser.add_argument('--dataset', default = dataset, help = 'load dataset from file or directory')

	parser.add_argument(
		'--augment-dataset',
		nargs   = '+',
		default = augment_dataset,
		choices = ['train', 'test'],
		help    = 'set(s) to augment')

	augmenter_choices = [augmenter[0][len('augmenter_'):] for augmenter in inspect.getmembers(
		augmenters, inspect.isfunction) if augmenter[0].startswith('augmenter_')]

	parser.add_argument(
		'--augmenter',
		default = augmenter,
		choices = augmenter_choices,
		help    = 'augmenter for augmentation: choices are generated from misc/augmenters.py')

	parser.add_argument('--augmentation-level', type = int, default = augmentation_level, help = 'augmentation level')


	parser.add_argument('--tests-dir',                      nargs = '?', default = tests_dir,
		help = 'tests output directory (providing no argument disables the tests output)')
	parser.add_argument('--show-dataset',                                action  = 'store_true',
		help = 'show the dataset')
	parser.add_argument('--visualize-model',                             default = visualize_model,
		help = 'visualize the model\'s filters and feature maps after training')
	parser.add_argument('--show-results',                                action  = 'store_true',
		help = 'show the test results')

	parser.add_argument('--runs',             type = int,                default = runs,
		help = 'training runs')
	parser.add_argument('--epochs',           type = int,                default = epochs,
		help = 'training epochs')
	parser.add_argument('--batch-size',       type = int,                default = batch_size,
		help = 'training batch size')

	parser.add_argument('--cnn-kernel-size',  type = int,   nargs = '+', default = cnn_kernel_size,
		help = 'CNN models kernel size')
	parser.add_argument('--cnn-pool-size',    type = int,   nargs = '+', default = cnn_pool_size,
		help = 'CNN models pooling size')

	parser.add_argument('--verbosity-level',  type = int,                default = verbosity_level,
		help = 'verbosity level (default is 2, maximum is 3)')

	parser.add_argument('--transform-s2h',                  nargs = '?', default = transform_s2h,
		help = 'enable square to hexagonal image transformation')
	parser.add_argument('--transform-s2s',                  nargs = '?', default = transform_s2s,
		help = 'enable square to square image transformation')
	parser.add_argument('--transform-rad-o',  type = float,              default = transform_rad_o,
		help = 'square to hexagonal image transformation hexagonal pixels outer radius')
	parser.add_argument('--transform-width',  type = int,                default = transform_width,
		help = 'square to square image transformation output width')
	parser.add_argument('--transform-height', type = int,                default = transform_height,
		help = 'square to square image transformation output height')


	return parser.parse_args()


################################################################################
# main
################################################################################

if __name__ == '__main__':
	args = parse_args()

	Hexnet_print(f'args={args}')
	print_newline()

	status = run(args)

	sys.exit(status)


