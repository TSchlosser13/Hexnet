#!/usr/bin/env python3.7


'''****************************************************************************
 * Hexnet.py: The Hexagonal Machine Learning Module
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
resize_dataset     = None
crop_dataset       = None
augment_dataset    = None
augmenter          = 'simple'
augmentation_level = 1

tests_dir          = 'tests/tmp'
visualize_model    = None

batch_size         = 32
epochs             =  1
loss               = None
runs               =  1
validation_split   =  0.0

cnn_kernel_size    = (3, 3)
cnn_pool_size      = (3, 3)

verbosity_level    = 2

transform_s2h      = False
transform_s2s      = False
transform_rad_o    =  1.0
transform_width    = 64
transform_height   = None

disable_tensorflow_warnings = True


################################################################################
# Disable TensorFlow warnings
################################################################################

import os

if disable_tensorflow_warnings:
	os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

	import tensorflow.python.util.deprecation as deprecation
	deprecation._PRINT_DEPRECATION_WARNINGS = False


################################################################################
# Imports
################################################################################

import argparse
import inspect
import sklearn
import sys

import numpy      as np
import tensorflow as tf

from datetime          import datetime
from matplotlib.pyplot import imsave

import datasets.datasets  as datasets
import misc.augmenters    as augmenters
import misc.losses        as losses
import misc.visualization as visualization
import models.models      as models

from core.Hexnet import Hexnet_init
from misc.misc   import Hexnet_print, print_newline


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
	resize_dataset     = args.resize_dataset
	crop_dataset       = args.crop_dataset
	augment_dataset    = args.augment_dataset
	augmenter_string   = args.augmenter
	augmentation_level = args.augmentation_level

	tests_dir          = args.tests_dir
	show_dataset       = args.show_dataset
	visualize_model    = args.visualize_model
	show_results       = args.show_results

	batch_size         = args.batch_size
	epochs             = args.epochs
	loss_string        = args.loss
	runs               = args.runs
	validation_split   = args.validation_split

	cnn_kernel_size    = args.cnn_kernel_size
	cnn_pool_size      = args.cnn_pool_size

	verbosity_level    = args.verbosity_level

	transform_s2h      = args.transform_s2h
	transform_s2s      = args.transform_s2s
	transform_rad_o    = args.transform_rad_o
	transform_width    = args.transform_width
	transform_height   = args.transform_height


	if model_string is not None:
		model_is_provided = True

		model_is_custom      = True if 'custom'      in model_string else False
		model_is_standalone  = True if 'standalone'  in model_string else False

		model_is_autoencoder = True if 'autoencoder' in model_string else False
		model_is_CNN         = True if 'CNN'         in model_string else False
		model_is_GAN         = True if 'GAN'         in model_string else False
	else:
		model_is_provided = False

	if augmenter_string is not None:
		augmenter_is_custom = True if 'custom' in augmenter_string else False

	if loss_string is not None:
		loss_is_provided = True

		loss_is_subpixel_loss = True if ('s2s' in loss_string or 's2h' in loss_string) else False
	else:
		loss_is_provided = False


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
	# Load the dataset
	############################################################################

	((train_classes, train_data, train_labels), (test_classes, test_data, test_labels)) = datasets.load_dataset(
		dataset         = dataset,
		create_h5       = True,
		verbosity_level = verbosity_level)


	############################################################################
	# Resize and crop the dataset
	############################################################################

	if resize_dataset is not None:
		(train_data, test_data) = datasets.resize_dataset(dataset_s = (train_data, test_data), resize_string = resize_dataset)

	if crop_dataset is not None:
		(train_data, test_data) = datasets.crop_dataset(dataset_s = (train_data, test_data), crop_string = crop_dataset)

	# TODO
	if model_is_provided and (model_is_autoencoder or model_is_GAN):
		if model_is_autoencoder:
			min_size_factor = 2**5
		else:
			min_size_factor = 2**4

		if train_data.shape[1] % min_size_factor:
			padding_h = min_size_factor - train_data.shape[1] % min_size_factor
			padding_h = (int(padding_h / 2) + padding_h % 2, int(padding_h / 2))
		else:
			padding_h = (0, 0)

		if train_data.shape[2] % min_size_factor:
			padding_w = min_size_factor - train_data.shape[2] % min_size_factor
			padding_w = (int(padding_w / 2) + padding_w % 2, int(padding_w / 2))
		else:
			padding_w = (0, 0)

		pad_width = ((0, 0), padding_h, padding_w, (0, 0))

		train_data = np.pad(train_data, pad_width, mode='constant', constant_values=0)
		test_data  = np.pad(test_data,  pad_width, mode='constant', constant_values=0)


	############################################################################
	# Prepare the dataset
	############################################################################

	class_labels_are_digits = True

	for class_label in train_classes:
		if not class_label.decode().isdigit():
			class_labels_are_digits = False
			break

	if class_labels_are_digits:
		train_labels = np.asarray([int(label.decode()) for label in train_labels])
		test_labels  = np.asarray([int(label.decode()) for label in test_labels])
	else:
		train_labels = np.asarray([int(np.where(train_classes == label)[0]) for label in train_labels])
		test_labels  = np.asarray([int(np.where(test_classes  == label)[0]) for label in test_labels])

	train_classes = list(set(train_labels))
	test_classes  = list(set(test_labels))

	if class_labels_are_digits:
		train_labels  -= min(train_classes)
		test_labels   -= min(test_classes)
		train_classes -= min(train_classes)
		test_classes  -= min(test_classes)

	train_test_data_n  = 255
	train_test_data_n /= 2
	train_data         = (train_data - train_test_data_n) / train_test_data_n
	test_data          = (test_data  - train_test_data_n) / train_test_data_n


	############################################################################
	# Augment the dataset
	############################################################################

	print_newline()

	if augment_dataset is not None:
		Hexnet_print('Dataset augmentation')

		augmenter = vars(augmenters)[f'augmenter_{augmenter_string}']

		if augmenter_is_custom:
			augmenter = augmenter()
		else:
			augmenter = augmenter(augmentation_level)

		if 'train' in augment_dataset:
			train_data = augmenter(images=train_data)

		if 'test' in augment_dataset:
			test_data = augmenter(images=test_data)

		print_newline()

	print_newline()


	############################################################################
	# Show the dataset
	############################################################################

	if show_dataset:
		train_data_for_visualization = np.clip(train_data + 0.5, 0, 1)
		test_data_for_visualization  = np.clip(test_data  + 0.5, 0, 1)

		datasets.show_dataset_classes(
			train_classes,
			train_data_for_visualization,
			train_labels,
			test_classes,
			test_data_for_visualization,
			test_labels,
			max_images_per_class   =  1,
			max_classes_to_display = 10)


	############################################################################
	# No model was provided - returning
	############################################################################

	if not model_is_provided:
		Hexnet_print('No model provided.')
		return 0


	############################################################################
	# Shuffle the dataset
	############################################################################

	train_data, train_labels = sklearn.utils.shuffle(train_data, train_labels)


	############################################################################
	# Start a new run
	############################################################################

	for run in range(1, runs + 1):
		run_string = f'run={run}/{runs}'

		dataset   = os.path.basename(dataset)
		timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')

		run_title = f'{model_string}_{dataset}_{timestamp}_epochs{epochs}-bs{batch_size}'

		if runs > 1:
			run_title = f'{run_title}_run{run}'


		########################################################################
		# Initialize / load the model
		########################################################################

		Hexnet_print(f'({run_string}) Model initialization')

		input_shape  = train_data.shape[1:4]
		output_shape = test_data.shape[1:4]
		classes      = len(train_classes)

		if load_model is None:
			model = vars(models)[f'model_{model_string}']

			if model_is_custom or model_is_standalone:
				model = model(input_shape, classes)
			elif model_is_autoencoder:
				model = model(input_shape)
			elif model_is_CNN:
				model = model(input_shape, classes, cnn_kernel_size, cnn_pool_size)
			else:
				model = model(input_shape, classes)
		else:
			model = tf.keras.models.load_model(load_model)

		if load_weights is not None:
			model.load_weights(load_weights)

		print_newline()


		########################################################################
		# Fit the model
		########################################################################

		if not loss_is_provided:
			if not model_is_autoencoder:
				loss = 'sparse_categorical_crossentropy'
			else:
				loss = 'mse'
		else:
			if not loss_is_subpixel_loss:
				loss = vars(losses)[f'loss_{loss_string}']()
			else:
				loss = vars(losses)[f'loss_{loss_string}'](input_shape, output_shape)

		if not model_is_autoencoder:
			metrics = ['accuracy']
		else:
			metrics = None

		if not model_is_standalone:
			model.compile(optimizer='adam', loss=loss, metrics=metrics)
		else:
			model.compile()

		Hexnet_print(f'({run_string}) Model summary')
		model.summary()
		print_newline()

		Hexnet_print(f'({run_string}) Training')

		if model_is_standalone and tests_dir is not None:
			os.makedirs(tests_dir, exist_ok=True)

		if model_is_standalone:
			model.fit(train_data, train_labels, batch_size, epochs, tests_dir, run_title)
		elif model_is_autoencoder:
			history = model.fit(train_data, train_data, batch_size, epochs, validation_split=validation_split)
		else:
			history = model.fit(train_data, train_labels, batch_size, epochs, validation_split=validation_split)

		print_newline()


		########################################################################
		# Visualize filters and feature maps for training and test results
		########################################################################

		if not model_is_standalone:
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

		if model_is_standalone:
			model.evaluate(test_data, test_labels, batch_size, epochs=10, tests_dir=tests_dir, run_title=run_title)
		elif model_is_autoencoder:
			test_loss = model.evaluate(test_data, test_data)
		else:
			test_loss, test_acc = model.evaluate(test_data, test_labels)

		if not model_is_standalone:
			predictions = model.predict(test_data)

			if not model_is_autoencoder:
				predictions_classes = predictions.argmax(axis=-1)
				Hexnet_print(f'({run_string}) test_acc={test_acc:.8f}, test_loss={test_loss:.8f}')
			else:
				Hexnet_print(f'({run_string}) test_loss={test_loss:.8f}')

			if tests_dir is not None:
				run_title_predictions = f'{run_title}_predictions'
				tests_dir_predictions = os.path.join(tests_dir, run_title_predictions)

				if not model_is_autoencoder:
					np.savetxt(f'{tests_dir_predictions}.csv',         predictions,         delimiter=',')
					np.savetxt(f'{tests_dir_predictions}_classes.csv', predictions_classes, delimiter=',', fmt='%i')
				else:
					os.makedirs(tests_dir_predictions, exist_ok=True)

					for image_counter, (image, label) in enumerate(zip(predictions, test_labels)):
						image_filename = f'label{label}_image{image_counter}.png'
						imsave(os.path.join(tests_dir_predictions, image_filename), image)


		########################################################################
		# Save the model
		########################################################################

		if not model_is_standalone and tests_dir is not None:
			if save_model:
				model.save(os.path.join(tests_dir, f'{run_title}_model.h5'))

			if save_weights:
				model.save_weights(os.path.join(tests_dir, f'{run_title}_weights.h5'))


		if run < runs:
			print_newline()
			print_newline()


	return 0


################################################################################
# parse_args
################################################################################

def parse_args():
	parser = argparse.ArgumentParser(description='Hexnet: The Hexagonal Machine Learning Module')


	model_choices     = [model[0][len('model_'):] for model in inspect.getmembers(models, inspect.isfunction) if model[0].startswith('model_')]
	augmenter_choices = [augmenter[0][len('augmenter_'):] for augmenter in inspect.getmembers(augmenters, inspect.isfunction) if augmenter[0].startswith('augmenter_')]
	loss_choices      = [loss[0][len('loss_'):] for loss in inspect.getmembers(losses, inspect.isclass) if loss[0].startswith('loss_')]

	augment_dataset_choices = ['train', 'test']

	parser.add_argument(
		'--model',
		nargs   = '?',
		default = model,
		choices = model_choices,
		help    = 'model to train and test: choices are generated from models/models.py (providing no argument disables training and testing)')

	parser.add_argument(
		'--augment-dataset',
		nargs   = '+',
		default = augment_dataset,
		choices = augment_dataset_choices,
		help    = 'set(s) to augment')

	parser.add_argument(
		'--augmenter',
		default = augmenter,
		choices = augmenter_choices,
		help    = 'augmenter for augmentation: choices are generated from misc/augmenters.py')

	parser.add_argument(
		'--loss',
		default = loss,
		choices = loss_choices,
		help    = 'custom loss for training and testing: choices are generated from misc/losses.py')


	parser.add_argument('--load-model',                                    default = load_model,         help = 'load model from file')
	parser.add_argument('--load-weights',                                  default = load_weights,       help = 'load model weights from file')
	parser.add_argument('--save-model',                                    action  = 'store_true',       help = 'save model to file')
	parser.add_argument('--save-weights',                                  action  = 'store_true',       help = 'save model weights to file')

	parser.add_argument('--dataset',                                       default = dataset,            help = 'load dataset from file or directory')
	parser.add_argument('--resize-dataset',                                default = resize_dataset,     help = 'resize dataset using "HxW" (e.g. 32x32)')
	parser.add_argument('--crop-dataset',                                  default = crop_dataset,       help = 'crop dataset using "HxW" with offset "+Y+X" (e.g. 32x32+2+2, 32x32, or +2+2)')
	parser.add_argument('--augmentation-level', type = int,                default = augmentation_level, help = 'augmentation level')

	parser.add_argument('--tests-dir',                        nargs = '?', default = tests_dir,          help = 'tests output directory (providing no argument disables the tests output)')
	parser.add_argument('--show-dataset',                                  action  = 'store_true',       help = 'show the dataset')
	parser.add_argument('--visualize-model',                               default = visualize_model,    help = 'visualize the model\'s filters and feature maps after training')
	parser.add_argument('--show-results',                                  action  = 'store_true',       help = 'show the test results')

	parser.add_argument('--batch-size',         type = int,                default = batch_size,         help = 'training batch size')
	parser.add_argument('--epochs',             type = int,                default = epochs,             help = 'training epochs')
	parser.add_argument('--runs',               type = int,                default = runs,               help = 'training runs')
	parser.add_argument('--validation-split',   type = float,              default = validation_split,   help = 'fraction of the training data to be used as validation data')

	parser.add_argument('--cnn-kernel-size',    type = int,   nargs = '+', default = cnn_kernel_size,    help = 'CNN models kernel size')
	parser.add_argument('--cnn-pool-size',      type = int,   nargs = '+', default = cnn_pool_size,      help = 'CNN models pooling size')

	parser.add_argument('--verbosity-level',    type = int,                default = verbosity_level,    help = 'verbosity level (default is 2, maximum is 3)')

	parser.add_argument('--transform-s2h',                    nargs = '?', default = transform_s2h,      help = 'enable square to hexagonal image transformation')
	parser.add_argument('--transform-s2s',                    nargs = '?', default = transform_s2s,      help = 'enable square to square image transformation')
	parser.add_argument('--transform-rad-o',    type = float,              default = transform_rad_o,    help = 'square to hexagonal image transformation hexagonal pixels outer radius')
	parser.add_argument('--transform-width',    type = int,                default = transform_width,    help = 'square to square image transformation output width')
	parser.add_argument('--transform-height',   type = int,                default = transform_height,   help = 'square to square image transformation output height')


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


