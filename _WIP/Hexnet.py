#!/usr/bin/env python3.7


'''****************************************************************************
 * Hexnet.py: TODO
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
# Parameters
################################################################################

model            = 'CNN'
load_model       = None
load_weights     = None

dataset          = 'datasets/MNIST/MNIST.h5'
augment_dataset  = None

tests_dir        = 'tests/tmp'
visualize_model  = None

runs             = 1
epochs           = 1
batch_size       = 32
kernel_size      = (3, 3)
pool_size        = (3, 3)

verbosity_level  = 2

transform_s2h    = False
transform_s2s    = False
transform_rad_o  = 1.0
transform_width  = 64
transform_height = None

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

from datetime import datetime

from core.Hexnet import Hexnet_init

import datasets.datasets  as datasets
import misc.augmentation  as augmentation
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

	model_string     = args.model
	load_model       = args.load_model
	load_weights     = args.load_weights
	save_model       = args.save_model
	save_weights     = args.save_weights

	dataset          = args.dataset
	augment_dataset  = args.augment_dataset

	tests_dir        = args.tests_dir
	show_dataset     = args.show_dataset
	visualize_model  = args.visualize_model
	show_results     = args.show_results

	runs             = args.runs
	epochs           = args.epochs
	batch_size       = args.batch_size
	kernel_size      = args.kernel_size
	pool_size        = args.pool_size

	verbosity_level  = args.verbosity_level

	transform_s2h    = args.transform_s2h
	transform_s2s    = args.transform_s2s
	transform_rad_o  = args.transform_rad_o
	transform_width  = args.transform_width
	transform_height = args.transform_height


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

		if 'train' in augment_dataset:
			train_data = augmentation.augment_images(images=train_data)

		if 'test' in augment_dataset:
			test_data  = augmentation.augment_images(images=test_data)

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

	if runs > 0:
		if tests_dir is not None:
			os.makedirs(tests_dir, exist_ok=True)

		if visualize_model is not None:
			os.makedirs(visualize_model, exist_ok=True)

	train_labels  = [int(np.where(train_classes == label)[0]) for label in train_labels]
	train_classes = list(set(train_labels))
	test_labels   = [int(np.where(test_classes == label)[0])  for label in test_labels]
	test_classes  = list(set(test_labels))

	train_data_shape  = train_data.shape
	test_data_shape   = test_data.shape
	train_test_data_n = 255.0
	train_data        = train_data.reshape(train_data_shape) / train_test_data_n
	test_data         = test_data.reshape(test_data_shape)   / train_test_data_n


	############################################################################
	# Start a new run
	############################################################################

	for run in range(1, runs + 1):
		run_string = f'run={run}/{runs}'

		dataset   = os.path.basename(dataset)
		timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
		run_title = f'{model_string}_{dataset}_{timestamp}_{epochs}-{batch_size}-{kernel_size[0]}-{pool_size[0]}'

		if runs > 1:
			run_title = f'{run_title}_run{run}'


		########################################################################
		# Initialize / load the model
		########################################################################

		Hexnet_print(f'({run_string}) Model initialization')

		input_shape = (train_data_shape[1:4])
		classes     = len(train_classes)

		if load_model is None:
			if 'ResNet' in model_string:
				model = vars(models)[f'model_{model_string}'](n=3, input_shape=input_shape, classes=classes)
			else:
				model = vars(models)[f'model_{model_string}'](input_shape, classes, kernel_size, pool_size)
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

		model.compile(
			optimizer = 'adam',
			loss      = 'sparse_categorical_crossentropy',
			metrics   = ['accuracy'])

		Hexnet_print(f'({run_string}) Training')
		history = model.fit(train_data, train_labels, batch_size, epochs, shuffle = True)
		print_newline()


		########################################################################
		# Visualize filters and feature maps for training and test results
		########################################################################

		if visualize_model is not None:
			Hexnet_print(f'({run_string}) Visualization')

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

		if tests_dir is not None or show_results:
			visualization.visualize_results(history, run_title, tests_dir, show_results)

		print_newline()

		Hexnet_print(f'({run_string}) Test')
		test_loss, test_acc = model.evaluate(test_data, test_labels)
		Hexnet_print(f'({run_string}) test_acc={test_acc} (test_loss={test_loss})')


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

def none_test(value):
	if not value or value == 'None':
		return None
	else:
		return value

def parse_args():
	parser = argparse.ArgumentParser(description='TODO')


	model_choices = [model[0][len('model_'):] for model in inspect.getmembers(
		models, inspect.isfunction) if model[0].startswith('model_')]

	parser.add_argument(
		'--model',
		type    = none_test, nargs = '?',
		default = model,
		choices = model_choices,
		help    = 'TODO')

	parser.add_argument('--load-model',                                      default = load_model,
		help = 'TODO')
	parser.add_argument('--load-weights',                                    default = load_weights,
		help = 'TODO')
	parser.add_argument('--save-model',                                      action  = 'store_true',
		help = 'TODO')
	parser.add_argument('--save-weights',                                    action  = 'store_true',
		help = 'TODO')


	parser.add_argument('--dataset',                                         default = dataset,
		help = 'TODO')

	parser.add_argument(
		'--augment-dataset',
		default = augment_dataset,
		choices = ['train', 'test', 'train test'],
		help    = 'TODO')


	parser.add_argument('--tests-dir',        type = none_test, nargs = '?', default = tests_dir,
		help = 'TODO')
	parser.add_argument('--show-dataset',                                    action  = 'store_true',
		help = 'TODO')
	parser.add_argument('--visualize-model',                                 default = visualize_model,
		help = 'TODO')
	parser.add_argument('--show-results',                                    action  = 'store_true',
		help = 'TODO')

	parser.add_argument('--runs',             type = int,                    default = runs,
		help = 'TODO')
	parser.add_argument('--epochs',           type = int,                    default = epochs,
		help = 'TODO')
	parser.add_argument('--batch-size',       type = int,                    default = batch_size,
		help = 'TODO')
	parser.add_argument('--kernel-size',      type = int,       nargs = '+', default = kernel_size,
		help = 'TODO')
	parser.add_argument('--pool-size',        type = int,       nargs = '+', default = pool_size,
		help = 'TODO')

	parser.add_argument('--verbosity-level',  type = int,                    default = verbosity_level,
		help = 'TODO')

	parser.add_argument('--transform-s2h',    type = none_test, nargs = '?', default = transform_s2h,
		help = 'TODO')
	parser.add_argument('--transform-s2s',    type = none_test, nargs = '?', default = transform_s2s,
		help = 'TODO')
	parser.add_argument('--transform-rad-o',  type = float,                  default = transform_rad_o,
		help = 'TODO')
	parser.add_argument('--transform-width',  type = int,                    default = transform_width,
		help = 'TODO')
	parser.add_argument('--transform-height', type = int,                    default = transform_height,
		help = 'TODO')


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

