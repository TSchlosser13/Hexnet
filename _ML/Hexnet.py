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

model               = 'CNN'
load_model          = False
load_weights        = False

dataset             = '../../Hexnet_datasets/MNIST/MNIST.h5'
create_dataset      = False
rand_seed           = 6
resize_dataset      = False
crop_dataset        = False
pad_dataset         = False
augment_dataset     = False
augmenter           = 'simple'
augmentation_level  = 1
augmentation_size   = 1

chunk_size          = 1000

output_dir          = 'tests/tmp'
visualize_colormap  = None
classes_per_set     = 6
samples_per_class   = 3

optimizer           = 'adam'
metrics             = 'auto'
batch_size          = 32
epochs              =  2
loss                = 'auto'
runs                =  1
validation_split    =  0.1

cnn_kernel_size     = (3, 3)
cnn_pool_size       = (3, 3)
resnet_stacks       = 3
resnet_n            = 3
resnet_filter_size  = 1.0

verbosity_level     = 2

transform_s2h       = False
transform_h2s       = False
transform_h2h       = False
transform_s2s       = False
transform_s2h_rad_o = 1.0
transform_h2s_len   = 1.0
transform_h2h_rad_o = 1.0
transform_s2s_res   = (64, 64)

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
import ast
import inspect
import sklearn
import sys

import numpy      as np
import tensorflow as tf

from datetime          import datetime
from matplotlib.pyplot import imsave
from pprint            import pprint
from time              import time
from tqdm              import tqdm

import datasets.datasets  as datasets
import misc.augmenters    as augmenters
import misc.losses        as losses
import misc.visualization as visualization
import models.models      as models

from core.Hexnet import Hexnet_init
from misc.misc   import array_to_one_hot_array, Hexnet_print, print_newline


################################################################################
# Enable TensorFlow memory growth
################################################################################

for gpu in tf.config.experimental.list_physical_devices('GPU'):
	tf.config.experimental.set_memory_growth(gpu, True)


################################################################################
# Start Hexnet
################################################################################

def run(args):

	############################################################################
	# Parameters
	############################################################################

	disable_training    = args.disable_training
	disable_testing     = args.disable_testing
	disable_output      = args.disable_output
	enable_tensorboard  = args.enable_tensorboard

	model_string        = args.model
	load_model          = args.load_model
	load_weights        = args.load_weights
	save_model          = args.save_model
	save_weights        = args.save_weights

	dataset             = args.dataset
	create_dataset      = args.create_dataset
	disable_rand        = args.disable_rand
	rand_seed           = args.rand_seed
	create_h5           = args.create_h5
	resize_dataset      = args.resize_dataset
	crop_dataset        = args.crop_dataset
	pad_dataset         = args.pad_dataset
	augment_dataset     = args.augment_dataset
	augmenter_string    = args.augmenter
	augmentation_level  = args.augmentation_level
	augmentation_size   = args.augmentation_size

	chunk_size          = args.chunk_size

	output_dir          = args.output_dir
	show_dataset        = args.show_dataset
	visualize_dataset   = args.visualize_dataset
	visualize_model     = args.visualize_model
	visualize_colormap  = args.visualize_colormap
	classes_per_set     = args.classes_per_set
	samples_per_class   = args.samples_per_class
	visualize_square    = args.visualize_square
	visualize_hexagonal = args.visualize_hexagonal
	show_results        = args.show_results

	optimizer           = args.optimizer
	metrics             = args.metrics
	batch_size          = args.batch_size
	epochs              = args.epochs
	loss_string         = args.loss
	runs                = args.runs
	validation_split    = args.validation_split

	cnn_kernel_size     = args.cnn_kernel_size
	cnn_pool_size       = args.cnn_pool_size
	resnet_stacks       = args.resnet_stacks
	resnet_n            = args.resnet_n
	resnet_filter_size  = args.resnet_filter_size

	verbosity_level     = args.verbosity_level

	transform_s2h       = args.transform_s2h
	transform_h2s       = args.transform_h2s
	transform_h2h       = args.transform_h2h
	transform_s2s       = args.transform_s2s
	transform_s2h_rad_o = args.transform_s2h_rad_o
	transform_h2s_len   = args.transform_h2s_len
	transform_h2h_rad_o = args.transform_h2h_rad_o
	transform_s2s_res   = args.transform_s2s_res


	############################################################################
	# Initialization
	############################################################################

	if model_string:
		model_is_provided = True

		model_is_custom      = True if 'custom'      in model_string else False
		model_is_standalone  = True if 'standalone'  in model_string else False

		model_is_autoencoder = True if 'autoencoder' in model_string else False
		model_is_cnn         = True if 'CNN'         in model_string else False
		model_is_gan         = True if 'GAN'         in model_string else False
		model_is_resnet      = True if 'ResNet'      in model_string else False

		model_is_from_keras   = True if 'keras'   in model_string else False
		model_is_from_sklearn = True if 'sklearn' in model_string else False
	else:
		model_is_provided = False

	if augmenter_string:
		augmenter_is_provided = True

		augmenter_is_custom = True if 'custom' in augmenter_string else False
	else:
		augmenter_is_provided = False

	if metrics != 'auto':
		metrics_are_provided = True
	else:
		metrics_are_provided = False

	if loss_string != 'auto':
		loss_is_provided = True

		subpixel_loss_identifiers = ('s2s', 's2h')
		loss_is_subpixel_loss     = True if any(identifier in loss_string for identifier in subpixel_loss_identifiers) else False

		loss_is_from_keras = loss_string.startswith('keras_')
	else:
		loss_is_provided = False

	if dataset:
		dataset = dataset.rstrip('/')

		dataset_is_provided = True

		if create_dataset: create_dataset = ast.literal_eval(create_dataset)
	else:
		dataset_is_provided = False

	if disable_output:
		output_dir = None
	elif not output_dir:
		disable_output = True

	disable_training    |= epochs < 1
	enable_training      = not disable_training
	enable_testing       = not disable_testing
	enable_output        = not disable_output
	disable_tensorboard  = not enable_tensorboard
	enable_rand          = not disable_rand

	train_classes      = []
	train_classes_orig = []
	train_data         = []
	train_filenames    = []
	train_labels       = []
	train_labels_orig  = []
	test_classes       = []
	test_classes_orig  = []
	test_data          = []
	test_filenames     = []
	test_labels        = []
	test_labels_orig   = []

	classification_reports = []

	if enable_tensorboard and enable_output:
		fit_callbacks = [tf.keras.callbacks.TensorBoard(log_dir=os.path.normpath(output_dir), histogram_freq=1)]
	else:
		fit_callbacks = None


	############################################################################
	# No dataset provided - returning
	############################################################################

	if not dataset_is_provided:
		print_newline()
		Hexnet_print('No dataset provided - returning')

		return 0


	############################################################################
	# Create classification dataset
	############################################################################

	if create_dataset:
		print_newline()

		dataset = datasets.create_dataset(
			dataset               = dataset,
			split_ratios          = create_dataset,
			output_dir            = output_dir,
			randomized_assignment = enable_rand,
			seed                  = rand_seed,
			verbosity_level       = verbosity_level)


	############################################################################
	# Transform the dataset
	############################################################################

	if any(operation != False for operation in (transform_s2h, transform_h2s, transform_h2h, transform_s2s)):
		print_newline()
		Hexnet_init()
		print_newline()

		if transform_s2h != False:
			if transform_s2h is None:
				transform_s2h = os.path.join(output_dir, f'{os.path.basename(dataset)}_s2h_rad_o_{transform_s2h_rad_o:.3f}')

			datasets.transform_dataset(
				dataset         = dataset,
				output_dir      = transform_s2h,
				mode            = 's2h',
				rad_o           = transform_s2h_rad_o,
				method          = 0,
				verbosity_level = verbosity_level)

		if transform_h2s != False:
			if transform_h2s is None:
				transform_h2s = os.path.join(output_dir, f'{os.path.basename(dataset)}_h2s_len_{transform_h2s_len:.3f}')

			datasets.transform_dataset(
				dataset         = dataset,
				output_dir      = transform_h2s,
				mode            = 'h2s',
				len             = transform_h2s_len,
				method          = 0,
				verbosity_level = verbosity_level)

		if transform_h2h != False:
			if transform_h2h is None:
				transform_h2h = os.path.join(output_dir, f'{os.path.basename(dataset)}_h2h_rad_o_{transform_h2h_rad_o:.3f}')

			datasets.transform_dataset(
				dataset         = dataset,
				output_dir      = transform_h2h,
				mode            = 'h2h',
				rad_o           = transform_h2h_rad_o,
				method          = 0,
				verbosity_level = verbosity_level)

		if transform_s2s != False:
			if transform_s2s is None:
				transform_s2s_res_str = "x".join(str(d) for d in transform_s2s_res)
				transform_s2s         = os.path.join(output_dir, f'{os.path.basename(dataset)}_s2s_res_{transform_s2s_res_str}')

			datasets.transform_dataset(
				dataset         = dataset,
				output_dir      = transform_s2s,
				mode            = 's2s',
				res             = transform_s2s_res,
				method          = 0,
				verbosity_level = verbosity_level)


	############################################################################
	# Visualize the dataset
	############################################################################

	if visualize_dataset and os.path.isfile(dataset) and dataset.lower().endswith('.csv'):
		print_newline()

		datasets.visualize_dataset(
			dataset,
			output_dir          = output_dir,
			visualize_colormap  = visualize_colormap,
			classes_per_set     = classes_per_set,
			samples_per_class   = samples_per_class,
			visualize_square    = visualize_square,
			visualize_hexagonal = visualize_hexagonal,
			create_h5           = create_h5,
			verbosity_level     = verbosity_level)


	############################################################################
	# Load the dataset
	############################################################################

	print_newline()

	dataset_data = datasets.load_dataset(dataset, create_h5, verbosity_level)

	loaded_dataset = dataset[2]

	if not loaded_dataset:
		return 0

	train_classes_orig = dataset_data[0][0]
	train_data         = dataset_data[0][1]
	train_filenames    = dataset_data[0][2]
	train_labels_orig  = dataset_data[0][3]
	test_classes_orig  = dataset_data[1][0]
	test_data          = dataset_data[1][1]
	test_filenames     = dataset_data[1][2]
	test_labels_orig   = dataset_data[1][3]

	print_newline()

	datasets.create_dataset_overview(train_classes_orig, train_labels_orig, test_labels_orig, dataset, output_dir)

	if type(train_data) is not np.ndarray:
		disable_training = True
		enable_training  = not disable_training

	if type(test_data) is not np.ndarray:
		disable_testing = True
		enable_testing  = not disable_testing


	############################################################################
	# Prepare the dataset
	############################################################################

	print_newline()
	Hexnet_print('Dataset preparation')

	class_labels_are_digits = True

	if enable_training:
		for class_label in train_classes_orig:
			if not class_label.isdigit():
				class_labels_are_digits = False
				break
	elif enable_testing:
		for class_label in test_classes_orig:
			if not class_label.isdigit():
				class_labels_are_digits = False
				break

	if enable_training:
		if class_labels_are_digits:
			train_labels = np.asarray([int(label) for label in train_labels_orig.flatten()])
		else:
			train_labels = np.asarray([np.where(label == train_classes_orig)[0][0] for label in train_labels_orig.flatten()])

		train_classes     = np.unique(train_labels)
		train_classes_len = len(train_classes)
		train_labels      = np.reshape(train_labels, newshape=train_labels_orig.shape)
		train_labels_len  = len(train_labels)

		if class_labels_are_digits:
			train_classes_min = min(train_classes)

			train_classes -= train_classes_min
			train_labels  -= train_classes_min

		if train_labels.ndim > 1:
			train_labels = array_to_one_hot_array(train_labels, train_classes_len)

	if enable_testing:
		if class_labels_are_digits:
			test_labels = np.asarray([int(label) for label in test_labels_orig.flatten()])
		else:
			test_labels = np.asarray([np.where(label == test_classes_orig)[0][0] for label in test_labels_orig.flatten()])

		test_classes     = np.unique(test_labels)
		test_classes_len = len(test_classes)
		test_labels      = np.reshape(test_labels, newshape=test_labels_orig.shape)
		test_labels_len  = len(test_labels)

		if class_labels_are_digits:
			test_classes_min = min(test_classes)

			test_classes -= test_classes_min
			test_labels  -= test_classes_min

		if test_labels.ndim > 1:
			test_labels = array_to_one_hot_array(test_labels, test_classes_len)


	############################################################################
	# Preprocess the dataset
	############################################################################

	if any(operation for operation in (resize_dataset, crop_dataset, pad_dataset)):
		Hexnet_print('Dataset preprocessing')

		if resize_dataset:
			(train_data, test_data) = datasets.resize_dataset(dataset_s = (train_data, test_data), resize_string = resize_dataset)

		if crop_dataset:
			(train_data, test_data) = datasets.crop_dataset(dataset_s = (train_data, test_data), crop_string = crop_dataset)

		if pad_dataset:
			(train_data, test_data) = datasets.pad_dataset(dataset_s = (train_data, test_data), pad_string = pad_dataset)


	############################################################################
	# Augment the dataset
	############################################################################

	if augment_dataset:
		Hexnet_print('Dataset augmentation')

		if augmentation_size != 1:
			if 'train' in augment_dataset:
				train_data, train_filenames, train_labels, train_labels_orig = \
					augmenters.augment_size(train_data, train_filenames, train_labels, train_labels_orig, augmentation_size)

			if 'test' in augment_dataset:
				test_data, test_filenames, test_labels, test_labels_orig = \
					augmenters.augment_size(test_data, test_filenames, test_labels, test_labels_orig, augmentation_size)

		if augmenter_is_provided:
			augmenter = vars(augmenters)[f'augmenter_{augmenter_string}']

			if not augmenter_is_custom:
				augmenter = augmenter(augmentation_level)
			else:
				augmenter = augmenter()

			if 'train' in augment_dataset:
				train_data = augmenter(images=train_data)

			if 'test' in augment_dataset:
				test_data = augmenter(images=test_data)


	############################################################################
	# Show the dataset
	############################################################################

	if show_dataset:
		datasets.show_dataset(
			train_classes_orig,
			train_data,
			train_labels_orig,
			test_classes_orig,
			test_data,
			test_labels_orig,
			max_images_per_class   =  1,
			max_classes_to_display = 10)


	############################################################################
	# Visualize the dataset
	############################################################################

	if visualize_dataset:
		print_newline()

		datasets.visualize_dataset(
			dataset,
			train_classes_orig,
			train_data,
			train_filenames,
			train_labels_orig,
			test_classes_orig,
			test_data,
			test_filenames,
			test_labels_orig,
			output_dir,
			visualize_colormap,
			classes_per_set,
			samples_per_class,
			visualize_square,
			visualize_hexagonal,
			create_h5,
			verbosity_level)


	############################################################################
	# No model provided - returning
	############################################################################

	if not model_is_provided:
		print_newline()
		Hexnet_print('No model provided - returning')

		return 0


	############################################################################
	# Standardize / normalize the dataset
	############################################################################

	if not model_is_gan:
		if visualize_dataset: print_newline()
		Hexnet_print('Dataset standardization')

		std_eps = np.finfo(np.float32).eps

		if enable_training:
			if train_data.ndim > 3:
				mean_axis = (1, 2)
			else:
				mean_axis = 1

			train_data = train_data.astype(np.float32, copy=False)

			for chunk_start in tqdm(range(0, train_data.shape[0], chunk_size)):
				chunk_end = min(chunk_start + chunk_size, train_data.shape[0])

				train_data_mean = np.mean(train_data[chunk_start:chunk_end], axis=mean_axis, keepdims=True)

				train_data_std = np.sqrt(((train_data[chunk_start:chunk_end] - train_data_mean)**2).mean(axis=mean_axis, keepdims=True))
				train_data_std[train_data_std == 0] = std_eps

				train_data[chunk_start:chunk_end] = (train_data[chunk_start:chunk_end] - train_data_mean) / train_data_std

		if enable_testing:
			if test_data.ndim > 3:
				mean_axis = (1, 2)
			else:
				mean_axis = 1

			test_data_orig = test_data
			test_data      = test_data.astype(np.float32)

			for chunk_start in tqdm(range(0, test_data.shape[0], chunk_size)):
				chunk_end = min(chunk_start + chunk_size, test_data.shape[0])

				test_data_mean = np.mean(test_data[chunk_start:chunk_end], axis=mean_axis, keepdims=True)

				test_data_std = np.sqrt(((test_data[chunk_start:chunk_end] - test_data_mean)**2).mean(axis=mean_axis, keepdims=True))
				test_data_std[test_data_std == 0] = std_eps

				test_data[chunk_start:chunk_end] = (test_data[chunk_start:chunk_end] - test_data_mean) / test_data_std
	else:
		if visualize_dataset: print_newline()
		Hexnet_print('Dataset normalization')

		data_min             = min(train_data.min(), test_data.min())
		data_max             = max(train_data.max(), test_data.max())
		normalization_factor = data_max - data_min

		if enable_training:
			for chunk_start in tqdm(range(0, train_data.shape[0], chunk_size)):
				train_data[chunk_start:chunk_end] = (train_data[chunk_start:chunk_end] - data_min) / normalization_factor

		if enable_testing:
			for chunk_start in tqdm(range(0, test_data.shape[0], chunk_size)):
				test_data[chunk_start:chunk_end] = (test_data[chunk_start:chunk_end] - data_min) / normalization_factor


	############################################################################
	# Shuffle the dataset
	############################################################################

	if enable_training:
		Hexnet_print('Dataset shuffling')

		(train_data, train_labels) = sklearn.utils.shuffle(train_data, train_labels)


	############################################################################
	# Start a new training and test run
	############################################################################

	dataset     = os.path.basename(dataset)
	tests_title = f'{model_string}__{dataset}'

	if not model_is_from_sklearn:
		tests_title = f'{tests_title}__epochs{epochs}-bs{batch_size}'

	print_newline()
	print_newline()

	for run in range(1, runs + 1):

		########################################################################
		# Current run information
		########################################################################

		run_string = f'run={run}/{runs}'
		timestamp  = datetime.now().strftime('%Y%m%d-%H%M%S')
		run_title  = f'{tests_title}__{timestamp}_run{run}'


		########################################################################
		# Initialize the model
		########################################################################

		Hexnet_print(f'({run_string}) Model initialization')

		if enable_training:
			input_shape = train_data.shape[1:4]
			classes     = train_classes_len
		elif enable_testing:
			input_shape = test_data.shape[1:4]
			classes     = test_classes_len

		if not load_model:
			model = vars(models)[f'model_{model_string}']

			if model_is_custom or model_is_standalone:
				model = model(input_shape, classes)
			elif model_is_autoencoder:
				model = model(input_shape)
			elif model_is_cnn:
				model = model(input_shape, classes, cnn_kernel_size, cnn_pool_size)
			elif model_is_resnet and not model_is_from_keras:
				model = model(input_shape, classes, resnet_stacks, resnet_n, resnet_filter_size)
			elif model_is_from_sklearn:
				model = model()
			else:
				model = model(input_shape, classes)
		elif not (model_is_standalone or model_is_from_sklearn):
			model = tf.keras.models.load_model(load_model)

		if load_weights and not (model_is_standalone or model_is_from_sklearn):
			model.load_weights(load_weights)


		########################################################################
		# Initialize loss and metrics
		########################################################################

		if not (model_is_standalone or model_is_from_sklearn):
			Hexnet_print(f'({run_string}) Loss and metrics initialization')

			if not metrics_are_provided:
				if not model_is_autoencoder:
					metrics = ['accuracy']
				else:
					metrics = []

			if not loss_is_provided:
				if not model_is_autoencoder:
					loss = 'SparseCategoricalCrossentropy'
				else:
					loss = 'MeanSquaredError'

				loss = tf.losses.get(loss)
			else:
				if not loss_is_subpixel_loss:
					if not loss_is_from_keras:
						loss = vars(losses)[f'loss_{loss_string}']()
					else:
						loss = vars(tf.keras.losses)[loss_string[len('keras_'):]]()
				else:
					output_shape = test_data.shape[1:4]
					loss         = vars(losses)[f'loss_{loss_string}'](input_shape, output_shape)


		########################################################################
		# Compile the model
		########################################################################

		if not model_is_from_sklearn:
			Hexnet_print(f'({run_string}) Model compilation')

			if not model_is_standalone:
				model.compile(optimizer, loss, metrics)
			else:
				model.compile()


		########################################################################
		# Model summary
		########################################################################

		print_newline()
		Hexnet_print(f'({run_string}) Model summary')

		if not model_is_from_sklearn:
			model.summary()
		else:
			Hexnet_print(model.get_params())


		########################################################################
		# Train the model
		########################################################################

		if enable_training:
			print_newline()
			Hexnet_print(f'({run_string}) Model training')


			# Model training time callback: training time per epoch and training time per sample

			class TimeHistory(tf.keras.callbacks.Callback):
				def on_train_begin(self, logs={}):
					self.times = []

				def on_epoch_begin(self, batch, logs={}):
					self.epoch_time_start = time()

				def on_epoch_end(self, batch, logs={}):
					self.times.append(time() - self.epoch_time_start)

			time_callback = TimeHistory()

			if fit_callbacks:
				fit_callbacks.append(time_callback)
			else:
				fit_callbacks = [time_callback]


			if model_is_standalone:
				model.fit(train_data, train_labels, batch_size, epochs, visualize_hexagonal, output_dir, run_title)
			elif model_is_autoencoder:
				history = model.fit(train_data, train_data, batch_size, epochs, validation_split=validation_split)
			elif model_is_from_sklearn:
				model.fit(np.reshape(train_data, newshape = (train_data.shape[0], -1)), train_labels)
			else:
				history = model.fit(train_data, train_labels, batch_size, epochs, validation_split=validation_split, callbacks=fit_callbacks)

				# Training time per epoch and training time per sample
				training_time_per_epoch  = [float(format(time, '.8f')) for time in time_callback.times]
				training_time_per_sample = [float(format(1000 * time / ((1 - validation_split) * train_labels_len), '.8f')) for time in time_callback.times]
				Hexnet_print(f'({run_string}) training time per epoch [s]: {training_time_per_epoch}, training time per sample [ms]: {training_time_per_sample}')


		########################################################################
		# Visualize filters, feature maps, activations, and training results
		########################################################################

		if not (model_is_standalone or model_is_from_sklearn):
			if visualize_model and enable_output:
				print_newline()
				Hexnet_print(f'({run_string}) Visualization')

				output_dir_visualizations = os.path.join(output_dir, f'{run_title}_visualizations')

				visualization.visualize_model(
					model,
					test_classes,
					test_data,
					test_data_orig,
					test_filenames,
					test_labels,
					visualize_colormap,
					visualize_hexagonal,
					output_dir           = output_dir_visualizations,
					max_images_per_class = 10,
					verbosity_level      = verbosity_level)

			if enable_training:
				print_newline()
				Hexnet_print(f'({run_string}) History')
				Hexnet_print(f'({run_string}) history.history.keys()={history.history.keys()}')

				if enable_output or show_results:
					visualization.visualize_training_results(history, run_title, output_dir, show_results)


		########################################################################
		# Evaluate the model
		########################################################################

		if enable_testing and not model_is_from_sklearn:
			print_newline()
			Hexnet_print(f'({run_string}) Model evaluation')

			if model_is_standalone:
				model.evaluate(test_data, test_labels, batch_size, epochs=10, visualize_hexagonal=visualize_hexagonal, output_dir=output_dir, run_title=run_title)
			elif model_is_autoencoder:
				test_loss_metrics = model.evaluate(test_data, test_data)
			else:
				_testing_time     = time()
				test_loss_metrics = model.evaluate(test_data, test_labels)
				_testing_time     = time() - _testing_time

				# Testing time and testing time per sample
				testing_time            = float(format(_testing_time, '.8f'))
				testing_time_per_sample = float(format(1000 * _testing_time / test_labels_len, '.8f'))
				Hexnet_print(f'({run_string}) testing time [s]: {testing_time}, testing time per sample [ms]: {testing_time_per_sample}')

			if not model_is_standalone:
				Hexnet_print(f'({run_string}) test_loss_metrics={test_loss_metrics}')


		########################################################################
		# Save test results
		########################################################################

		if enable_testing and enable_output and not model_is_standalone:
			print_newline()
			Hexnet_print(f'({run_string}) Saving test results')

			if not model_is_from_sklearn:
				predictions = model.predict(test_data)
			else:
				predictions = model.predict_proba(np.reshape(test_data, newshape = (test_data.shape[0], -1)))

			if not model_is_autoencoder:
				classification_report = visualization.visualize_test_results(
					predictions,
					test_classes,
					test_classes_orig,
					test_filenames,
					test_labels,
					test_labels_orig,
					run_title,
					output_dir)

				print_newline()
				Hexnet_print(f'({run_string}) Classification report')
				pprint(classification_report)

				classification_reports.append(classification_report)
			else:
				loss_newshape = (test_data.shape[0], -1)
				test_losses   = loss(np.reshape(test_data, newshape=loss_newshape), np.reshape(predictions, newshape=loss_newshape))

				output_dir_predictions = os.path.join(output_dir, f'{run_title}_predictions')
				os.makedirs(output_dir_predictions, exist_ok=True)

				with open(f'{output_dir_predictions}.csv', 'w') as predictions_file:
					print('label_orig,filename,label,loss', file=predictions_file)

					for label_orig, filename, label, loss in zip(test_labels_orig, test_filenames, test_labels, test_losses):
						loss = float(format(loss, '.8f'))
						print(f'{label_orig},{filename},{label},{loss}', file=predictions_file)

				for image_counter, (image, label) in enumerate(zip(tqdm(predictions), test_labels)):
					image_filename = f'label{label}_image{image_counter}.png'

					if not visualize_hexagonal:
						imsave(os.path.join(output_dir_predictions, image_filename), image)
					else:
						visualization.visualize_hexarray(image, os.path.join(output_dir_predictions, image_filename))


		########################################################################
		# Save the model
		########################################################################

		if (save_model or save_weights) and enable_output and not (model_is_standalone or model_is_from_sklearn):
			print_newline()
			Hexnet_print(f'({run_string}) Saving the model')

			if save_model:
				model.save(os.path.join(output_dir, f'{run_title}_model.h5'))

			if save_weights:
				model.save_weights(os.path.join(output_dir, f'{run_title}_weights.h5'))


		if run < runs:
			print_newline()
			print_newline()


	############################################################################
	# Save global test results
	############################################################################

	if enable_testing and enable_output and runs > 1 and not (model_is_standalone or model_is_autoencoder):
		timestamp   = datetime.now().strftime('%Y%m%d-%H%M%S')
		tests_title = f'{tests_title}__{timestamp}'

		visualization.visualize_global_test_results(classification_reports, tests_title, output_dir)


	return 0


################################################################################
# parse_args
################################################################################

def parse_args(args=None, namespace=None):
	parser = argparse.ArgumentParser(description='Hexnet: The Hexagonal Machine Learning Module')


	loss_choices_to_ignore = ['Loss', 'Reduction']

	model_choices     = [model[0][len('model_'):] for model in inspect.getmembers(models, inspect.isfunction) if model[0].startswith('model_')]
	augmenter_choices = [augmenter[0][len('augmenter_'):] for augmenter in inspect.getmembers(augmenters) if augmenter[0].startswith('augmenter_')]

	loss_choices  = ['auto']
	loss_choices += [loss[0][len('loss_'):] for loss in inspect.getmembers(losses, inspect.isclass) if loss[0].startswith('loss_')]
	loss_choices += [f'keras_{loss[0]}' for loss in inspect.getmembers(tf.keras.losses, inspect.isclass) if loss[0] not in loss_choices_to_ignore]

	augment_dataset_choices = ['train', 'test']


	parser.add_argument('--disable-training',                               action  = 'store_true',        help = 'disable training')
	parser.add_argument('--disable-testing',                                action  = 'store_true',        help = 'disable testing')
	parser.add_argument('--disable-output',                                 action  = 'store_true',        help = 'disable training and test results\' output')
	parser.add_argument('--enable-tensorboard',                             action  = 'store_true',        help = 'enable TensorBoard')

	parser.add_argument(
		'--model',
		nargs   = '?',
		default = model,
		choices = model_choices,
		help    = 'model for training and testing: choices are generated from models/models.py (providing no argument disables training and testing)')

	parser.add_argument('--load-model',                                     default = load_model,          help = 'load model from HDF5')
	parser.add_argument('--load-weights',                                   default = load_weights,        help = 'load model weights from HDF5')
	parser.add_argument('--save-model',                                     action  = 'store_true',        help = 'save model as HDF5')
	parser.add_argument('--save-weights',                                   action  = 'store_true',        help = 'save model weights as HDF5')

	parser.add_argument('--dataset',                           nargs = '?', default = dataset,             help = 'load dataset from HDF5 or directory')
	parser.add_argument('--create-dataset',                                 default = create_dataset,      help = 'create classification dataset from dataset using "{set:fraction}" (e.g., "{\'train\':0.9,\'test\':0.1}")')
	parser.add_argument('--disable-rand',                                   action  = 'store_true',        help = 'classification dataset creation: disable randomized file dataset set assignment')
	parser.add_argument('--rand-seed',           type = int,                default = rand_seed,           help = 'classification dataset creation: seed for randomized file dataset set assignment')
	parser.add_argument('--create-h5',                                      action  = 'store_true',        help = 'save dataset as HDF5')
	parser.add_argument('--resize-dataset',                                 default = resize_dataset,      help = 'resize dataset using "HxW" (e.g., 32x32)')
	parser.add_argument('--crop-dataset',                                   default = crop_dataset,        help = 'crop dataset using "HxW" with offset "+Y+X" (e.g., 32x32+2+2, 32x32, or +2+2)')
	parser.add_argument('--pad-dataset',                                    default = pad_dataset,         help = 'pad dataset using "T,B,L,R" (e.g., 2,2,2,2)')

	parser.add_argument(
		'--augment-dataset',
		nargs   = '+',
		default = augment_dataset,
		choices = augment_dataset_choices,
		help    = 'dataset set(s) for augmentation')

	parser.add_argument(
		'--augmenter',
		nargs   = '?',
		default = augmenter,
		choices = augmenter_choices,
		help    = 'augmenter: choices are generated from misc/augmenters.py')

	parser.add_argument('--augmentation-level',  type = int,                default = augmentation_level,  help = 'augmentation level')
	parser.add_argument('--augmentation-size',   type = float,              default = augmentation_size,   help = 'augmentation dataset set(s) size factor')

	parser.add_argument('--chunk-size',          type = int,                default = chunk_size,          help = 'preprocessing chunk size')

	parser.add_argument('--output-dir',                        nargs = '?', default = output_dir,          help = 'training and test results\' output directory (providing no argument disables the output)')
	parser.add_argument('--show-dataset',                                   action  = 'store_true',        help = 'show the dataset')
	parser.add_argument('--visualize-dataset',                              action  = 'store_true',        help = 'visualize the dataset after preprocessing and augmentation')
	parser.add_argument('--visualize-model',                                action  = 'store_true',        help = 'visualize the model\'s filters, feature maps, and activations after training')
	parser.add_argument('--visualize-colormap',                             default = visualize_colormap,  help = 'visualization color map (e.g., viridis)')
	parser.add_argument('--classes-per-set',                                default = classes_per_set,     help = 'classes per set to visualize')
	parser.add_argument('--samples-per-class',                              default = samples_per_class,   help = 'samples per class to visualize')
	parser.add_argument('--visualize-square',                               action  = 'store_true',        help = 'visualize as square arrays')
	parser.add_argument('--visualize-hexagonal',                            action  = 'store_true',        help = 'visualize as hexagonal arrays')
	parser.add_argument('--show-results',                                   action  = 'store_true',        help = 'show the test results')

	parser.add_argument('--optimizer',                                      default = optimizer,           help = 'optimizer for training')
	parser.add_argument('--metrics',                           nargs = '*', default = metrics,             help = 'metrics for training and testing')
	parser.add_argument('--batch-size',          type = int,                default = batch_size,          help = 'batch size for training and testing')
	parser.add_argument('--epochs',              type = int,                default = epochs,              help = 'epochs for training')

	parser.add_argument(
		'--loss',
		default = loss,
		choices = loss_choices,
		help    = 'custom loss for training and testing: choices are generated from misc/losses.py')

	parser.add_argument('--runs',                type = int,                default = runs,                help = 'number of training and test runs')
	parser.add_argument('--validation-split',    type = float,              default = validation_split,    help = 'fraction of the training data to be used as validation data')

	parser.add_argument('--cnn-kernel-size',     type = int,   nargs = '+', default = cnn_kernel_size,     help = 'CNN models\' kernel size')
	parser.add_argument('--cnn-pool-size',       type = int,   nargs = '+', default = cnn_pool_size,       help = 'CNN models\' pooling size')
	parser.add_argument('--resnet-stacks',       type = int,                default = resnet_stacks,       help = 'ResNet models\' number of stacks')
	parser.add_argument('--resnet-n',            type = int,                default = resnet_n,            help = 'ResNet models\' number of residual blocks\' n')
	parser.add_argument('--resnet-filter-size',  type = float,              default = resnet_filter_size,  help = 'ResNet models\' filter size factor (convolutional layers)')

	parser.add_argument('--verbosity-level',     type = int,                default = verbosity_level,     help = 'verbosity level (default is 2, maximum is 3)')

	parser.add_argument('--transform-s2h',                     nargs = '?', default = transform_s2h,       help = 'enable square to hexagonal image transformation')
	parser.add_argument('--transform-h2s',                     nargs = '?', default = transform_h2s,       help = 'enable hexagonal to square image transformation')
	parser.add_argument('--transform-h2h',                     nargs = '?', default = transform_h2h,       help = 'enable hexagonal to hexagonal image transformation')
	parser.add_argument('--transform-s2s',                     nargs = '?', default = transform_s2s,       help = 'enable square to square image transformation')
	parser.add_argument('--transform-s2h-rad-o', type = float,              default = transform_s2h_rad_o, help = 'square to hexagonal image transformation hexagonal pixels\' outer radius')
	parser.add_argument('--transform-h2s-len',   type = float,              default = transform_h2s_len,   help = 'hexagonal to square image transformation square pixels\' side length')
	parser.add_argument('--transform-h2h-rad-o', type = float,              default = transform_h2h_rad_o, help = 'hexagonal to hexagonal image transformation hexagonal pixels\' outer radius')
	parser.add_argument('--transform-s2s-res',   type = int,   nargs = '+', default = transform_s2s_res,   help = 'square to square image transformation output resolution')


	return parser.parse_args(args, namespace)


################################################################################
# main
################################################################################

if __name__ == '__main__':
	args = parse_args()

	Hexnet_print(f'args={args}')

	status = run(args)

	sys.exit(status)

