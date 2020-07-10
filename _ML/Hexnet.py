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
load_model          = None
load_weights        = None

dataset             = 'datasets/MNIST/MNIST.h5'
create_dataset      = None
resize_dataset      = None
crop_dataset        = None
pad_dataset         = None
augment_dataset     = None
augmenter           = 'simple'
augmentation_level  = 1

output_dir          = 'tests/tmp'

batch_size          = 32
epochs              =  1
loss                = None
runs                =  1
validation_split    =  0.0

cnn_kernel_size     = (3, 3)
cnn_pool_size       = (3, 3)

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
from tqdm              import tqdm

import datasets.datasets  as datasets
import misc.augmenters    as augmenters
import misc.losses        as losses
import misc.visualization as visualization
import models.models      as models

from core.Hexnet import Hexnet_init
from misc.misc   import Hexnet_print, print_newline


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

	model_string        = args.model
	load_model          = args.load_model
	load_weights        = args.load_weights
	save_model          = args.save_model
	save_weights        = args.save_weights

	dataset             = args.dataset
	create_dataset      = args.create_dataset
	create_h5           = args.create_h5
	resize_dataset      = args.resize_dataset
	crop_dataset        = args.crop_dataset
	pad_dataset         = args.pad_dataset
	augment_dataset     = args.augment_dataset
	augmenter_string    = args.augmenter
	augmentation_level  = args.augmentation_level

	output_dir          = args.output_dir
	show_dataset        = args.show_dataset
	visualize_dataset   = args.visualize_dataset
	visualize_model     = args.visualize_model
	visualize_hexagonal = args.visualize_hexagonal
	show_results        = args.show_results

	batch_size          = args.batch_size
	epochs              = args.epochs
	loss_string         = args.loss
	runs                = args.runs
	validation_split    = args.validation_split

	cnn_kernel_size     = args.cnn_kernel_size
	cnn_pool_size       = args.cnn_pool_size

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

	if model_string is not None:
		model_is_provided = True

		model_is_custom      = True if 'custom'      in model_string else False
		model_is_standalone  = True if 'standalone'  in model_string else False

		model_is_autoencoder = True if 'autoencoder' in model_string else False
		model_is_CNN         = True if 'CNN'         in model_string else False
		model_is_GAN         = True if 'GAN'         in model_string else False

		model_is_from_sklearn = True if 'sklearn' in model_string else False
	else:
		model_is_provided = False

	if augmenter_string is not None:
		augmenter_is_custom = True if 'custom' in augmenter_string else False

	if loss_string is not None:
		loss_is_provided = True

		subpixel_loss_identifiers = ('s2s', 's2h')
		loss_is_subpixel_loss = True if any(identifier in loss_string for identifier in subpixel_loss_identifiers) else False
	else:
		loss_is_provided = False

	if dataset is not None:
		dataset = dataset.rstrip('/')
		dataset_is_provided = True

		if create_dataset is not None:
			create_dataset = ast.literal_eval(create_dataset)

		visualize_hexagonal_identifiers = ('hex', 's2h', 'h2h')
		if any(identifier in dataset for identifier in visualize_hexagonal_identifiers): visualize_hexagonal = True
	else:
		dataset_is_provided = False

	if disable_output:
		output_dir = None
	elif output_dir is None:
		disable_output = True

	disable_training |= epochs < 1
	enable_training   = not disable_training
	enable_testing    = not disable_testing
	enable_output     = not disable_output

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

	if create_dataset is not None:
		print_newline()

		datasets.create_dataset(dataset, split_ratios=create_dataset, verbosity_level=verbosity_level)

		dataset = f'{dataset}_classification_dataset'


	############################################################################
	# Transform the dataset
	############################################################################

	if any(operation != False for operation in (transform_s2h, transform_h2s, transform_h2h, transform_s2s)):
		print_newline()
		Hexnet_init()
		print_newline()

		if transform_s2h != False:
			if transform_s2h is None:
				transform_s2h = f'{dataset}_s2h'

			datasets.transform_dataset(
				dataset         = dataset,
				output_dir      = transform_s2h,
				mode            = 's2h',
				rad_o           = transform_s2h_rad_o,
				method          = 0,
				verbosity_level = verbosity_level)

		if transform_h2s != False:
			if transform_h2s is None:
				transform_h2s = f'{dataset}_h2s'

			datasets.transform_dataset(
				dataset         = dataset,
				output_dir      = transform_h2s,
				mode            = 'h2s',
				len             = transform_h2s_len,
				method          = 0,
				verbosity_level = verbosity_level)

		if transform_h2h != False:
			if transform_h2h is None:
				transform_h2h = f'{dataset}_h2h'

			datasets.transform_dataset(
				dataset         = dataset,
				output_dir      = transform_h2h,
				mode            = 'h2h',
				rad_o           = transform_h2h_rad_o,
				method          = 0,
				verbosity_level = verbosity_level)

		if transform_s2s != False:
			if transform_s2s is None:
				transform_s2s = f'{dataset}_s2s'

			datasets.transform_dataset(
				dataset         = dataset,
				output_dir      = transform_s2s,
				mode            = 's2s',
				res             = transform_s2s_res,
				method          = 0,
				verbosity_level = verbosity_level)


	############################################################################
	# Load the dataset
	############################################################################

	print_newline()

	((train_classes_orig, train_data, train_filenames, train_labels_orig),
	 (test_classes_orig,  test_data,  test_filenames,  test_labels_orig)) = \
		datasets.load_dataset(dataset, create_h5, verbosity_level)


	############################################################################
	# Prepare the dataset
	############################################################################

	print_newline()
	Hexnet_print('Dataset preparation')

	class_labels_are_digits = True

	for class_label in train_classes_orig:
		if not class_label.isdigit():
			class_labels_are_digits = False
			break

	if class_labels_are_digits:
		train_labels = np.asarray([int(label) for label in train_labels_orig])
		test_labels  = np.asarray([int(label) for label in test_labels_orig])
	else:
		train_labels = np.asarray([np.where(label == train_classes_orig)[0][0] for label in train_labels_orig])
		test_labels  = np.asarray([np.where(label == test_classes_orig)[0][0]  for label in test_labels_orig])

	train_classes = list(set(train_labels))
	test_classes  = list(set(test_labels))

	if class_labels_are_digits:
		classes_min = min(train_classes)

		train_labels  -= classes_min
		test_labels   -= classes_min
		train_classes -= classes_min
		test_classes  -= classes_min


	############################################################################
	# Preprocess the dataset
	############################################################################

	if any(operation is not None for operation in (resize_dataset, crop_dataset, pad_dataset)):
		Hexnet_print('Dataset preprocessing')

		if resize_dataset is not None:
			(train_data, test_data) = datasets.resize_dataset(dataset_s = (train_data, test_data), resize_string = resize_dataset)

		if crop_dataset is not None:
			(train_data, test_data) = datasets.crop_dataset(dataset_s = (train_data, test_data), crop_string = crop_dataset)

		if pad_dataset is not None:
			(train_data, test_data) = datasets.pad_dataset(dataset_s = (train_data, test_data), pad_string = pad_dataset)


	############################################################################
	# Augment the dataset
	############################################################################

	if augment_dataset is not None:
		Hexnet_print('Dataset augmentation')

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

	if not model_is_GAN:
		if visualize_dataset: print_newline()
		Hexnet_print('Dataset standardization')

		mean_axis = (1, 2)
		std_eps   = np.finfo(np.float32).eps

		train_data_mean = np.mean(train_data, axis=mean_axis, keepdims=True)
		test_data_mean  = np.mean(test_data,  axis=mean_axis, keepdims=True)

		train_data_std = np.sqrt(((train_data - train_data_mean) ** 2).mean(axis=mean_axis, keepdims=True))
		test_data_std  = np.sqrt(((test_data  - test_data_mean)  ** 2).mean(axis=mean_axis, keepdims=True))
		train_data_std[train_data_std == 0] = std_eps
		test_data_std[test_data_std   == 0] = std_eps

		train_data = (train_data - train_data_mean) / train_data_std
		test_data  = (test_data  - test_data_mean)  / test_data_std
	else:
		if visualize_dataset: print_newline()
		Hexnet_print('Dataset normalization')

		normalization_factor = (2**8 - 1) / 2

		train_data = (train_data - normalization_factor) / normalization_factor
		test_data  = (test_data  - normalization_factor) / normalization_factor


	############################################################################
	# Shuffle the dataset
	############################################################################

	Hexnet_print('Dataset shuffling')

	(train_data, train_labels) = sklearn.utils.shuffle(train_data, train_labels)


	############################################################################
	# Start a new training and test run
	############################################################################

	dataset     = os.path.basename(dataset)
	tests_title = f'{model_string}_{dataset}'

	if not model_is_from_sklearn:
		tests_title = f'{tests_title}_epochs{epochs}-bs{batch_size}'

	run_title_base = tests_title

	print_newline()
	print_newline()

	for run in range(1, runs + 1):

		########################################################################
		# Current run information
		########################################################################

		run_string = f'run={run}/{runs}'
		timestamp  = datetime.now().strftime('%Y%m%d-%H%M%S')
		run_title  = f'{run_title_base}_{timestamp}'

		if runs > 1:
			run_title = f'{run_title}_run{run}'


		########################################################################
		# Initialize the model
		########################################################################

		Hexnet_print(f'({run_string}) Model initialization')

		input_shape = train_data.shape[1:4]
		classes     = len(train_classes)

		if load_model is None:
			model = vars(models)[f'model_{model_string}']

			if model_is_custom or model_is_standalone:
				model = model(input_shape, classes)
			elif model_is_autoencoder:
				model = model(input_shape)
			elif model_is_CNN:
				model = model(input_shape, classes, cnn_kernel_size, cnn_pool_size)
			elif model_is_from_sklearn:
				model = model()
			else:
				model = model(input_shape, classes)
		elif not (model_is_standalone or model_is_from_sklearn):
			model = tf.keras.models.load_model(load_model)

		if load_weights is not None and not (model_is_standalone or model_is_from_sklearn):
			model.load_weights(load_weights)


		########################################################################
		# Initialize loss and metrics
		########################################################################

		if not model_is_from_sklearn:
			Hexnet_print(f'({run_string}) Loss and metrics initialization')

			if not loss_is_provided:
				if not model_is_autoencoder:
					loss = 'sparse_categorical_crossentropy'
				else:
					loss = 'mse'

				loss = tf.losses.get(loss)
			else:
				if not loss_is_subpixel_loss:
					loss = vars(losses)[f'loss_{loss_string}']()
				else:
					output_shape = test_data.shape[1:4]
					loss = vars(losses)[f'loss_{loss_string}'](input_shape, output_shape)

			if not model_is_autoencoder:
				metrics = ['accuracy']
			else:
				metrics = None


		########################################################################
		# Compile the model
		########################################################################

		if not model_is_from_sklearn:
			Hexnet_print(f'({run_string}) Model compilation')

			if not model_is_standalone:
				model.compile(optimizer='adam', loss=loss, metrics=metrics)
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

			if model_is_standalone:
				model.fit(train_data, train_labels, batch_size, epochs, visualize_hexagonal, output_dir, run_title)
			elif model_is_autoencoder:
				history = model.fit(train_data, train_data, batch_size, epochs, validation_split=validation_split)
			elif model_is_from_sklearn:
				model.fit(np.reshape(train_data, newshape = (train_data.shape[0], -1)), train_labels)
			else:
				history = model.fit(train_data, train_labels, batch_size, epochs, validation_split=validation_split)


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
					test_labels,
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
				test_loss = model.evaluate(test_data, test_data)
			else:
				test_loss, test_acc = model.evaluate(test_data, test_labels)

			if not model_is_standalone:
				if not model_is_autoencoder:
					Hexnet_print(f'({run_string}) test_acc={test_acc:.8f}, test_loss={test_loss:.8f}')
				else:
					Hexnet_print(f'({run_string}) test_loss={test_loss:.8f}')


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
					test_classes_orig,
					test_filenames,
					test_labels,
					test_labels_orig,
					run_title,
					output_dir)

				classification_reports.append(classification_report)
			else:
				loss_newshape = (test_data.shape[0], -1)
				test_losses = loss(np.reshape(test_data, newshape=loss_newshape), np.reshape(predictions, newshape=loss_newshape))

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
			if model_is_autoencoder: print_newline()
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

	if runs > 1:
		timestamp   = datetime.now().strftime('%Y%m%d-%H%M%S')
		tests_title = f'{tests_title}_{timestamp}'

		visualization.visualize_global_test_results(classification_reports, tests_title, output_dir)


	return 0


################################################################################
# parse_args
################################################################################

def parse_args(args=None, namespace=None):
	parser = argparse.ArgumentParser(description='Hexnet: The Hexagonal Machine Learning Module')


	model_choices     = [model[0][len('model_'):] for model in inspect.getmembers(models, inspect.isfunction) if model[0].startswith('model_')]
	augmenter_choices = [augmenter[0][len('augmenter_'):] for augmenter in inspect.getmembers(augmenters) if augmenter[0].startswith('augmenter_')]
	loss_choices      = [loss[0][len('loss_'):] for loss in inspect.getmembers(losses, inspect.isclass) if loss[0].startswith('loss_')]

	augment_dataset_choices = ['train', 'test']

	parser.add_argument(
		'--model',
		nargs   = '?',
		default = model,
		choices = model_choices,
		help    = 'model for training and testing: choices are generated from models/models.py (providing no argument disables training and testing)')

	parser.add_argument(
		'--augment-dataset',
		nargs   = '+',
		default = augment_dataset,
		choices = augment_dataset_choices,
		help    = 'dataset set(s) for augmentation')

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


	parser.add_argument('--disable-training',                               action  = 'store_true',        help = 'disable training')
	parser.add_argument('--disable-testing',                                action  = 'store_true',        help = 'disable testing')
	parser.add_argument('--disable-output',                                 action  = 'store_true',        help = 'disable training and test results\' output')

	parser.add_argument('--load-model',                                     default = load_model,          help = 'load model from HDF5')
	parser.add_argument('--load-weights',                                   default = load_weights,        help = 'load model weights from HDF5')
	parser.add_argument('--save-model',                                     action  = 'store_true',        help = 'save model as HDF5')
	parser.add_argument('--save-weights',                                   action  = 'store_true',        help = 'save model weights as HDF5')

	parser.add_argument('--dataset',                           nargs = '?', default = dataset,             help = 'load dataset from HDF5 or directory')
	parser.add_argument('--create-dataset',                    nargs = '?', default = create_dataset,      help = 'create classification dataset from dataset using "{set:fraction}" (e.g., {\'train\':0.9,\'test\':0.1})')
	parser.add_argument('--create-h5',                                      action  = 'store_true',        help = 'save dataset as HDF5')
	parser.add_argument('--resize-dataset',                                 default = resize_dataset,      help = 'resize dataset using "HxW" (e.g., 32x32)')
	parser.add_argument('--crop-dataset',                                   default = crop_dataset,        help = 'crop dataset using "HxW" with offset "+Y+X" (e.g., 32x32+2+2, 32x32, or +2+2)')
	parser.add_argument('--pad-dataset',                                    default = pad_dataset,         help = 'pad dataset using "T,B,L,R" (e.g., 2,2,2,2)')
	parser.add_argument('--augmentation-level',  type = int,                default = augmentation_level,  help = 'augmentation level for augmentation')

	parser.add_argument('--output-dir',                        nargs = '?', default = output_dir,          help = 'training and test results\' output directory (providing no argument disables the output)')
	parser.add_argument('--show-dataset',                                   action  = 'store_true',        help = 'show the dataset')
	parser.add_argument('--visualize-dataset',                              action  = 'store_true',        help = 'visualize the dataset after preprocessing and augmentation')
	parser.add_argument('--visualize-model',                                action  = 'store_true',        help = 'visualize the model\'s filters, feature maps, and activations after training')
	parser.add_argument('--visualize-hexagonal',                            action  = 'store_true',        help = 'visualize as hexagonal arrays')
	parser.add_argument('--show-results',                                   action  = 'store_true',        help = 'show the test results')

	parser.add_argument('--batch-size',          type = int,                default = batch_size,          help = 'training batch size')
	parser.add_argument('--epochs',              type = int,                default = epochs,              help = 'training epochs')
	parser.add_argument('--runs',                type = int,                default = runs,                help = 'training runs')
	parser.add_argument('--validation-split',    type = float,              default = validation_split,    help = 'fraction of the training data to be used as validation data')

	parser.add_argument('--cnn-kernel-size',     type = int,   nargs = '+', default = cnn_kernel_size,     help = 'CNN models\' kernel size')
	parser.add_argument('--cnn-pool-size',       type = int,   nargs = '+', default = cnn_pool_size,       help = 'CNN models\' pooling size')

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

