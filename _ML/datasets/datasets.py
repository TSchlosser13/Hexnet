'''****************************************************************************
 * datasets.py: Dataset IO, Transformation (e.g., Padding), and Visualization
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

import csv
import cv2
import h5py
import math
import os
import random
import shutil

import matplotlib.pyplot as plt
import numpy             as np
import pandas            as pd
import tensorflow        as tf

from glob                            import glob
from matplotlib.pyplot               import imsave
from natsort                         import natsorted
from sklearn.feature_extraction.text import TfidfVectorizer
from time                            import time
from tqdm                            import tqdm

from core.Hexnet        import Hexsamp_s2h, Hexsamp_h2s, Hexsamp_h2h, Sqsamp_s2s
from misc.misc          import Hexnet_print, Hexnet_print_warning, normalize_array
from misc.visualization import visualize_array, visualize_hexarray


################################################################################
# Dataset class
################################################################################

class dataset:
	def __init__(self):
		self.input_dir_or_file = None

		self.data = {}

		self.loaded_dataset = False


################################################################################
# Create classification dataset from dataset using "{set:fraction}"
#  (e.g., "{'train':0.9,'test':0.1}") and save to memory
################################################################################

def create_dataset(dataset, split_ratios, output_dir, randomized_assignment=True, seed=6, verbosity_level=2):
	Hexnet_print(f'Creating classification dataset from dataset {dataset}')

	start_time = time()

	split_ratios_len       = len(split_ratios)
	split_ratios_sets      = list(split_ratios.keys())
	split_ratios_fractions = list(split_ratios.values())

	classification_dataset = os.path.join(output_dir, f'{os.path.basename(dataset)}_classification_dataset_seed_{seed}')
	os.makedirs(classification_dataset, exist_ok=True)


	if randomized_assignment:
		random.seed(seed)


	max_files_to_copy         = max([len(glob(os.path.join(set_class, '*'))) for set_class in glob(os.path.join(dataset, '*'))])
	max_files_to_copy_per_set = [math.ceil(fraction * max_files_to_copy) for fraction in split_ratios_fractions]

	for set_class in natsorted(glob(os.path.join(dataset, '*'))):

		# Step 1: (randomized) file dataset set assignment

		current_class = os.path.basename(set_class)

		if verbosity_level >= 1:
			Hexnet_print(f'\t> current_class={current_class}')

		files_to_copy = natsorted(glob(os.path.join(set_class, '*')))

		if not files_to_copy:
			continue

		files_to_copy_len    = len(files_to_copy)
		copied_files         = []
		copied_files_per_set = split_ratios_len * [0]

		if verbosity_level >= 2:
			Hexnet_print(f'\t\t> max_files_to_copy_per_set={max_files_to_copy_per_set} (files_to_copy_len={files_to_copy_len})')

		for current_set in split_ratios_sets:
			os.makedirs(os.path.join(classification_dataset, current_set, current_class), exist_ok=True)

		if randomized_assignment:
			files_to_copy = random.sample(files_to_copy, files_to_copy_len)

		for file_to_copy_index, file_to_copy in enumerate(tqdm(files_to_copy)):
			if not randomized_assignment:
				cumulative_split_ratio_fraction = 0

				for split_ratio_fraction_index, split_ratio_fraction in enumerate(split_ratios_fractions):
					cumulative_split_ratio_fraction += split_ratio_fraction

					if file_to_copy_index < cumulative_split_ratio_fraction * files_to_copy_len:
						set_selector = split_ratio_fraction_index
						break

			while True:
				if randomized_assignment:
					set_selector = random.randint(0, split_ratios_len - 1)

				if copied_files_per_set[set_selector] < max_files_to_copy_per_set[set_selector]:
					copy_file_to = os.path.join(classification_dataset, split_ratios_sets[set_selector], current_class, os.path.basename(file_to_copy))

					shutil.copyfile(file_to_copy, copy_file_to)

					copied_files.append(copy_file_to)
					copied_files_per_set[set_selector] += 1

					break

		copied_files_len = len(copied_files)

		if verbosity_level >= 2:
			Hexnet_print(f'\t\t> copied_files_per_set={copied_files_per_set} (copied_files_len={copied_files_len})')


		# Step 2: randomized file dataset set balancing via file duplication

		for current_set_index, current_set in enumerate(split_ratios_sets):
			while copied_files_per_set[current_set_index] < max_files_to_copy_per_set[current_set_index]:
				file_selector = round(random.randint(0, copied_files_len - 1))

				file_to_copy = copied_files[file_selector]
				copy_file_to = os.path.basename(file_to_copy).split('.')
				copy_file_to = '.'.join(copy_file_to[:-1]) + '_' + str(copied_files_per_set[current_set_index]).zfill(8) + '.' + copy_file_to[-1]
				copy_file_to = os.path.join(classification_dataset, current_set, current_class, copy_file_to)

				shutil.copyfile(file_to_copy, copy_file_to)

				copied_files_per_set[current_set_index] += 1

		if verbosity_level >= 2:
			copied_files_len = sum(copied_files_per_set)
			Hexnet_print(f'\t\t> copied_files_per_set={copied_files_per_set} (copied_files_len={copied_files_len}) after balancing')


	time_diff = time() - start_time

	Hexnet_print(f'Created classification dataset from dataset {dataset} in {time_diff:.3f} seconds')


	return classification_dataset


################################################################################
# Create HDF5 file (*.h5) from dataset
################################################################################

def create_dataset_h5(
	dataset,
	train_classes,
	train_data,
	train_filenames,
	train_labels,
	test_classes,
	test_data,
	test_filenames,
	test_labels):

	train_classes   = train_classes.astype('S')
	train_filenames = train_filenames.astype('S')
	train_labels    = train_labels.astype('S')
	test_classes    = test_classes.astype('S')
	test_filenames  = test_filenames.astype('S')
	test_labels     = test_labels.astype('S')

	with h5py.File(dataset, 'w') as h5py_file:
		h5py_file.create_dataset('train_classes',   data=train_classes,   compression='lzf')
		h5py_file.create_dataset('train_data',      data=train_data,      compression='lzf')
		h5py_file.create_dataset('train_filenames', data=train_filenames, compression='lzf')
		h5py_file.create_dataset('train_labels',    data=train_labels,    compression='lzf')
		h5py_file.create_dataset('test_classes',    data=test_classes,    compression='lzf')
		h5py_file.create_dataset('test_data',       data=test_data,       compression='lzf')
		h5py_file.create_dataset('test_filenames',  data=test_filenames,  compression='lzf')
		h5py_file.create_dataset('test_labels',     data=test_labels,     compression='lzf')


################################################################################
# Create dataset overview from dataset
################################################################################

def create_dataset_overview(current_dataset, output_dir):
	classes      = current_dataset.data['train']['classes']
	train_labels = current_dataset.data['train']['labels']
	test_labels  = current_dataset.data['test']['labels']
	dataset      = current_dataset.input_dir_or_file


	# Prepare dataset overview table: entries

	total_string = 'Total'

	unique, counts             = np.unique(train_labels, return_counts=True)
	train_labels_unique_counts = dict(zip(unique, counts))
	unique, counts             = np.unique(test_labels, return_counts=True)
	test_labels_unique_counts  = dict(zip(unique, counts))

	labels_unique_counts = {key: train_value + test_value for (key, train_value), (_, test_value) in \
		zip(train_labels_unique_counts.items(), test_labels_unique_counts.items())}

	train_labels_unique_counts_total = sum(train_labels_unique_counts.values())
	test_labels_unique_counts_total  = sum(test_labels_unique_counts.values())
	labels_unique_counts_total       = sum(labels_unique_counts.values())

	entries_max_len = max(np.vectorize(len)(classes).max(), len(total_string), len(str(labels_unique_counts_total)))


	# Create dataset overview table: rows and columns

	total_string = total_string.rjust(entries_max_len)

	header_entries = '|'.join([f' {c.rjust(entries_max_len)} '      for c in classes])
	train_entries  = '|'.join([f' {str(v).rjust(entries_max_len)} ' for v in train_labels_unique_counts.values()])
	test_entries   = '|'.join([f' {str(v).rjust(entries_max_len)} ' for v in test_labels_unique_counts.values()])
	total_entries  = '|'.join([f' {str(v).rjust(entries_max_len)} ' for v in labels_unique_counts.values()])

	train_entries_total = str(train_labels_unique_counts_total).rjust(entries_max_len, ' ')
	test_entries_total  = str(test_labels_unique_counts_total).rjust(entries_max_len, ' ')
	total_entries_total = str(labels_unique_counts_total).rjust(entries_max_len, ' ')

	header = '| Set \ Class |' + header_entries + '| ' + total_string        + ' |'
	train  = '| Train       |' + train_entries  + '| ' + train_entries_total + ' |'
	test   = '| Test        |' + test_entries   + '| ' + test_entries_total  + ' |'
	total  = '| Total       |' + total_entries  + '| ' + total_entries_total + ' |'
	hline  = len(header) * '-'

	dataset_overview = \
		f'{hline}\n'  \
		f'{header}\n' \
		f'{hline}\n'  \
		f'{train}\n'  \
		f'{test}\n'   \
		f'{hline}\n'  \
		f'{total}\n'  \
		f'{hline}'


	# Output dataset overview table

	Hexnet_print(f'Dataset overview\n{dataset_overview}')

	if output_dir:
		filename = os.path.join(output_dir, f'{os.path.basename(dataset)}_dataset_overview.dat')

		os.makedirs(output_dir, exist_ok=True)

		with open(filename, 'w') as file:
			print(dataset_overview, file=file)


################################################################################
# copytree (recursively copy an entire directory tree): ignore files function
################################################################################

def copytree_ignore_files(directory, files):
	return [file for file in files if os.path.isfile(os.path.join(directory, file))]


################################################################################
# Load dataset into memory: load single file
################################################################################

def load_file(current_file, class_file):
	current_file_lower = current_file.lower()

	if current_file_lower.endswith('.csv'):
		file_data = np.loadtxt(class_file, delimiter=',')
		file_type = 'CSV'
	elif current_file_lower.endswith('.npy'):
		file_data = np.load(class_file)
		file_type = 'NPY'
	elif current_file_lower.endswith('.txt'):
		with open(class_file) as f:
			file_data = f.read()
			file_type = 'TXT'
	else:
		file_data = cv2.cvtColor(cv2.imread(class_file, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

	return file_data, file_type


################################################################################
# Load dataset into memory
################################################################################

def load_dataset(dataset_string, create_h5=False, verbosity_level=2):
	Hexnet_print(f'Loading dataset {dataset_string}')


	start_time = time()


	current_dataset = dataset()

	train_classes   = []
	train_data      = []
	train_filenames = []
	train_labels    = []
	test_classes    = []
	test_data       = []
	test_filenames  = []
	test_labels     = []

	loaded_dataset = False


	# Determine the type of dataset

	dataset_is_file         = False
	dataset_is_dir_of_files = False
	dataset_is_dir_of_dirs  = False

	dataset_dirs = natsorted(glob(os.path.join(dataset_string, '*')))

	if os.path.isfile(dataset_string):
		dataset_is_file = True
	else:
		for dataset_dir in dataset_dirs:
			if os.path.isfile(dataset_dir):
				dataset_is_dir_of_files = True
			else:
				dataset_is_dir_of_dirs = True


	# Load the dataset

	# Dataset is file
	if dataset_is_file and dataset_string.lower().endswith('.h5'):
		with h5py.File(dataset_string, 'r') as h5py_file:
			train_classes   = np.asarray(h5py_file['train_classes']).astype('U')
			train_data      = np.asarray(h5py_file['train_data'])
			train_filenames = np.asarray(h5py_file['train_filenames']).astype('U')
			train_labels    = np.asarray(h5py_file['train_labels']).astype('U')
			test_classes    = np.asarray(h5py_file['test_classes']).astype('U')
			test_data       = np.asarray(h5py_file['test_data'])
			test_filenames  = np.asarray(h5py_file['test_filenames']).astype('U')
			test_labels     = np.asarray(h5py_file['test_labels']).astype('U')

		loaded_dataset = True

	# Dataset is directory of files
	elif dataset_is_dir_of_files:
		Hexnet_print_warning('Cannot load dataset from directory of files')

	# Dataset is directory of directories: datasets without train or test sets
	elif dataset_is_dir_of_dirs and not any('train' in c or 'test' in c for c in dataset_dirs):
		for set_class in dataset_dirs:
			current_class = os.path.basename(set_class)

			if verbosity_level >= 2:
				Hexnet_print(f'\t\t> current_class={current_class}')

			train_classes.append(current_class)

			for class_file in tqdm(natsorted(glob(os.path.join(set_class, '*')))):
				current_file = os.path.basename(class_file)

				if verbosity_level >= 3:
					Hexnet_print(f'\t\t\t> current_file={current_file}')

				file_data, file_type = load_file(current_file, class_file)

				train_data.append(file_data)
				train_filenames.append(current_file)
				train_labels.append(current_class)

		train_classes   = np.asarray(train_classes)
		train_data      = np.asarray(train_data)
		train_filenames = np.asarray(train_filenames)
		train_labels    = np.asarray(train_labels)

		loaded_dataset = True

	# Dataset is directory of directories
	elif dataset_is_dir_of_dirs:
		for dataset_set in dataset_dirs:
			current_set = os.path.basename(dataset_set)

			if verbosity_level >= 1:
				Hexnet_print(f'\t> current_set={current_set}')

			for set_class in natsorted(glob(os.path.join(dataset_set, '*'))):
				current_class = os.path.basename(set_class)

				if verbosity_level >= 2:
					Hexnet_print(f'\t\t> current_class={current_class}')

				if 'train' in current_set:
					train_classes.append(current_class)
				elif 'test' in current_set:
					test_classes.append(current_class)

				for class_file in tqdm(natsorted(glob(os.path.join(set_class, '*')))):
					current_file = os.path.basename(class_file)

					if verbosity_level >= 3:
						Hexnet_print(f'\t\t\t> current_file={current_file}')

					file_data, file_type = load_file(current_file, class_file)

					if 'train' in current_set:
						train_data.append(file_data)
						train_filenames.append(current_file)
						train_labels.append(current_class)
					elif 'test' in current_set:
						test_data.append(file_data)
						test_filenames.append(current_file)
						test_labels.append(current_class)


		if file_type == 'TXT':
			# Convert train and test data to TF-IDF features

			train_test_data = train_data + test_data

			vectorizer = TfidfVectorizer()

			vectorizer.fit(train_test_data)

			train_data = vectorizer.transform(train_data)
			test_data  = vectorizer.transform(test_data)

			train_data = [row for row in train_data.toarray()]
			test_data  = [row for row in test_data.toarray()]


		# Zero-fill ragged nested sequences

		data_max_size = max([data.size for data in train_data + test_data])

		for data_index, data in enumerate(train_data):
			if data.size < data_max_size:
				train_data[data_index] = np.pad(train_data[data_index], (0, data_max_size - data.size), 'constant', constant_values=0)

		for data_index, data in enumerate(test_data):
			if data.size < data_max_size:
				test_data[data_index] = np.pad(test_data[data_index], (0, data_max_size - data.size), 'constant', constant_values=0)


		train_classes   = np.asarray(train_classes)
		train_data      = np.asarray(train_data)
		train_filenames = np.asarray(train_filenames)
		train_labels    = np.asarray(train_labels)
		test_classes    = np.asarray(test_classes)
		test_data       = np.asarray(test_data)
		test_filenames  = np.asarray(test_filenames)
		test_labels     = np.asarray(test_labels)

		loaded_dataset = True

	# No dataset provided
	elif all(not dataset_type for dataset_type in (dataset_is_file, dataset_is_dir_of_files, dataset_is_dir_of_dirs)):
		Hexnet_print_warning('No dataset provided')

	# No identifiable dataset provided
	else:
		Hexnet_print_warning('No identifiable dataset provided')


	if loaded_dataset:
		time_diff = time() - start_time

		Hexnet_print(f'Loaded dataset {dataset_string} in {time_diff:.3f} seconds')


	if create_h5:
		dataset_string = f'{dataset_string}.h5'

		create_dataset_h5(
			dataset_string,
			train_classes,
			train_data,
			train_filenames,
			train_labels,
			test_classes,
			test_data,
			test_filenames,
			test_labels)


	current_dataset.input_dir_or_file = dataset_string

	current_dataset.data['train'] = {}
	current_dataset.data['test']  = {}

	current_dataset.data['train']['classes']   = train_classes
	current_dataset.data['train']['data']      = train_data
	current_dataset.data['train']['filenames'] = train_filenames
	current_dataset.data['train']['labels']    = train_labels

	current_dataset.data['test']['classes']   = test_classes
	current_dataset.data['test']['data']      = test_data
	current_dataset.data['test']['filenames'] = test_filenames
	current_dataset.data['test']['labels']    = test_labels

	current_dataset.loaded_dataset = loaded_dataset


	return current_dataset


################################################################################
# Transform dataset and save to memory
################################################################################

def transform_dataset(
	dataset,
	output_dir,
	mode            = 's2h',
	rad_o           = 1.0,
	len             = 1.0,
	res             = (64, 64),
	method          = 0,
	verbosity_level = 2):

	if os.path.exists(output_dir):
		Hexnet_print(f'Dataset {output_dir} exists already (skipping transformation)')
		return

	if os.path.isfile(f'{output_dir}.h5'):
		Hexnet_print(f'Dataset {output_dir}.h5 exists already (skipping transformation)')
		return

	Hexnet_print(f'Transforming dataset {dataset}')

	start_time = time()

	increase_verbosity = True if verbosity_level >= 3 else False

	shutil.copytree(dataset, output_dir, ignore=copytree_ignore_files)

	for directory in natsorted(glob(os.path.join(dataset, '**/'), recursive=True)):
		if verbosity_level >= 1:
			Hexnet_print(f'\t> directory={directory}')

		directory_filename_s = os.path.join(directory, '*')

		found_file = False

		for file in glob(directory_filename_s):
			if os.path.isfile(file):
				found_file = True
				break

		if not found_file:
			continue

		directory_output_dir = os.path.join(output_dir, os.path.relpath(directory, dataset))

		if mode == 's2h':
			Hexsamp_s2h(
				filename_s         = directory_filename_s,
				output_dir         = directory_output_dir,
				rad_o              = rad_o,
				method             = method,
				increase_verbosity = increase_verbosity)
		elif mode == 'h2s':
			Hexsamp_h2s(
				filename_s         = directory_filename_s,
				output_dir         = directory_output_dir,
				len                = len,
				method             = method,
				increase_verbosity = increase_verbosity)
		elif mode == 'h2h':
			Hexsamp_h2h(
				filename_s         = directory_filename_s,
				output_dir         = directory_output_dir,
				rad_o              = rad_o,
				method             = method,
				increase_verbosity = increase_verbosity)
		elif mode == 's2s':
			Sqsamp_s2s(
				filename_s         = directory_filename_s,
				output_dir         = directory_output_dir,
				res                = res,
				method             = method,
				increase_verbosity = increase_verbosity)

	time_diff = time() - start_time

	Hexnet_print(f'Transformed dataset {dataset} in {time_diff:.3f} seconds')


################################################################################
# Resize dataset using "HxW" (e.g., 32x32)
################################################################################

def resize_dataset(dataset_s, resize_string, method='nearest'):
	# HxW
	resize   = resize_string.split('x')
	resize_H = int(resize[0])
	resize_W = int(resize[1])
	size     = (resize_H, resize_W)

	if type(dataset_s) is not list:
		dataset_s = list(dataset_s)

	dataset_s = [tf.image.resize(dataset, size, method).numpy() if dataset.size else None for dataset in dataset_s]

	return dataset_s


################################################################################
# Crop dataset using "HxW" with offset "+Y+X"
#  (e.g., 32x32+2+2, 32x32, or +2+2)
################################################################################

def crop_dataset(dataset_s, crop_string):
	# HxW+Y+X
	crop        = crop_string.split('+')
	crop_size   = crop[0].split('x')
	crop_offset = crop[1:]

	if type(dataset_s) is not list:
		dataset_s = list(dataset_s)

	if '+' not in crop_string:
		crop_Y = 0
		crop_X = 0
	else:
		crop_Y = int(crop_offset[0])
		crop_X = int(crop_offset[1])

	if 'x' not in crop_string:
		crop_H = dataset_s[0].shape[1] - crop_Y
		crop_W = dataset_s[0].shape[2] - crop_X
	else:
		crop_H = int(crop_size[0])
		crop_W = int(crop_size[1])

	slice_H = slice(crop_Y, crop_Y + crop_H)
	slice_W = slice(crop_X, crop_X + crop_W)

	dataset_s = [dataset[:, slice_H, slice_W, :] if dataset.size else None for dataset in dataset_s]

	return dataset_s


################################################################################
# Pad dataset using "T,B,L,R" (e.g., 2,2,2,2)
################################################################################

def pad_dataset(dataset_s, pad_string, mode='constant', constant_values=0):
	# T,B,L,R
	pad       = pad_string.split(',')
	pad_H     = (int(pad[0]), int(pad[1]))
	pad_W     = (int(pad[2]), int(pad[3]))
	pad_width = ((0, 0), pad_H, pad_W, (0, 0))

	if type(dataset_s) is not list:
		dataset_s = list(dataset_s)

	dataset_s = [np.pad(dataset, pad_width, mode, constant_values=constant_values) if dataset.size else None for dataset in dataset_s]

	return dataset_s


################################################################################
# Show dataset via Matplotlib
################################################################################

def show_dataset(
	train_classes,
	train_data,
	train_labels,
	test_classes,
	test_data,
	test_labels,
	max_images_per_class   =  1,
	max_classes_to_display = 10):

	if train_data.size:
		classes_len = len(train_classes)
	elif test_data.size:
		classes_len = len(test_classes)

	nrows     = 2 * max_images_per_class
	ncols     = min(classes_len, max_classes_to_display)
	figsize_2 = max_images_per_class * ncols
	index     = 1

	plt.figure('Dataset classes')
	plt.subplots_adjust(wspace=0.5, hspace=0.5)

	for class_counter, train_class in enumerate(train_classes):
		if class_counter == max_classes_to_display:
			break

		class_indices = np.where(train_labels == train_class)[0]
		class_labels  = train_labels[class_indices]
		class_data    = train_data[class_indices]

		for image_counter in range(max_images_per_class):
			train_label = class_labels[image_counter]

			if train_label.ndim:
				train_label = list(train_label)

			plt.subplot(nrows, ncols, index)
			plt.title(f'train image {index}\n(label {train_label})')
			plt.imshow(class_data[image_counter])

			index += 1

	index = figsize_2 + 1

	for class_counter, test_class in enumerate(test_classes):
		if class_counter == max_classes_to_display:
			break

		class_indices = np.where(test_labels == test_class)[0]
		class_labels  = test_labels[class_indices]
		class_data    = test_data[class_indices]

		for image_counter in range(max_images_per_class):
			test_label = class_labels[image_counter]

			if test_label.ndim:
				test_label = list(test_label)

			plt.subplot(nrows, ncols, index)
			plt.title(f'test image {index - figsize_2}\n(label {test_label})')
			plt.imshow(class_data[image_counter])

			index += 1

	plt.show()

	plt.close()


################################################################################
# Visualize dataset overview and save to memory
################################################################################

def create_dataset_overview_visualization(dataset_visualized, classes_per_set, samples_per_class, mode, randomized_visualization=False):
	image_basename = os.path.join(dataset_visualized, os.path.basename(dataset_visualized))

	if mode == 'square':
		image_wildcard = '*_sq.png'
	elif mode == 'hexagonal':
		image_wildcard = '*_hex.png'
	else: # 'baseline'
		image_wildcard = '*.png'


	sets    = natsorted(glob(os.path.join(dataset_visualized, '*')))
	classes = natsorted(glob(os.path.join(sets[0], '*')))

	if classes_per_set is None or classes_per_set == 'all':
		classes_per_set = len(classes)
	else:
		classes_per_set = int(classes_per_set)


	sample_image = natsorted(glob(os.path.join(classes[0], image_wildcard)))[0]
	sample_image = cv2.imread(sample_image, cv2.IMREAD_COLOR)

	image_shape      = sample_image.shape
	image_dtype      = sample_image.dtype
	image_fill_value = np.iinfo(image_dtype).max


	# Dataset overview image's and class overview images' shapes

	classes_per_set   = min(classes_per_set, len(classes))
	samples_per_class = min(samples_per_class, 3)

	image_spacing = (10, 10)

	overview_image_size       = (samples_per_class, classes_per_set)
	class_overview_image_size = overview_image_size

	overview_image_shape = (
		overview_image_size[0] * image_shape[0] + (overview_image_size[0] - 1) * image_spacing[0],
		overview_image_size[1] * image_shape[1] + (overview_image_size[1] - 1) * image_spacing[1],
		image_shape[2]
	)

	class_overview_image_shape = (
		class_overview_image_size[0] * image_shape[0] + (class_overview_image_size[0] - 1) * image_spacing[0],
		class_overview_image_size[1] * image_shape[1] + (class_overview_image_size[1] - 1) * image_spacing[1],
		image_shape[2]
	)


	# Dataset overview image

	overview_image = np.full(shape=overview_image_shape, fill_value=image_fill_value, dtype=image_dtype)

	for current_class_index in range(overview_image_size[1]):
		current_class = classes[current_class_index]

		available_images     = natsorted(glob(os.path.join(current_class, image_wildcard)))
		available_images_len = len(available_images)

		if not current_class_index:
			x_start_index = current_class_index * image_shape[1]
		else:
			x_start_index = current_class_index * (image_shape[1] + image_spacing[1])

		if randomized_visualization:
			current_images = []

		for current_image_index in range(overview_image_size[0]):
			if randomized_visualization:
				while True:
					current_image = random.randint(0, available_images_len - 1)

					if current_image not in current_images:
						current_images.append(current_image)
						break
			else:
				current_image = min(current_image_index, available_images_len - 1)

			current_image = available_images[current_image]
			current_image = cv2.imread(current_image, cv2.IMREAD_COLOR)

			if not current_image_index:
				y_start_index = current_image_index * image_shape[0]
			else:
				y_start_index = current_image_index * (image_shape[0] + image_spacing[0])

			overview_image[
				y_start_index : y_start_index + image_shape[0],
				x_start_index : x_start_index + image_shape[1], :] = current_image

	cv2.imwrite(f'{image_basename}_dataset_overview_image.jpg', overview_image)


	# Dataset class overview images

	for current_class_index in range(overview_image_size[1]):
		class_overview_image = np.full(shape=class_overview_image_shape, fill_value=image_fill_value, dtype=image_dtype)

		current_class          = classes[current_class_index]
		current_class_basename = os.path.basename(current_class)

		available_images     = glob(os.path.join(current_class, image_wildcard))
		available_images_len = len(available_images)

		for x in range(class_overview_image_size[1]):
			if not x:
				x_start_index = x * image_shape[1]
			else:
				x_start_index = x * (image_shape[1] + image_spacing[1])

			if randomized_visualization:
				current_images = []

			for y in range(class_overview_image_size[0]):
				if randomized_visualization:
					while True:
						current_image = random.randint(0, available_images_len - 1)

						if current_image not in current_images:
							current_images.append(current_image)
							break
				else:
					current_image = min(y * class_overview_image_size[1] + x, available_images_len - 1)

				current_image = available_images[current_image]
				current_image = cv2.imread(current_image, cv2.IMREAD_COLOR)

				if not y:
					y_start_index = y * image_shape[0]
				else:
					y_start_index = y * (image_shape[0] + image_spacing[0])

				overview_image[
					y_start_index : y_start_index + image_shape[0],
					x_start_index : x_start_index + image_shape[1], :] = current_image

		cv2.imwrite(f'{image_basename}_dataset_class_overview_image_{current_class_basename}.jpg', overview_image)


################################################################################
# Visualize dataset and save to memory
################################################################################

def visualize_dataset(
	dataset,
	train_classes       = None,
	train_data          = None,
	train_filenames     = None,
	train_labels        = None,
	test_classes        = None,
	test_data           = None,
	test_filenames      = None,
	test_labels         = None,
	output_dir          = None,
	colormap            = None,
	classes_per_set     = None,
	samples_per_class   = None,
	visualize_square    = None,
	visualize_hexagonal = None,
	create_h5           = False,
	verbosity_level     = 2):

	Hexnet_print(f'Visualizing dataset {dataset}')

	start_time = time()

	output_dir_dataset = os.path.join(output_dir, os.path.basename(dataset))

	if create_h5:
		dataset = f'{output_dir_dataset}_visualized.h5'

		create_dataset_h5(
			dataset,
			train_classes,
			train_data,
			train_filenames,
			train_labels,
			test_classes,
			test_data,
			test_filenames,
			test_labels)
	elif os.path.isfile(dataset) and dataset.lower().endswith('.csv'):
		dataset_visualized = f'{output_dir_dataset}_visualized'

		with open(dataset) as dataset_file:
			dataset_reader = csv.reader(dataset_file)
			dataset_data   = list(dataset_reader)[1:]

		for label, filename, data in tqdm(dataset_data):
			current_output_dir = os.path.join(dataset_visualized, label)
			os.makedirs(current_output_dir, exist_ok=True)

			with open(os.path.join(current_output_dir, filename), 'w') as current_data_file:
				print(data.replace('"', ''), file=current_data_file)
	else:
		dataset_visualized = f'{output_dir_dataset}_visualized'

		if samples_per_class is not None and samples_per_class != 'all':
			visualization_counter = {}

			for current_class in set(train_labels):
				visualization_counter[current_class] = 0

			samples_per_class = int(samples_per_class)

		if os.path.isfile(dataset) and dataset.lower().endswith('.h5'):
			for current_class in train_classes:
				os.makedirs(os.path.join(dataset_visualized, 'train', current_class), exist_ok=True)

			for current_class in test_classes:
				os.makedirs(os.path.join(dataset_visualized, 'test', current_class), exist_ok=True)
		else:
			shutil.copytree(dataset, dataset_visualized, ignore=copytree_ignore_files)

		for current_set, current_data, current_filenames, current_labels in \
		 zip(('train', 'test'), (train_data, test_data), (train_filenames, test_filenames), (train_labels, test_labels)):

			if (type(current_data) is list and not current_data) or not current_data.size:
				continue

			if samples_per_class is not None and samples_per_class != 'all':
				for current_class in set(train_labels):
					visualization_counter[current_class] = 0

			if verbosity_level >= 1:
				Hexnet_print(f'\t> current_set={current_set}')

			for file, filename, label in zip(tqdm(current_data), current_filenames, current_labels):
				if samples_per_class is not None and samples_per_class != 'all' and \
				   visualization_counter[label] >= classes_per_set * samples_per_class:
					continue

				if samples_per_class is not None and samples_per_class != 'all':
					visualization_counter[label] = visualization_counter[label] + 1

				if os.path.isdir(os.path.join(dataset_visualized, current_set)):
					filename = os.path.join(dataset_visualized, current_set, label, filename)
				else:
					filename = os.path.join(dataset_visualized, label, filename)

				if verbosity_level >= 3:
					Hexnet_print(f'\t\t\t> filename={filename}')

				filename_lower = filename.lower()

				if filename_lower.endswith('.csv'):
					np.savetxt(filename, np.reshape(file, newshape = (1, file.shape[0])), delimiter=',')
				elif filename_lower.endswith('.npy'):
					np.save(filename, file)
				else:
					if colormap is not None:
						file = cv2.cvtColor(file, cv2.COLOR_RGB2GRAY)

					imsave(filename, file, cmap=colormap)

					if visualize_square:
						_filename = '.'.join(filename.split('.')[:-1])
						visualize_array(normalize_array(file), _filename, colormap)

					if visualize_hexagonal:
						_filename = '.'.join(filename.split('.')[:-1])
						visualize_hexarray(normalize_array(file), _filename, colormap)

	if visualize_square:
		overview_visualization_mode = 'square'
	elif visualize_hexagonal:
		overview_visualization_mode = 'hexagonal'
	else:
		overview_visualization_mode = 'baseline'

	# TODO: datasets without train or test sets
	try:
		create_dataset_overview_visualization(dataset_visualized, classes_per_set, samples_per_class, overview_visualization_mode)
	except:
		pass

	time_diff = time() - start_time

	Hexnet_print(f'Visualized dataset {dataset} in {time_diff:.3f} seconds')

