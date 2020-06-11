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


import cv2
import h5py
import os
import random
import shutil
import uuid

import matplotlib.pyplot as plt
import numpy             as np
import tensorflow        as tf

from glob              import glob
from matplotlib.pyplot import imsave
from natsort           import natsorted
from time              import time
from tqdm              import tqdm

from core.Hexnet        import Hexsamp_s2h, Hexsamp_h2s, Hexsamp_h2h, Sqsamp_s2s
from misc.misc          import Hexnet_print, normalize_array
from misc.visualization import visualize_hexarray


def create_dataset(dataset, split_ratios, verbosity_level=2):
	Hexnet_print(f'Creating classification dataset from dataset {dataset}')

	start_time = time()

	split_ratios_len       = len(split_ratios)
	split_ratios_sets      = list(split_ratios.keys())
	split_ratios_fractions = list(split_ratios.values())

	classification_dataset = f'{dataset}_classification_dataset'
	os.makedirs(classification_dataset, exist_ok=True)


	for set_class in natsorted(glob(os.path.join(dataset, '*'))):

		# Step 1: randomized image dataset set assignment

		current_class = os.path.basename(set_class)

		if verbosity_level >= 1:
			Hexnet_print(f'\t> current_class={current_class}')

		images_to_copy = glob(os.path.join(set_class, '*'))

		if not images_to_copy:
			continue

		images_to_copy_len         = len(images_to_copy)
		max_images_to_copy_per_set = [round(fraction * images_to_copy_len) for fraction in split_ratios_fractions]
		copied_images              = []
		copied_images_per_set      = split_ratios_len * [0]

		if verbosity_level >= 2:
			Hexnet_print(f'\t\t> max_images_to_copy_per_set={max_images_to_copy_per_set} (images_to_copy_len={images_to_copy_len})')

		for current_set in split_ratios_sets:
			os.makedirs(os.path.join(classification_dataset, current_set, current_class), exist_ok=True)

		for image_to_copy in tqdm(random.sample(images_to_copy, images_to_copy_len)):
			while True:
				set_selector = random.randint(0, split_ratios_len - 1)

				if copied_images_per_set[set_selector] < max_images_to_copy_per_set[set_selector]:
					copy_image_to = os.path.join(classification_dataset, split_ratios_sets[set_selector], current_class, os.path.basename(image_to_copy))

					shutil.copyfile(image_to_copy, copy_image_to)

					copied_images.append(copy_image_to)
					copied_images_per_set[set_selector] += 1

					break

		copied_images_len = len(copied_images)

		if verbosity_level >= 2:
			Hexnet_print(f'\t\t> copied_images_per_set={copied_images_per_set} (copied_images_len={copied_images_len})')


		# Step 2: randomized image dataset set balancing: duplicate and hash assigned images

		for current_set_index, current_set in enumerate(split_ratios_sets):
			while copied_images_per_set[current_set_index] < max_images_to_copy_per_set[current_set_index]:
				image_selector = round(random.randint(0, copied_images_len - 1))

				image_to_copy = copied_images[image_selector]
				copy_image_to = os.path.basename(image_to_copy).split('.')
				copy_image_to = '.'.join(copy_image_to[:-1]) + '_' + str(uuid.uuid4()) + '.' + copy_image_to[-1]
				copy_image_to = os.path.join(classification_dataset, current_set, current_class, copy_image_to)

				shutil.copyfile(image_to_copy, copy_image_to)

				copied_images_per_set[current_set_index] += 1

		if verbosity_level >= 2:
			copied_images_len = sum(copied_images_per_set)
			Hexnet_print(f'\t\t> copied_images_per_set={copied_images_per_set} (copied_images_len={copied_images_len}) after balancing')


	time_diff = time() - start_time

	Hexnet_print(f'Created classification dataset from dataset {dataset} in {time_diff:.3f} seconds')


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
		h5py_file.create_dataset('train_classes',   data=train_classes)
		h5py_file.create_dataset('train_data',      data=train_data)
		h5py_file.create_dataset('train_filenames', data=train_filenames)
		h5py_file.create_dataset('train_labels',    data=train_labels)
		h5py_file.create_dataset('test_classes',    data=test_classes)
		h5py_file.create_dataset('test_data',       data=test_data)
		h5py_file.create_dataset('test_filenames',  data=test_filenames)
		h5py_file.create_dataset('test_labels',     data=test_labels)


def copytree_ignore_files(directory, files):
	return [file for file in files if os.path.isfile(os.path.join(directory, file))]


def load_dataset(dataset, create_h5=False, verbosity_level=2):
	Hexnet_print(f'Loading dataset {dataset}')

	start_time = time()

	train_classes   = []
	train_data      = []
	train_filenames = []
	train_labels    = []
	test_classes    = []
	test_data       = []
	test_filenames  = []
	test_labels     = []

	if os.path.isfile(dataset) and dataset.endswith('.h5'):
		with h5py.File(dataset, 'r') as h5py_file:
			train_classes   = np.asarray(h5py_file['train_classes']).astype('U')
			train_data      = np.asarray(h5py_file['train_data'])
			train_filenames = np.asarray(h5py_file['train_filenames']).astype('U')
			train_labels    = np.asarray(h5py_file['train_labels']).astype('U')
			test_classes    = np.asarray(h5py_file['test_classes']).astype('U')
			test_data       = np.asarray(h5py_file['test_data'])
			test_filenames  = np.asarray(h5py_file['test_filenames']).astype('U')
			test_labels     = np.asarray(h5py_file['test_labels']).astype('U')
	else:
		for dataset_set in natsorted(glob(os.path.join(dataset, '*'))):
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

				for class_image in natsorted(glob(os.path.join(set_class, '*'))):
					current_image = os.path.basename(class_image)

					if verbosity_level >= 3:
						Hexnet_print(f'\t\t\t> current_image={current_image}')

					if 'train' in current_set:
						train_data.append(cv2.imread(class_image, cv2.IMREAD_COLOR))
						train_filenames.append(current_image)
						train_labels.append(current_class)
					elif 'test' in current_set:
						test_data.append(cv2.imread(class_image, cv2.IMREAD_COLOR))
						test_filenames.append(current_image)
						test_labels.append(current_class)

		train_classes   = np.asarray(train_classes)
		train_data      = np.asarray(train_data)
		train_filenames = np.asarray(train_filenames)
		train_labels    = np.asarray(train_labels)
		test_classes    = np.asarray(test_classes)
		test_data       = np.asarray(test_data)
		test_filenames  = np.asarray(test_filenames)
		test_labels     = np.asarray(test_labels)

	time_diff = time() - start_time

	Hexnet_print(f'Loaded dataset {dataset} in {time_diff:.3f} seconds')

	if create_h5:
		dataset = f'{dataset}.h5'

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

	return ((train_classes, train_data, train_filenames, train_labels),
	        (test_classes,  test_data,  test_filenames,  test_labels))


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


def resize_dataset(dataset_s, resize_string, method='nearest'):
	# HxW
	resize   = resize_string.split('x')
	resize_H = int(resize[0])
	resize_W = int(resize[1])
	size     = (resize_H, resize_W)

	if type(dataset_s) is not list:
		dataset_s = list(dataset_s)

	dataset_s = [tf.image.resize(dataset, size, method).numpy() for dataset in dataset_s]

	return dataset_s


def crop_dataset(dataset_s, crop_string):
	# HxW+Y+X
	crop        = crop_string.split('+')
	crop_size   = crop[0].split('x')
	crop_offset = crop[1:]

	if type(dataset_s) is not list:
		dataset_s = list(dataset_s)

	if not '+' in crop_string:
		crop_Y = 0
		crop_X = 0
	else:
		crop_Y = int(crop_offset[0])
		crop_X = int(crop_offset[1])

	if not 'x' in crop_string:
		crop_H = dataset_s[0].shape[1] - crop_Y
		crop_W = dataset_s[0].shape[2] - crop_X
	else:
		crop_H = int(crop_size[0])
		crop_W = int(crop_size[1])

	slice_H = slice(crop_Y, crop_Y + crop_H)
	slice_W = slice(crop_X, crop_X + crop_W)

	dataset_s = [dataset[:, slice_H, slice_W, :] for dataset in dataset_s]

	return dataset_s


def pad_dataset(dataset_s, pad_string, mode='constant', constant_values=0):
	# T,B,L,R
	pad       = pad_string.split(',')
	pad_H     = (int(pad[0]), int(pad[1]))
	pad_W     = (int(pad[2]), int(pad[3]))
	pad_width = ((0, 0), pad_H, pad_W, (0, 0))

	if type(dataset_s) is not list:
		dataset_s = list(dataset_s)

	dataset_s = [np.pad(dataset, pad_width, mode, constant_values=constant_values) for dataset in dataset_s]

	return dataset_s


def show_dataset(
	train_classes,
	train_data,
	train_labels,
	test_classes,
	test_data,
	test_labels,
	max_images_per_class   =  1,
	max_classes_to_display = 10):

	nrows     = 2 * max_images_per_class
	ncols     = min(len(train_classes), max_classes_to_display)
	figsize_2 = max_images_per_class * ncols
	index     = 1

	plt.figure('Dataset classes')
	plt.subplots_adjust(wspace=0.5, hspace=0.5)

	for class_counter, train_class in enumerate(train_classes):
		if class_counter == max_classes_to_display:
			break

		class_label_indices = np.where(train_labels == train_class)[0]

		for image_counter in range(max_images_per_class):
			plt.subplot(nrows, ncols, index)
			plt.title(f'train image {index}\n(class {train_class})')
			plt.imshow(train_data[class_label_indices[image_counter]])

			index += 1

	index = figsize_2 + 1

	for class_counter, test_class in enumerate(test_classes):
		if class_counter == max_classes_to_display:
			break

		class_label_indices = np.where(test_labels == test_class)[0]

		for image_counter in range(max_images_per_class):
			plt.subplot(nrows, ncols, index)
			plt.title(f'test image {index - figsize_2}\n(class {test_class})')
			plt.imshow(test_data[class_label_indices[image_counter]])

			index += 1

	plt.show()

	plt.close()


def visualize_dataset(
	dataset,
	train_classes,
	train_data,
	train_filenames,
	train_labels,
	test_classes,
	test_data,
	test_filenames,
	test_labels,
	visualize_hexagonal,
	create_h5       = False,
	verbosity_level = 2):

	Hexnet_print(f'Visualizing dataset {dataset}')

	start_time = time()

	if create_h5:
		dataset = f'{dataset}_visualized.h5'

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
	else:
		dataset_visualized = f'{dataset}_visualized'

		if os.path.isfile(dataset) and dataset.endswith('.h5'):
			for current_class in train_classes:
				os.makedirs(os.path.join(dataset_visualized, 'train', current_class), exist_ok=True)

			for current_class in test_classes:
				os.makedirs(os.path.join(dataset_visualized, 'test', current_class), exist_ok=True)
		else:
			shutil.copytree(dataset, dataset_visualized, ignore=copytree_ignore_files)

		for current_set, current_data, current_filenames, current_labels in \
		 zip(('train', 'test'), (train_data, test_data), (train_filenames, test_filenames), (train_labels, test_labels)):

			if verbosity_level >= 1:
				Hexnet_print(f'\t> current_set={current_set}')

			for image, filename, label in zip(tqdm(current_data), current_filenames, current_labels):
				image_filename = os.path.join(dataset_visualized, current_set, label, filename)

				if verbosity_level >= 3:
					Hexnet_print(f'\t\t\t> image_filename={image_filename}')

				if not visualize_hexagonal:
					imsave(image_filename, image)
				else:
					image_filename = '.'.join(image_filename.split('.')[:-1])
					visualize_hexarray(normalize_array(image), image_filename)

	time_diff = time() - start_time

	Hexnet_print(f'Visualized dataset {dataset} in {time_diff:.3f} seconds')


