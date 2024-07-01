'''****************************************************************************
 * augmenters.py: Dataset Augmenters
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

import random
import uuid

import imgaug            as ia
import imgaug.augmenters as iaa
import numpy             as np

from core.Hexarray import rotate_Hexarray, scale_Hexarray, translate_Hexarray
from misc.misc     import test_image_batch


################################################################################
# Augmentation "size": augmentation via duplication
################################################################################

def augment_size(data, filenames, labels, labels_orig, augmentation_size):
	if augmentation_size != 1:
		target_size = round(augmentation_size * data.shape[0])

		if augmentation_size > 1:
			size_diff = target_size - data.shape[0]

			augmentation_selectors = np.random.randint(data.shape[0], size=size_diff)

			data_augmentation        = np.empty_like(data,        shape = (size_diff, data.shape[1], data.shape[2], data.shape[3]))
			filenames_augmentation   = np.empty_like(filenames,   shape=size_diff)
			labels_augmentation      = np.empty_like(labels,      shape=size_diff)
			labels_orig_augmentation = np.empty_like(labels_orig, shape=size_diff)

			for augmentation_selector_index, augmentation_selector in enumerate(augmentation_selectors):
				filename = filenames[augmentation_selector].split('.')

				data_augmentation[augmentation_selector_index]        = data[augmentation_selector]
				filenames_augmentation[augmentation_selector_index]   = '.'.join(filename[:-1]) + '_' + str(uuid.uuid4()) + '.' + filename[-1]
				labels_augmentation[augmentation_selector_index]      = labels[augmentation_selector]
				labels_orig_augmentation[augmentation_selector_index] = labels_orig[augmentation_selector]

			data        = np.concatenate((data,        data_augmentation))
			filenames   = np.concatenate((filenames,   filenames_augmentation))
			labels      = np.concatenate((labels,      labels_augmentation))
			labels_orig = np.concatenate((labels_orig, labels_orig_augmentation))
		else:
			size_diff = data.shape[0] - target_size

			while size_diff:
				deletion_selectors = np.random.randint(data.shape[0], size=size_diff)

				data        = np.delete(data,        deletion_selectors, axis=0)
				filenames   = np.delete(filenames,   deletion_selectors)
				labels      = np.delete(labels,      deletion_selectors)
				labels_orig = np.delete(labels_orig, deletion_selectors)

				size_diff = data.shape[0] - target_size

	return (data, filenames, labels, labels_orig)




################################################################################
# Augmenter "simple": rotation, translation, and scaling
################################################################################

def augmenter_simple(augmentation_level=1):
	lp = augmentation_level / 100

	sometimes = lambda aug: iaa.Sometimes(0.5, aug)

	augmenter = iaa.Sequential([
		sometimes(
			iaa.Affine(
				rotate            = (-2 * augmentation_level, 2 * augmentation_level),
				translate_percent = {'x': (-lp, lp), 'y': (-lp, lp)},
				scale             = {'x': (1 - lp, 1 + lp), 'y': (1 - lp, 1 + lp)}
			)
		),
		iaa.Flipud(1.0)
	])

	return augmenter


################################################################################
# Augmenter "complex_custom": different randomized transformations
################################################################################

def augmenter_complex_custom():
	# This augmeter is based on https://github.com/aleju/imgaug?tab=readme-ov-file#example-very-complex-augmentation-pipeline

	# Sometimes(0.5, ...) applies the given augmenter in 50% of all cases,
	# e.g. Sometimes(0.5, GaussianBlur(0.3)) would blur roughly every second image.
	sometimes = lambda aug: iaa.Sometimes(0.5, aug)

	# Define our sequence of augmentation steps that will be applied to every image
	# All augmenters with per_channel=0.5 will sample one value _per image_
	# in 50% of all cases. In all other cases they will sample new values
	# _per channel_.

	augmenter = iaa.Sequential(
		[
			# apply the following augmenters to most images
			iaa.Fliplr(0.5), # horizontally flip 50% of all images
			iaa.Flipud(0.2), # vertically flip 20% of all images
			# crop images by -5% to 10% of their height/width
			sometimes(iaa.CropAndPad(
				percent  = (-0.05, 0.1),
				pad_mode = ia.ALL,
				pad_cval = (0, 255)
			)),
			sometimes(iaa.Affine(
				scale             = {"x": (0.8, 1.2), "y": (0.8, 1.2)},   # scale images to 80-120% of their size, individually per axis
				translate_percent = {"x": (-0.2, 0.2), "y": (-0.2, 0.2)}, # translate by -20 to +20 percent (per axis)
				rotate            = (-45, 45),                            # rotate by -45 to +45 degrees
				shear             = (-16, 16),                            # shear by -16 to +16 degrees
				order             = [0, 1],                               # use nearest neighbour or bilinear interpolation (fast)
				cval              = (0, 255),                             # if mode is constant, use a cval between 0 and 255
				mode              = ia.ALL                                # use any of scikit-image's warping modes (see 2nd image from the top for examples)
			)),
			# execute 0 to 5 of the following (less important) augmenters per image
			# don't execute all of them, as that would often be way too strong
			iaa.SomeOf((0, 5),
				[
					sometimes(iaa.Superpixels(p_replace=(0, 1.0), n_segments=(20, 200))), # convert images into their superpixel representation
					iaa.OneOf([
						iaa.GaussianBlur((0, 3.0)), # blur images with a sigma between 0 and 3.0
						iaa.AverageBlur(k=(2, 7)),  # blur image using local means with kernel sizes between 2 and 7
						iaa.MedianBlur(k=(3, 11)),  # blur image using local medians with kernel sizes between 2 and 7
					]),
					iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)), # sharpen images
					iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)),      # emboss images
					# search either for all edges or for directed edges,
					# blend the result with the original image using a blobby mask
					iaa.SimplexNoiseAlpha(iaa.OneOf([
						iaa.EdgeDetect(alpha=(0.5, 1.0)),
						iaa.DirectedEdgeDetect(alpha=(0.5, 1.0), direction=(0.0, 1.0)),
					])),
					iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5), # add gaussian noise to images
					iaa.OneOf([
						iaa.Dropout((0.01, 0.1), per_channel=0.5), # randomly remove up to 10% of the pixels
						iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05), per_channel=0.2),
					]),
					iaa.Invert(0.05, per_channel=True),   # invert color channels
					iaa.Add((-10, 10), per_channel=0.5),  # change brightness of images (by -10 to 10 of original value)
					iaa.AddToHueAndSaturation((-20, 20)), # change hue and saturation
					# either change the brightness of the whole image (sometimes
					# per channel) or change the brightness of subareas
					iaa.OneOf([
						iaa.Multiply((0.5, 1.5), per_channel=0.5),
						iaa.FrequencyNoiseAlpha(
							exponent = (-4, 0),
							first    = iaa.Multiply((0.5, 1.5), per_channel=True),
							second   = iaa.LinearContrast((0.5, 2.0))
						)
					]),
					iaa.LinearContrast((0.5, 2.0), per_channel=0.5), # improve or worsen the contrast
					iaa.Grayscale(alpha=(0.0, 1.0)),
					sometimes(iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)), # move pixels locally around (with random strengths)
					sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))),                 # sometimes move parts of the image around
					sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.1)))
				],
				random_order=True
			)
		],
		random_order=True
	)


	return augmenter




################################################################################
# Augmentation "rotate_Hexarray": hexagonal rotation
################################################################################

def randomized_rotate_Hexarray(Hexarray_s):
	Hexarray_s = test_image_batch(Hexarray_s)

	angle_factor         = 60
	angle_selector_range = (1, 5)

	angle_selector = random.randint(angle_selector_range[0], angle_selector_range[1])
	angle          = angle_selector * angle_factor

	Hexarray_s_rotated = rotate_Hexarray(Hexarray_s, angle)

	return Hexarray_s_rotated


################################################################################
# Augmentation "scale_Hexarray": hexagonal scaling
################################################################################

def randomized_scale_Hexarray(Hexarray_s):
	Hexarray_s = test_image_batch(Hexarray_s)

	res_factor         = Hexarray_s.shape[1:3]
	res_selector_range = (0.5, 2)

	res_selector = random.uniform(res_selector_range[0], res_selector_range[1])
	res          = np.round(res_selector * res_factor)

	Hexarray_s_scaled = scale_Hexarray(Hexarray_s, res, method = 0, fill_value = 0)

	return Hexarray_s_scaled


################################################################################
# Augmentation "translate_Hexarray": hexagonal translation
################################################################################

def randomized_translate_Hexarray(Hexarray_s):
	Hexarray_s = test_image_batch(Hexarray_s)

	translation_factor         = Hexarray_s.shape[1:3]
	translation_selector_range = (-0.5, 0.5)

	translation_selector = np.random.uniform(translation_selector_range[0], translation_selector_range[1], size=2)
	translation          = np.round(translation_selector * translation_factor)

	Hexarray_s_translated = translate_Hexarray(Hexarray_s, translation, axis = (1, 2), fill_value = 0, cyclic_translation = False)

	return Hexarray_s_translated


################################################################################
# Augmenter "simple_hex_custom": hexagonal rotation, scaling, and translation
################################################################################

class augmenter_simple_hex_custom():
	def __init__(self):
		self.augmentation_functions = (randomized_rotate_Hexarray, randomized_scale_Hexarray, randomized_translate_Hexarray)

	def __call__(self, images):
		images_augmented = []

		for image in images:
			augmentation_selectors = np.random.random(size=3) < 0.5

			for augmentation_function, augmentation_selector in zip(self.augmentation_functions, augmentation_selectors):
				if augmentation_selector:
					image = augmentation_function(image)

			images_augmented.append(image)

		return images_augmented

