'''****************************************************************************
 * misc.py: Helper Functions
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

import decimal
import sys

import numpy as np


def putchar(char):
	sys.stdout.write(char)

def print_newline():
	putchar('\n')


def Hexnet_print(string, filename=None):
	if not filename:
		print(f'[Hexnet] {string}')
	else:
		with open(filename, 'w') as file:
			print(f'[Hexnet] {string}', file=file)

def Hexnet_print_warning(string, filename=None):
	Hexnet_print(f'(WARNING) {string}', filename)


def test_array(array):
	if type(array) is not np.ndarray:
		array = np.asarray(array)

	return array

def test_image_batch(image_batch):
	image_batch = test_array(image_batch)

	if image_batch.ndim == 3:
		image_batch = image_batch[np.newaxis, ...]

	return image_batch


def normalize_array(array):
	array = test_array(array)

	array_min   = array.min()
	array_max   = array.max()
	array_range = array_max - array_min

	if array_range:
		array = (array - array_min) / array_range
	elif array_min:
		array /= array_min

	return array


def array_to_one_hot_array(array, classes):
	array = test_array(array)

	one_hot_array = np.zeros_like(array, shape = (array.shape[0], classes))

	one_hot_array[np.arange(array.shape[0]), array.T] = 1

	return one_hot_array


def round_half_up(value):
	return int(decimal.Decimal(value).quantize(decimal.Decimal('1'), rounding=decimal.ROUND_HALF_UP))

