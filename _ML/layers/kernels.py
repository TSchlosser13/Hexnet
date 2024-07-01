#!/usr/bin/env python3.7


'''****************************************************************************
 * kernels.py: Kernels for Hexagonal Layers
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

import math

import numpy      as np
import tensorflow as tf

if __name__ == '__main__':
	import sys
	sys.path[0] = '..'

from layers.masks import build_masks
from misc.misc    import round_half_up


################################################################################
# Build kernels for square group convolution layers
################################################################################

def rotate_square_kernel(kernel, angle=90):
	if type(kernel) is np.ndarray:
		angle %= 360

		if angle == 90:
			kernel_rotated = list(zip(*reversed(kernel)))
		elif angle == -90 or angle == 270:
			kernel_rotated = list(reversed(list(zip(*kernel))))
		elif angle == 180:
			kernel_rotated = list(reversed([list(reversed(row)) for row in kernel]))
	elif tf.is_tensor(kernel):
		# HWIOC -> NHWC
		kernel = tf.transpose(
			a    = kernel,
			perm = (3, 0, 1, 2),
			name = 'rotate_square_kernel_kernel_transpose')

		kernel_rotated = tf.image.rot90(
			image = kernel,
			k     = 1,
			name  = 'rotate_square_kernel_kernel_rotated_rot90')

		# NHWC -> HWIOC
		kernel_rotated = tf.transpose(
			a    = kernel_rotated,
			perm = (1, 2, 3, 0),
			name = 'rotate_square_kernel_kernel_rotated_transpose')

	return kernel_rotated


################################################################################
# Build kernels for hexagonal group convolution layers
################################################################################

# Helper function: rotate point by a given angle around the center of the image
def rotate_hexagonal_coordinate(p, angle=60, convert_to_radians=True):
	if convert_to_radians:
		angle = math.radians(angle)

	return (math.cos(angle) * p[0] - math.sin(angle) * p[1], math.sin(angle) * p[0] + math.cos(angle) * p[1])

# Build kernels function
def rotate_hexagonal_kernels(kernels, angle=60):
	even = 0
	odd  = 1


	kernel_h           = kernels[even].shape[0]
	kernel_w           = kernels[even].shape[1]
	kernel_center      = int(kernel_h / 2)
	center_is_even_row = 1 - kernel_center % 2

	if type(kernels[even]) is np.ndarray:
		kernels_rotated = (
			[[0 for w in range(kernel_w)] for h in range(kernel_h)] if kernels[even] is not None else None,
			[[0 for w in range(kernel_w)] for h in range(kernel_h)] if kernels[odd]  is not None else None
		)
	elif tf.is_tensor(kernels[even]):
		kernels_rotated = (
			[[kernels[even][h, w, :, :] for w in range(kernel_w)] for h in range(kernel_h)] if kernels[even] is not None else None,
			[[kernels[odd][h, w, :, :]  for w in range(kernel_w)] for h in range(kernel_h)] if kernels[odd]  is not None else None
		)

	if type(kernels[even]) is np.ndarray:
		kernel_masks = build_masks(mask_shape=kernels[even].shape)
	elif tf.is_tensor(kernels[even]):
		kernel_masks = build_masks(mask_shape=kernels[even].shape[0:2])


	for kernel in kernels:
		if kernel is None:
			continue

		if kernel is kernels[even]:
			kernel_rotated = kernels_rotated[even]
			kernel_mask    = kernel_masks[even]
			shift_factor   = 1
		else:
			kernel_rotated = kernels_rotated[odd]
			kernel_mask    = kernel_masks[odd]
			shift_factor   = -1

		for h in range(kernel_h):
			for w in range(kernel_w):
				if not kernel_mask[h][w]:
					continue
				elif h == kernel_center and w == kernel_center:
					if type(kernels[even]) is np.ndarray:
						kernel_rotated[kernel_center][kernel_center] = kernel[kernel_center][kernel_center]
					elif tf.is_tensor(kernels[even]):
						kernel_rotated[kernel_center][kernel_center] = kernel[kernel_center, kernel_center, :, :]

					continue

				coordinate = (kernel_center - h, w - kernel_center)

				if center_is_even_row:
					if not h % 2:
						coordinate_hex = (coordinate[0] * 1.5, coordinate[1] * math.sqrt(3))
					else:
						coordinate_hex = (coordinate[0] * 1.5, (coordinate[1] + shift_factor * 0.5) * math.sqrt(3))
				else:
					if not h % 2:
						coordinate_hex = (coordinate[0] * 1.5, (coordinate[1] - shift_factor * 0.5) * math.sqrt(3))
					else:
						coordinate_hex = (coordinate[0] * 1.5, coordinate[1] * math.sqrt(3))

				coordinate_hex_rotated = rotate_hexagonal_coordinate(coordinate_hex, angle)

				coordinate_rotated_row = round_half_up(coordinate_hex_rotated[0] / 1.5)

				if center_is_even_row:
					if not coordinate_rotated_row % 2:
						coordinate_rotated = (coordinate_rotated_row, coordinate_hex_rotated[1] / math.sqrt(3))
					else:
						coordinate_rotated = (coordinate_rotated_row, coordinate_hex_rotated[1] / math.sqrt(3) - shift_factor * 0.5)
				else:
					if not coordinate_rotated_row % 2:
						coordinate_rotated = (coordinate_rotated_row, coordinate_hex_rotated[1] / math.sqrt(3))
					else:
						coordinate_rotated = (coordinate_rotated_row, coordinate_hex_rotated[1] / math.sqrt(3) + shift_factor * 0.5)

				coordinate_rotated = (coordinate_rotated[0], round_half_up(coordinate_rotated[1]))

				hwr = (kernel_h - 1 - (coordinate_rotated[0] + kernel_center), coordinate_rotated[1] + kernel_center)

				if type(kernels[even]) is np.ndarray:
					kernel_rotated[hwr[0]][hwr[1]] = kernel[h][w]
				elif tf.is_tensor(kernels[even]):
					kernel_rotated[hwr[0]][hwr[1]] = kernel[h, w, :, :]


	if tf.is_tensor(kernels[even]):
		kernels_rotated_h = [[], []]

		for h in range(kernel_h):
			kernels_rotated_h[even].append(
				tf.stack(
					values = kernels_rotated[even][h],
					axis   = 0,
					name   = 'rotate_hexagonal_kernels_kernels_rotated_h_even_stack'))

			kernels_rotated_h[odd].append(
				tf.stack(
					values = kernels_rotated[odd][h],
					axis   = 0,
					name   = 'rotate_hexagonal_kernels_kernels_rotated_h_odd_stack'))

		kernels_rotated_h[even] = tf.stack(
			values = kernels_rotated_h[even],
			axis   = 0,
			name   = 'rotate_hexagonal_kernels_kernels_rotated_h_even_stack')

		kernels_rotated_h[odd] = tf.stack(
			values = kernels_rotated_h[odd],
			axis   = 0,
			name   = 'rotate_hexagonal_kernels_kernels_rotated_h_odd_stack')

		kernels_rotated = (kernels_rotated_h[even], kernels_rotated_h[odd])


	return kernels_rotated


################################################################################
# Show kernels for square group convolution layers
################################################################################

def test_rotate_square_kernel(kernel_size = (3, 3)):
	print(f'>> test_rotate_square_kernel: kernel_size={kernel_size}')

	test_kernel = np.zeros(shape=kernel_size, dtype=np.int32)

	for h in range(test_kernel.shape[0]):
		for w in range(test_kernel.shape[1]):
			test_kernel[h][w] = h * test_kernel.shape[1] + w + 1

	print(f'test_kernel =\n{test_kernel}')

	for angle in (90, 180, 270):
		print(f'> angle={angle}')

		test_kernel_rotated = rotate_square_kernel(kernel=test_kernel, angle=angle)

		print(f'test_kernel_rotated =\n{np.asarray(test_kernel_rotated)}')


################################################################################
# Show kernels for hexagonal group convolution layers
################################################################################

def test_rotate_hexagonal_kernels(kernel_size = (3, 3)):
	even = 0
	odd  = 1

	print(f'>> test_rotate_hexagonal_kernels: kernel_size={kernel_size}')

	test_kernels = build_masks(mask_shape=kernel_size)

	for h in range(test_kernels[even].shape[0]):
		for w in range(test_kernels[even].shape[1]):
			if test_kernels[even][h][w]:
				test_kernels[even][h][w] = h * test_kernels[even].shape[1] + w + 1

			if test_kernels[odd][h][w]:
				test_kernels[odd][h][w] = h * test_kernels[odd].shape[1] + w + 1

	print(f'test_kernels[even] =\n{test_kernels[even]}\ntest_kernels[odd] =\n{test_kernels[odd]}')

	for angle in (60, 120, 180, 240, 300):
		print(f'> angle={angle}')

		test_kernels_rotated = rotate_hexagonal_kernels(kernels=test_kernels, angle=angle)

		print(f'test_kernels_rotated[even] =\n{np.asarray(test_kernels_rotated[even])}\ntest_kernels_rotated[odd] =\n{np.asarray(test_kernels_rotated[odd])}')


################################################################################
# main
################################################################################

if __name__ == '__main__':
	for i in (3, 5, 7):
		test_rotate_square_kernel(kernel_size = (i, i))

	for i in (3, 5, 7):
		test_rotate_hexagonal_kernels(kernel_size = (i, i))

