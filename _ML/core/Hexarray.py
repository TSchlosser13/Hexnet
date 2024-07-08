#!/usr/bin/env python3.7


'''****************************************************************************
 * Hexarray.py: Operations on Hexagonal Arrays
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

import numpy as np

if __name__ == '__main__':
	import sys
	sys.path[0] = '..'

from core.Hexint  import rotate_Hexint
from core.Hexsamp import Hexsamp_h2h
from misc.misc    import round_half_up, test_image_batch


################################################################################
# Rotate Hexarray by a given angle (n × 60°) around the center of the image
################################################################################

def rotate_Hexarray(Hexarray_s, angle=60):
	Hexarray_s = test_image_batch(Hexarray_s)

	Hexarray_h         = Hexarray_s.shape[1]
	Hexarray_w         = Hexarray_s.shape[2]
	Hexarray_center    = (int(Hexarray_h / 2), int(Hexarray_w / 2))
	center_is_even_row = 1 - Hexarray_center[0] % 2

	Hexarray_s_rotated = np.zeros_like(Hexarray_s)


	shift_factor = 1 # Hexarrays start with an even row

	for h in range(Hexarray_h):
		for w in range(Hexarray_w):
			if h == Hexarray_center[0] and w == Hexarray_center[1]:
				Hexarray_s_rotated[:, h, w, :] = Hexarray_s[:, h, w, :]
				continue

			coordinate = (Hexarray_center[0] - h, w - Hexarray_center[1])

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

			coordinate_hex_rotated = rotate_Hexint(coordinate_hex, angle)

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

			hwr = (Hexarray_h - 1 - (coordinate_rotated[0] + Hexarray_center[0]), coordinate_rotated[1] + Hexarray_center[1])

			if 0 <= hwr[0] < Hexarray_h and 0 <= hwr[1] < Hexarray_w:
				Hexarray_s_rotated[:, hwr[0], hwr[1], :] = Hexarray_s[:, h, w, :]


	return Hexarray_s_rotated


################################################################################
# Scale Hexarray via Hexarray to Hexarray transformation
################################################################################

def scale_Hexarray(Hexarray_s, res = (64, 64), method = 0, fill_value = 0):
	Hexarray_s = test_image_batch(Hexarray_s)

	Hexarray_s_transformed_shape = (Hexarray_s.shape[0],) + res + (Hexarray_s.shape[3],)
	Hexarray_s_transformed       = np.empty_like(Hexarray_s, shape=Hexarray_s_transformed_shape)

	Hexarray_s_transformed = Hexsamp_h2h(h1_s=Hexarray_s, h2_s=Hexarray_s_transformed, method=method)

	print(f'Hexarray_s_transformed=\n{np.reshape(Hexarray_s_transformed[0], newshape=res)}')

	if fill_value is not None:
		Hexarray_s_scaled       = np.full_like(Hexarray_s, fill_value)
		Hexarray_s_scaled_shape = Hexarray_s_scaled.shape

		Hexarray_s_scaled_h      = Hexarray_s_scaled_shape[1]
		Hexarray_s_scaled_w      = Hexarray_s_scaled_shape[2]
		Hexarray_s_scaled_center = (int(Hexarray_s_scaled_h / 2), int(Hexarray_s_scaled_w / 2))

		Hexarray_s_transformed_h      = Hexarray_s_transformed_shape[1]
		Hexarray_s_transformed_w      = Hexarray_s_transformed_shape[2]
		Hexarray_s_transformed_center = (int(Hexarray_s_transformed_h / 2), int(Hexarray_s_transformed_w / 2))

		Hexarray_s_scaled_slice_h_start = max(0, Hexarray_s_scaled_center[0] - int(Hexarray_s_transformed_h / 2))
		Hexarray_s_scaled_slice_w_start = max(0, Hexarray_s_scaled_center[1] - int(Hexarray_s_transformed_w / 2))
		Hexarray_s_scaled_slice_h_stop  = min(Hexarray_s_scaled_slice_h_start + Hexarray_s_transformed_shape[1], Hexarray_s_scaled_h)
		Hexarray_s_scaled_slice_w_stop  = min(Hexarray_s_scaled_slice_w_start + Hexarray_s_transformed_shape[2], Hexarray_s_scaled_w)
		Hexarray_s_scaled_slice_h       = slice(Hexarray_s_scaled_slice_h_start, Hexarray_s_scaled_slice_h_stop)
		Hexarray_s_scaled_slice_w       = slice(Hexarray_s_scaled_slice_w_start, Hexarray_s_scaled_slice_w_stop)
		Hexarray_s_scaled_slice         = (slice(None), Hexarray_s_scaled_slice_h, Hexarray_s_scaled_slice_w, slice(None))

		Hexarray_s_transformed_slice_h_start = max(0, Hexarray_s_transformed_center[0] - int(Hexarray_s_scaled_h / 2))
		Hexarray_s_transformed_slice_w_start = max(0, Hexarray_s_transformed_center[1] - int(Hexarray_s_scaled_w / 2))
		Hexarray_s_transformed_slice_h_stop  = min(Hexarray_s_transformed_slice_h_start + Hexarray_s_scaled_shape[1], Hexarray_s_transformed_h)
		Hexarray_s_transformed_slice_w_stop  = min(Hexarray_s_transformed_slice_w_start + Hexarray_s_scaled_shape[2], Hexarray_s_transformed_w)
		Hexarray_s_transformed_slice_h       = slice(Hexarray_s_transformed_slice_h_start, Hexarray_s_transformed_slice_h_stop)
		Hexarray_s_transformed_slice_w       = slice(Hexarray_s_transformed_slice_w_start, Hexarray_s_transformed_slice_w_stop)
		Hexarray_s_transformed_slice         = (slice(None), Hexarray_s_transformed_slice_h, Hexarray_s_transformed_slice_w, slice(None))

		Hexarray_s_scaled[Hexarray_s_scaled_slice] = Hexarray_s_transformed[Hexarray_s_transformed_slice]
	else:
		Hexarray_s_scaled = Hexarray_s_transformed

	return Hexarray_s_scaled


################################################################################
# Translate Hexarray by a given offset (e.g., (x, y))
################################################################################

def translate_Hexarray(Hexarray_s, translation = (1, 1), axis = (1, 2), fill_value = 0, cyclic_translation = False):
	Hexarray_s = test_image_batch(Hexarray_s)

	if any(current_translation for current_translation in translation):
		if not cyclic_translation:
			Hexarray_s_translated = np.full_like(Hexarray_s, fill_value)

			Hexarray_s_slice            = min(axis) * [slice(None)]
			Hexarray_s_translated_slice = min(axis) * [slice(None)]

			for current_translation in translation:
				if not current_translation:
					Hexarray_s_slice.append(slice(None))
					Hexarray_s_translated_slice.append(slice(None))
				elif current_translation < 0:
					Hexarray_s_slice.append(slice(-current_translation, None))
					Hexarray_s_translated_slice.append(slice(None, current_translation))
				else:
					Hexarray_s_slice.append(slice(None, -current_translation))
					Hexarray_s_translated_slice.append(slice(current_translation, None))

			Hexarray_s_slice            += ((Hexarray_s.ndim - 1) - max(axis)) * [slice(None)]
			Hexarray_s_translated_slice += ((Hexarray_s.ndim - 1) - max(axis)) * [slice(None)]

			Hexarray_s_slice            = tuple(Hexarray_s_slice)
			Hexarray_s_translated_slice = tuple(Hexarray_s_translated_slice)

			Hexarray_s_translated[Hexarray_s_translated_slice] = Hexarray_s[Hexarray_s_slice]
		else:
			Hexarray_s_translated = np.roll(Hexarray_s, shift=translation, axis=axis)
	else:
		Hexarray_s_translated = Hexarray_s

	return Hexarray_s_translated


################################################################################
# Rotate Hexarray test function
################################################################################

def test_rotate_Hexarray(Hexarray_shape = (3, 3, 1)):
	print(f'>> test_rotate_Hexarray: Hexarray_shape={Hexarray_shape}')

	test_Hexarray = np.empty(shape=Hexarray_shape, dtype=np.int32)

	for h in range(Hexarray_shape[0]):
		for w in range(Hexarray_shape[1]):
			test_Hexarray[h][w][0] = h * Hexarray_shape[1] + w + 1

	print(f'test_Hexarray =\n{np.reshape(test_Hexarray, newshape = (i, i))}')

	for angle in (60, 120, 180, 240, 300):
		print(f'> angle={angle}')

		test_Hexarray_rotated = rotate_Hexarray(Hexarray_s=test_Hexarray, angle=angle)

		print(f'test_Hexarray_rotated =\n{np.reshape(test_Hexarray_rotated[0], newshape = (i, i))}')


################################################################################
# Scale Hexarray test function
################################################################################

def test_scale_Hexarray(Hexarray_shape = (3, 3, 1)):
	print(f'>> test_scale_Hexarray: Hexarray_shape={Hexarray_shape}')

	test_Hexarray = np.empty(shape=Hexarray_shape, dtype=np.int32)

	for h in range(Hexarray_shape[0]):
		for w in range(Hexarray_shape[1]):
			test_Hexarray[h][w][0] = h * Hexarray_shape[1] + w + 1

	print(f'test_Hexarray =\n{np.reshape(test_Hexarray, newshape = (i, i))}')

	for res_factor in (2, 4, 6, 8):
		res = (res_factor, res_factor)
		print(f'> res_factor={res_factor}: res={res}')

		test_image_batchcaled = scale_Hexarray(Hexarray_s=test_Hexarray, res=res)

		print(f'test_image_batchcaled =\n{np.reshape(test_image_batchcaled[0], newshape = (i, i))}')


################################################################################
# Translate Hexarray test function
################################################################################

def test_translate_Hexarray(Hexarray_shape = (3, 3, 1)):
	print(f'>> test_translate_Hexarray: Hexarray_shape={Hexarray_shape}')

	test_Hexarray = np.empty(shape=Hexarray_shape, dtype=np.int32)

	for h in range(Hexarray_shape[0]):
		for w in range(Hexarray_shape[1]):
			test_Hexarray[h][w][0] = h * Hexarray_shape[1] + w + 1

	print(f'test_Hexarray =\n{np.reshape(test_Hexarray, newshape = (i, i))}')

	for translation_factor in (-2, -1, 0, 1, 2):
		translation = (translation_factor, translation_factor)
		axis        = (1, 2)
		print(f'> translation_factor={translation_factor}: translation={translation}, axis={axis}')

		test_Hexarray_translated = translate_Hexarray(Hexarray_s=test_Hexarray, translation=translation, axis=axis)

		print(f'test_Hexarray_translated =\n{np.reshape(test_Hexarray_translated[0], newshape = (i, i))}')


################################################################################
# main
################################################################################

if __name__ == '__main__':
	for i in (3, 5, 7):
		test_rotate_Hexarray(Hexarray_shape = (i, i, 1))

	for i in (3, 5, 7):
		test_scale_Hexarray(Hexarray_shape = (i, i, 1))

	for i in (3, 5, 7):
		test_translate_Hexarray(Hexarray_shape = (i, i, 1))

