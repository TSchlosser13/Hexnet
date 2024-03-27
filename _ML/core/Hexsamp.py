'''****************************************************************************
 * Hexsamp.py: Hexagonal Image Transformation
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

from misc.misc import test_image_batch


################################################################################
# Interpolation kernel function
################################################################################

def Hexsamp_kernel(x, y, method):
	if not method:
		if x < 1 and y < 1:
			k = (1 - x) * (1 - y)
		else:
			k = 0.0
	else:
		k = 1.0

	return k


################################################################################
# Hexarray to Hexarray transformation (TensorFlow version)
################################################################################

@tf.function
def Hexsamp_h2h(h1_s, h2_s, method):
	h1_s_is_tensor = tf.is_tensor(h1_s)
	h2_s_is_tensor = tf.is_tensor(h2_s)

	if h1_s_is_tensor or h2_s_is_tensor:
		if not h1_s_is_tensor:
			h1_s = tf.convert_to_tensor(h1_s, dtype=tf.float32)

		if not h2_s_is_tensor:
			h2_s = tf.convert_to_tensor(h2_s, dtype=tf.float32)

		s12_is_tensor = True
	else:
		h1_s = test_image_batch(h1_s)
		h2_s = test_image_batch(h2_s)

		s12_is_tensor = False


	h1_s_shape = h1_s.shape
	h2_s_shape = h2_s.shape

	hexarrays   = h1_s_shape[0]
	h1_s_height = h1_s_shape[1]
	h2_s_height = h2_s_shape[1]
	h1_s_width  = h1_s_shape[2]
	h2_s_width  = h2_s_shape[2]
	depth       = h1_s_shape[3]


	h1_s_rad_o  = 1.0
	h1_s_rad_i  = (math.sqrt(3) / 2) * h1_s_rad_o
	h1_s_dia_o  = 2 * h1_s_rad_o
	h1_s_dia_i  = 2 * h1_s_rad_i
	h1_s_dist_w = h1_s_dia_i
	h1_s_dist_h = 1.5 * h1_s_rad_o

	if h1_s_height > 1:
		h1_s_width_hex  = h1_s_width * h1_s_dia_i + h1_s_rad_i
		h1_s_height_hex = h1_s_dia_o + (h1_s_height - 1) * h1_s_dist_h
	else:
		h1_s_width_hex  = h1_s_width * h1_s_dia_i
		h1_s_height_hex = h1_s_dia_o


	if h2_s_height > 1:
		h2_s_rad_i_w = ((h1_s_width_hex - h1_s_dia_i) / (h2_s_width - 0.5)) / 2
		h2_s_rad_i_h = (((h1_s_height_hex - h1_s_rad_o) / (h2_s_height - 0.5)) / 2) / (math.sqrt(3) / 2)
	else:
		h2_s_rad_i_w = (h1_s_width_hex / h2_s_width) / 2
		h2_s_rad_i_h = ((h1_s_height_hex - h1_s_rad_o) / 2) / (math.sqrt(3) / 2)

	h2_s_rad_i = max(h2_s_rad_i_w, h2_s_rad_i_h)

	h2_s_rad_o  = h2_s_rad_i / (math.sqrt(3) / 2)
	h2_s_dia_o  = 2 * h2_s_rad_o
	h2_s_dia_i  = 2 * h2_s_rad_i
	h2_s_dist_w = h2_s_dia_i
	h2_s_dist_h = 1.5 * h2_s_rad_o

	if h2_s_height > 1:
		h2_s_width_hex  = h2_s_width * h2_s_dia_i + h2_s_rad_i
		h2_s_height_hex = h2_s_dia_o + (h2_s_height - 1) * h2_s_dist_h
	else:
		h2_s_width_hex  = h2_s_width * h2_s_dia_i
		h2_s_height_hex = h2_s_dia_o


	if s12_is_tensor:
		h2_s_list   = h2_s_height * [h2_s_width * [None]]
		h2_s_list_h = []




	r  = max(h1_s_dia_i, h2_s_dia_i)
	ri = math.ceil(r)
	wb = ((h1_s_width_hex  - h1_s_dia_i) - (h2_s_width_hex  - h2_s_dia_i)) / 2
	hb = ((h1_s_height_hex - h1_s_dia_o) - (h2_s_height_hex - h2_s_dia_o)) / 2

	for h in range(h2_s_height):
		ht   = hb + h * h2_s_dist_h
		hth  = ht / h1_s_dist_h
		hthi = round(hth)

		for w in range(h2_s_width):
			wt   = wb + w * h2_s_dia_i if not (h    % 2) else wb + h2_s_rad_i + w * h2_s_dia_i
			wth  = wt / h1_s_dia_i     if not (hthi % 2) else -h1_s_rad_i + wt / h1_s_dia_i
			wthi = round(wth)

			if not s12_is_tensor:
				o = np.zeros(shape = (hexarrays, depth))
			else:
				o = tf.zeros(shape = (tf.shape(h2_s)[0], depth))

			on = 0.0


			for y in range(hthi - ri, hthi + ri + 1):
				hh = y * h1_s_dist_h

				for x in range(wthi - ri, wthi + ri + 1):
					wh = x * h1_s_dia_i if not (y % 2) else h1_s_rad_i + x * h1_s_dia_i
					rx = abs(wt - wh)
					ry = abs(ht - hh)

					if x >= 0 and x < h1_s_width and y >= 0 and y < h1_s_height and rx < r and ry < r:
						k = Hexsamp_kernel(rx / r, ry / r, method)

						o  += k * h1_s[:, y, x, :]
						on += k

			if on:
				o /= on


			if not s12_is_tensor:
				h2_s[:, h, w, :] = np.round(np.minimum(o, 255))
			else:
				h2_s_list[h][w] = o


		if s12_is_tensor:
			h2_s_list_h.append(
				tf.stack(
					values = h2_s_list[h],
					axis   = 1,
					name   = 'Hexsamp_h2h_h2_s_list_h_stack'))

	if s12_is_tensor:
		h2_s = tf.stack(
			values = h2_s_list_h,
			axis   = 1,
			name   = 'Hexsamp_h2h_h2_s_stack')


	return h2_s

