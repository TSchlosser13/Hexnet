'''****************************************************************************
 * layers.py: TODO
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


import math

import numpy      as np
import tensorflow as tf

from scipy.optimize import linear_sum_assignment
from time           import time

from layers.masks   import build_masks
from layers.offsets import build_offsets
from misc.misc      import Hexnet_print


_ENABLE_DEBUGGING = False


class SConv2D(tf.keras.layers.Layer):
	def __init__(
		self,
		filters              = 1,
		kernel_size          = (3, 3),
		strides              = (1, 1),
		padding              = 'SAME',
		data_format          = 'NHWC',
		dilation_rate        = (1, 1),
		activation           = tf.nn.relu,
		use_bias             = True,
		kernel_initializer   = tf.initializers.glorot_uniform,
		bias_initializer     = tf.initializers.zeros,
		kernel_regularizer   = None,
		bias_regularizer     = None,
		activity_regularizer = None,
		kernel_constraint    = None,
		bias_constraint      = None,
		**kwargs):

		super().__init__(**kwargs)

		self.filters              = filters
		self.kernel_size          = kernel_size
		self.strides              = strides
		self.padding              = padding
		self.data_format          = data_format
		self.dilation_rate        = dilation_rate
		self.activation           = activation
		self.use_bias             = use_bias
		self.kernel_initializer   = kernel_initializer
		self.bias_initializer     = bias_initializer
		self.kernel_regularizer   = kernel_regularizer
		self.bias_regularizer     = bias_regularizer
		self.activity_regularizer = activity_regularizer
		self.kernel_constraint    = kernel_constraint
		self.bias_constraint      = bias_constraint

	def build(self, input_shape):
		super().build(input_shape)

		kernel_shape = (self.kernel_size[0], self.kernel_size[1], input_shape[3], self.filters) # WHIOC
		bias_shape   = self.filters

		self.kernel = self.add_variable(
			name        = 'SConv2D_kernel_add_variable',
			shape       = kernel_shape,
			initializer = self.kernel_initializer,
			trainable   = True)

		self.bias = self.add_variable(
			name        = 'SConv2D_bias_add_variable',
			shape       = bias_shape,
			initializer = self.bias_initializer,
			trainable   = True)

	def call(self, input):
		output = tf.nn.conv2d(
			input       = input,
			filters     = self.kernel,
			strides     = self.strides,
			padding     = self.padding,
			data_format = self.data_format,
			dilations   = self.dilation_rate,
			name        = 'SConv2D_output_conv2d')

		output = tf.nn.bias_add(
			value       = output,
			bias        = self.bias,
			data_format = self.data_format,
			name        = 'SConv2D_output_bias_add')

		output = self.activation(
			features = output,
			name     = 'SConv2D_output_activation')

		return output


class SMaxPool2D(tf.keras.layers.Layer):
	def __init__(
		self,
		pool_size   = (3, 3),
		strides     = None,
		padding     = 'SAME',
		data_format = 'NHWC',
		**kwargs):

		super().__init__(**kwargs)

		self.pool_size = pool_size

		if strides is None:
			self.strides = pool_size
		else:
			self.strides = strides

		self.padding     = padding
		self.data_format = data_format

	def build(self, input_shape):
		super().build(input_shape)

	def call(self, input):
		output = tf.nn.max_pool2d(
			input       = input,
			ksize       = self.pool_size,
			strides     = self.strides,
			padding     = self.padding,
			data_format = self.data_format,
			name        = 'SMaxPool2D_output_max_pool2d')

		return output


class HConv2D(tf.keras.layers.Layer):
	def __init__(
		self,
		filters              = 1,
		kernel_size          = (3, 3),
		strides              = (1, 1),
		padding              = 'SAME',
		data_format          = 'NHWC',
		dilation_rate        = (1, 1),
		activation           = tf.nn.relu,
		use_bias             = True,
		kernel_initializer   = tf.initializers.glorot_uniform,
		bias_initializer     = tf.initializers.zeros,
		kernel_regularizer   = None,
		bias_regularizer     = None,
		activity_regularizer = None,
		kernel_constraint    = None,
		bias_constraint      = None,
		**kwargs):

		super().__init__(**kwargs)

		self.filters              = filters
		self.kernel_size          = kernel_size
		self.strides              = strides
		self.padding              = padding
		self.data_format          = data_format
		self.dilation_rate        = dilation_rate
		self.activation           = activation
		self.use_bias             = use_bias
		self.kernel_initializer   = kernel_initializer
		self.bias_initializer     = bias_initializer
		self.kernel_regularizer   = kernel_regularizer
		self.bias_regularizer     = bias_regularizer
		self.activity_regularizer = activity_regularizer
		self.kernel_constraint    = kernel_constraint
		self.bias_constraint      = bias_constraint

	def build_masks(self, mask_size=(3, 3)):
		return build_masks(mask_size)

	def build(self, input_shape):
		super().build(input_shape)


		kernel_shape = (self.kernel_size[0], self.kernel_size[1], input_shape[3], self.filters) # WHIOC
		bias_shape   = self.filters

		self.kernel = self.add_variable(
			name        = 'SConv2D_kernel_add_variable',
			shape       = kernel_shape,
			initializer = self.kernel_initializer,
			trainable   = True)

		self.bias = self.add_variable(
			name        = 'SConv2D_bias_add_variable',
			shape       = bias_shape,
			initializer = self.bias_initializer,
			trainable   = True)

		(kernel_mask_even_rows, _) = self.build_masks(mask_size=self.kernel_size)

		self.kernel_mask_even_rows = tf.convert_to_tensor(
			value = kernel_mask_even_rows,
			dtype = tf.float32,
			name  = 'HConv2D_kernel_mask_even_rows_convert_to_tensor')


		if type(self.strides) is not tuple:
			self.strides = (2 * self.strides, self.strides)
		else:
			self.strides = (2 * self.strides[0], self.strides[1])

	def call(self, input):
		kernel_masked_even_rows = tf.einsum('ijkl,ij->ijkl', self.kernel, self.kernel_mask_even_rows)

		if input.shape[1] > 1:
			kernel_masked_odd_rows = []

			kernel_center = int(kernel_masked_even_rows.shape[0] / 2)

			if kernel_center % 2:
				kernel_shift = -1
			else:
				kernel_shift = 1

			for kernel_row in range(kernel_masked_even_rows.shape[0]):
				if abs(kernel_center - kernel_row) % 2:
					kernel_masked_odd_rows.append(
						tf.roll(
							input = kernel_masked_even_rows[kernel_row],
							shift = kernel_shift,
							axis  = 0,
							name  = 'HConv2D_kernel_masked_odd_rows_roll'))

			kernel_masked_odd_rows = tf.stack(
				values = kernel_masked_odd_rows,
				axis   = 0,
				name   = 'HConv2D_kernel_masked_odd_rows_stack')


		output_even_rows = tf.nn.conv2d(
			input       = input,
			filters     = kernel_masked_even_rows,
			strides     = self.strides,
			padding     = self.padding,
			data_format = self.data_format,
			dilations   = self.dilation_rate,
			name        = 'HConv2D_output_even_rows_conv2d')

		if input.shape[1] > 1:
			output_odd_rows = tf.nn.conv2d(
				input       = input[:, 1:, :, :],
				filters     = kernel_masked_odd_rows,
				strides     = self.strides,
				padding     = self.padding,
				data_format = self.data_format,
				dilations   = self.dilation_rate,
				name        = 'HConv2D_output_odd_rows_conv2d')


		if input.shape[1] > 1:
			output = tf.concat(
				values = (output_even_rows[:, 0:1, :, :], output_odd_rows[:, 0:1, :, :]),
				axis   = 1,
				name   = 'HConv2D_output_concat')

			for h in range(1, min(output_even_rows.shape[1], output_odd_rows.shape[1])):
				output = tf.concat(
					values = (output, output_even_rows[:, h:h+1, :, :], output_odd_rows[:, h:h+1, :, :]),
					axis   = 1,
					name   = 'HConv2D_output_concat')

			if output_even_rows.shape[1] > output_odd_rows.shape[1]:
				output = tf.concat(
					values = (output, output_even_rows[:, -1:, :, :]),
					axis   = 1,
					name   = 'HConv2D_output_concat')
		else:
			output = output_even_rows

		output = tf.nn.bias_add(
			value       = output,
			bias        = self.bias,
			data_format = self.data_format,
			name        = 'HConv2D_output_bias_add')

		output = self.activation(
			features = output,
			name     = 'HConv2D_output_activation')

		return output


class HMaxPool2D(tf.keras.layers.Layer):
	def __init__(
		self,
		pool_size   = (3, 3),
		strides     = None,
		padding     = 'SAME',
		data_format = 'NHWC',
		**kwargs):

		super().__init__(**kwargs)

		self.pool_size = pool_size

		if strides is None:
			self.strides = pool_size
		else:
			self.strides = strides

		self.padding     = padding
		self.data_format = data_format

	def build_masks(self, mask_size=(3, 3)):
		return build_masks(mask_size)

	def build_offsets(self, mask_size=(3, 3)):
		return build_offsets(mask_size)

	def build(self, input_shape):
		super().build(input_shape)


		self.pool_size2 = (int((self.pool_size[0] - 1) / 2), int((self.pool_size[1] - 1) / 2))

		(kernel_mask_even_rows, kernel_mask_odd_rows)                 = self.build_masks(mask_size=self.pool_size)
		(self.kernel_offsets_even_rows, self.kernel_offsets_odd_rows) = self.build_offsets(mask_size=self.pool_size)

		self.kernel_mask_even_rows = tf.convert_to_tensor(
			value = kernel_mask_even_rows,
			dtype = tf.float32,
			name  = 'HMaxPool2D_kernel_mask_even_rows_convert_to_tensor')

		self.kernel_mask_odd_rows = tf.convert_to_tensor(
			value = kernel_mask_odd_rows,
			dtype = tf.float32,
			name  = 'HMaxPool2D_kernel_mask_odd_rows_convert_to_tensor')




		input_offsets         = []
		input_offsets_base    = [0, 0]
		input_offsets_current = [0, 0]
		input_offsets_min     = (0, 0)
		input_offsets_max     = (0, 0)

		input_shape_h = input_shape[1]
		input_shape_w = input_shape[2]

		input_offsets_shape_min = (-self.pool_size2[0], -self.pool_size2[1])
		input_offsets_shape_max = (input_shape_h - 1 + self.pool_size2[0], input_shape_w - 1 + self.pool_size2[1])

		input_offsets_shape = (input_offsets_shape_max[0] - input_offsets_shape_min[0] + 1, input_offsets_shape_max[1] - input_offsets_shape_min[1] + 1)


		for h in range(input_shape_h):
			for w in range(input_shape_w):
				if input_offsets_current[0] >= input_offsets_shape_min[0] and \
				   input_offsets_current[0] <= input_offsets_shape_max[0] and \
				   input_offsets_current[1] >= input_offsets_shape_min[1] and \
				   input_offsets_current[1] <= input_offsets_shape_max[1]:

					input_offsets.append((input_offsets_current[0], input_offsets_current[1]))
					input_offsets_min = (min(input_offsets_min[0], input_offsets_current[0]), min(input_offsets_min[1], input_offsets_current[1]))
					input_offsets_max = (max(input_offsets_max[0], input_offsets_current[0]), max(input_offsets_max[1], input_offsets_current[1]))

				if not input_offsets_current[0] % 2:
					kernel_offsets = self.kernel_offsets_even_rows
				else:
					kernel_offsets = self.kernel_offsets_odd_rows

				input_offsets_current[0] += kernel_offsets[0][1]
				input_offsets_current[1] += kernel_offsets[0][0]

			if not input_offsets_base[0] % 2:
				kernel_offsets = self.kernel_offsets_even_rows
			else:
				kernel_offsets = self.kernel_offsets_odd_rows

			input_offsets_base[0] += kernel_offsets[1][1]
			input_offsets_base[1] += kernel_offsets[1][0]
			input_offsets_current  = input_offsets_base.copy()


		input_offsets_padding_h = (-input_offsets_min[0] + self.pool_size2[0], input_offsets_max[0] - (input_shape_h - 1) + self.pool_size2[0])
		input_offsets_padding_w = (-input_offsets_min[1] + self.pool_size2[1], input_offsets_max[1] - (input_shape_w - 1) + self.pool_size2[1])
		self.paddings           = ((0, 0), input_offsets_padding_h, input_offsets_padding_w, (0, 0))

		input_offsets_padded   = [(offset[0] + input_offsets_padding_h[0], offset[1] + input_offsets_padding_w[0]) for offset in input_offsets]
		input_offsets_center   = (sum(input_offsets_padded[:][0]) / len(input_offsets_padded[:][0]), sum(input_offsets_padded[:][1]) / len(input_offsets_padded[:][1]))
		input_offsets_centered = [(offset[0] - input_offsets_center[0], offset[1] - input_offsets_center[1]) for offset in input_offsets_padded]

		output_shape_base = math.sqrt(len(input_offsets_padded))
		output_shape_h    = math.floor((input_shape_h / input_shape_w) * output_shape_base)
		output_shape_w    = math.floor((input_shape_w / input_shape_h) * output_shape_base)
		output_shape      = (input_shape[0], output_shape_h, output_shape_w, input_shape[3])

		output_offsets          = [(h, w) for w in range(output_shape_w) for h in range(output_shape_h)]
		output_offsets_scale    = (input_offsets_shape[0] / output_shape_h, input_offsets_shape[1] / output_shape_w)
		output_offsets_scaled   = [(output_offsets_scale[0] * offset[0], output_offsets_scale[1] * offset[1]) for offset in output_offsets]
		output_offsets_center   = (sum(output_offsets_scaled[:][0]) / len(output_offsets_scaled[:][0]), sum(output_offsets_scaled[:][1]) / len(output_offsets_scaled[:][1]))
		output_offsets_centered = [(offset[0] - output_offsets_center[0], offset[1] - output_offsets_center[1]) for offset in output_offsets_scaled]




		cost_matrix = np.full(shape=(len(input_offsets_centered), len(output_offsets_centered)), fill_value=np.inf)

		for h in range(cost_matrix.shape[0]):
			for w in range(cost_matrix.shape[1]):
				cost_matrix[h][w] = np.linalg.norm(np.array(input_offsets_centered[h]) - np.array(output_offsets_centered[w]))

		start_time = time()
		row_indices, col_indices = linear_sum_assignment(cost_matrix)
		time_diff = time() - start_time

		Hexnet_print(f'(HMaxPool2D) Initialized pooling layer in {time_diff:.3f} seconds ({input_shape}->{output_shape})')

		if _ENABLE_DEBUGGING:
			costs     = cost_matrix[row_indices][col_indices]
			costs_sum = costs.sum()

			Hexnet_print('(linear_sum_assignment) '
				f'row_indices =\n{row_indices},\n'
				f'col_indices =\n{col_indices},\n'
				f'costs =\n{costs},\n'
				f'costs_sum={costs_sum}')


		self.pooling_offsets = np.empty(shape=(output_shape_h, output_shape_w, 2), dtype=np.int32)

		for index_counter, col_index in enumerate(col_indices):
			pooling_offsets_index      = (int(col_index / output_shape_w), col_index % output_shape_w)
			input_offsets_padded_index = row_indices[index_counter]

			self.pooling_offsets[pooling_offsets_index] = input_offsets_padded[input_offsets_padded_index]

		if _ENABLE_DEBUGGING:
			Hexnet_print(f'pooling_offsets =\n{self.pooling_offsets}')

	def call(self, input):
		input = tf.pad(
			tensor          = input,
			paddings        = self.paddings,
			mode            = 'CONSTANT',
			constant_values = 0,
			name            = 'HMaxPool2D_input_pad')


		input_pooling_list   = self.pooling_offsets.shape[0] * [self.pooling_offsets.shape[1] * [None]]
		input_pooling_list_h = []

		for h in range(self.pooling_offsets.shape[0]):
			for w in range(self.pooling_offsets.shape[1]):
				if self.pooling_offsets[h, w, 0] % 2:
					kernel_mask = self.kernel_mask_odd_rows
				else:
					kernel_mask = self.kernel_mask_even_rows

				h_slice_to_mask          = slice(self.pooling_offsets[h, w, 0] - self.pool_size2[0], self.pooling_offsets[h, w, 0] + self.pool_size2[0] + 1)
				w_slice_to_mask          = slice(self.pooling_offsets[h, w, 1] - self.pool_size2[1], self.pooling_offsets[h, w, 1] + self.pool_size2[1] + 1)
				input_pooling_list[h][w] = tf.einsum('ijkl,jk->ijkl', input[:, h_slice_to_mask, w_slice_to_mask, :], kernel_mask)

			input_pooling_list_h.append(
				tf.concat(
					values = input_pooling_list[h],
					axis   = 2,
					name   = 'HMaxPool2D_input_pooling_list_h_concat'))

		input_pooling_list = tf.concat(
			values = input_pooling_list_h,
			axis   = 1,
			name   = 'HMaxPool2D_input_pooling_list_concat')


		output = tf.nn.max_pool2d(
			input       = input_pooling_list,
			ksize       = self.pool_size,
			strides     = self.strides,
			padding     = 'VALID',
			data_format = self.data_format,
			name        = 'HMaxPool2D_output_max_pool2d')

		return output


