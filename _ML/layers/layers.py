'''****************************************************************************
 * layers.py: Square and Hexagonal Layers for Use with Keras
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

from scipy.optimize import linear_sum_assignment

from core.Hexsamp   import Hexsamp_h2h
from layers.kernels import rotate_hexagonal_kernels, rotate_square_kernel
from layers.masks   import build_masks
from layers.offsets import build_offsets
from misc.misc      import Hexnet_print


################################################################################
# Global variables
################################################################################

_ENABLE_DEBUGGING = False








################################################################################
# Layer classes: convolutional, pooling, and sampling layers
################################################################################




class SConv2D(tf.keras.layers.Layer):
	"""Square 2D convolution layer (e.g. spatial convolution over images).

	This layer creates a convolution kernel that is convolved
	with the layer input to produce a tensor of
	outputs. If `use_bias` is True,
	a bias vector is created and added to the outputs. Finally, if
	`activation` is not `None`, it is applied to the outputs as well.

	When using this layer as the first layer in a model,
	provide the keyword argument `input_shape`
	(tuple of integers, does not include the sample axis),
	e.g. `input_shape=(128, 128, 3)` for 128x128 RGB pictures
	in `data_format="channels_last"`.

	Arguments:
		filters: Integer, the dimensionality of the output space
			(i.e. the number of output filters in the convolution).
		kernel_size: An integer or tuple/list of 2 integers, specifying the
			height and width of the 2D convolution window.
			Can be a single integer to specify the same value for
			all spatial dimensions.
		strides: An integer or tuple/list of 2 integers,
			specifying the strides of the convolution along the height and width.
			Can be a single integer to specify the same value for
			all spatial dimensions.
			Specifying any stride value != 1 is incompatible with specifying
			any `dilation_rate` value != 1.
		padding: one of `"valid"` or `"same"` (case-insensitive).
		data_format: A string,
			one of `channels_last` (default) or `channels_first`.
			The ordering of the dimensions in the inputs.
			`channels_last` corresponds to inputs with shape
			`(batch, height, width, channels)` while `channels_first`
			corresponds to inputs with shape
			`(batch, channels, height, width)`.
			It defaults to the `image_data_format` value found in your
			Keras config file at `~/.keras/keras.json`.
			If you never set it, then it will be "channels_last".
		dilation_rate: an integer or tuple/list of 2 integers, specifying
			the dilation rate to use for dilated convolution.
			Can be a single integer to specify the same value for
			all spatial dimensions.
			Currently, specifying any `dilation_rate` value != 1 is
			incompatible with specifying any stride value != 1.
		activation: Activation function to use.
			If you don't specify anything, no activation is applied
			(ie. "linear" activation: `a(x) = x`).
		use_bias: Boolean, whether the layer uses a bias vector.
		kernel_initializer: Initializer for the `kernel` weights matrix.
		bias_initializer: Initializer for the bias vector.
		kernel_regularizer: Regularizer function applied to
			the `kernel` weights matrix.
		bias_regularizer: Regularizer function applied to the bias vector.
		activity_regularizer: Regularizer function applied to
			the output of the layer (its "activation")..
		kernel_constraint: Constraint function applied to the kernel matrix.
		bias_constraint: Constraint function applied to the bias vector.

	Input shape:
		4D tensor with shape:
		`(samples, channels, rows, cols)` if data_format='channels_first'
		or 4D tensor with shape:
		`(samples, rows, cols, channels)` if data_format='channels_last'.

	Output shape:
		4D tensor with shape:
		`(samples, filters, new_rows, new_cols)` if data_format='channels_first'
		or 4D tensor with shape:
		`(samples, new_rows, new_cols, filters)` if data_format='channels_last'.
		`rows` and `cols` values might have changed due to padding.
	"""

	def __init__(
		self,
		filters              = 1,
		kernel_size          = (3, 3),
		strides              = (1, 1),
		padding              = 'SAME',
		data_format          = 'NHWC',
		dilation_rate        = (1, 1),
		activation           = None,
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

		self.filters = filters

		if type(kernel_size) is int:
			self.kernel_size = (kernel_size, kernel_size)
		elif type(kernel_size) is not tuple:
			self.kernel_size = tuple(kernel_size)

			if len(kernel_size) == 1:
				self.kernel_size *= 2
		else:
			self.kernel_size = kernel_size

		self.strides       = strides
		self.padding       = padding
		self.data_format   = data_format
		self.dilation_rate = dilation_rate

		if type(activation) is str:
			self.activation = tf.keras.activations(activation)
		else:
			self.activation = activation

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

		kernel_shape = (self.kernel_size[0], self.kernel_size[1], input_shape[3], self.filters) # HWIOC
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

		if self.activation is not None:
			output = self.activation(
				features = output,
				name     = 'SConv2D_output_activation')

		return output


class SConv2DTranspose(SConv2D):
	"""Square 2D transposed convolution layer (e.g. spatial convolution over images, sometimes called Deconvolution).

	This layer creates a convolution kernel that is convolved
	with the layer input to produce a tensor of
	outputs. If `use_bias` is True,
	a bias vector is created and added to the outputs. Finally, if
	`activation` is not `None`, it is applied to the outputs as well.

	When using this layer as the first layer in a model,
	provide the keyword argument `input_shape`
	(tuple of integers, does not include the sample axis),
	e.g. `input_shape=(128, 128, 3)` for 128x128 RGB pictures
	in `data_format="channels_last"`.

	Arguments:
		filters: Integer, the dimensionality of the output space
			(i.e. the number of output filters in the convolution).
		kernel_size: An integer or tuple/list of 2 integers, specifying the
			height and width of the 2D convolution window.
			Can be a single integer to specify the same value for
			all spatial dimensions.
		strides: An integer or tuple/list of 2 integers,
			specifying the strides of the convolution along the height and width.
			Can be a single integer to specify the same value for
			all spatial dimensions.
			Specifying any stride value != 1 is incompatible with specifying
			any `dilation_rate` value != 1.
		padding: one of `"valid"` or `"same"` (case-insensitive).
		data_format: A string,
			one of `channels_last` (default) or `channels_first`.
			The ordering of the dimensions in the inputs.
			`channels_last` corresponds to inputs with shape
			`(batch, height, width, channels)` while `channels_first`
			corresponds to inputs with shape
			`(batch, channels, height, width)`.
			It defaults to the `image_data_format` value found in your
			Keras config file at `~/.keras/keras.json`.
			If you never set it, then it will be "channels_last".
		dilation_rate: an integer or tuple/list of 2 integers, specifying
			the dilation rate to use for dilated convolution.
			Can be a single integer to specify the same value for
			all spatial dimensions.
			Currently, specifying any `dilation_rate` value != 1 is
			incompatible with specifying any stride value != 1.
		activation: Activation function to use.
			If you don't specify anything, no activation is applied
			(ie. "linear" activation: `a(x) = x`).
		use_bias: Boolean, whether the layer uses a bias vector.
		kernel_initializer: Initializer for the `kernel` weights matrix.
		bias_initializer: Initializer for the bias vector.
		kernel_regularizer: Regularizer function applied to
			the `kernel` weights matrix.
		bias_regularizer: Regularizer function applied to the bias vector.
		activity_regularizer: Regularizer function applied to
			the output of the layer (its "activation")..
		kernel_constraint: Constraint function applied to the kernel matrix.
		bias_constraint: Constraint function applied to the bias vector.

	Input shape:
		4D tensor with shape:
		`(samples, channels, rows, cols)` if data_format='channels_first'
		or 4D tensor with shape:
		`(samples, rows, cols, channels)` if data_format='channels_last'.

	Output shape:
		4D tensor with shape:
		`(samples, filters, new_rows, new_cols)` if data_format='channels_first'
		or 4D tensor with shape:
		`(samples, new_rows, new_cols, filters)` if data_format='channels_last'.
		`rows` and `cols` values might have changed due to padding.
	"""

	def build(self, input_shape):
		super().build(input_shape)

		kernel_shape = (self.kernel_size[0], self.kernel_size[1], self.filters, input_shape[3]) # HWOIC
		bias_shape   = self.filters

		self.kernel = self.add_variable(
			name        = 'SConv2DTranspose_kernel_add_variable',
			shape       = kernel_shape,
			initializer = self.kernel_initializer,
			trainable   = True)

		self.bias = self.add_variable(
			name        = 'SConv2DTranspose_bias_add_variable',
			shape       = bias_shape,
			initializer = self.bias_initializer,
			trainable   = True)

	def call(self, input):
		if type(self.strides) is int:
			output_shape = (tf.shape(input)[0], self.strides * input.shape[1], self.strides * input.shape[2], self.filters)
		else:
			output_shape = (tf.shape(input)[0], self.strides[0] * input.shape[1], self.strides[1] * input.shape[2], self.filters)

		output = tf.nn.conv2d_transpose(
			input        = input,
			filters      = self.kernel,
			output_shape = output_shape,
			strides      = self.strides,
			padding      = self.padding,
			data_format  = self.data_format,
			dilations    = self.dilation_rate,
			name         = 'SConv2DTranspose_output_conv2d_transpose')

		output = tf.nn.bias_add(
			value       = output,
			bias        = self.bias,
			data_format = self.data_format,
			name        = 'SConv2DTranspose_output_bias_add')

		if self.activation is not None:
			output = self.activation(
				features = output,
				name     = 'SConv2DTranspose_output_activation')

		return output


class SGConv2D(SConv2D):
	"""Square 2D group convolution layer (e.g. spatial convolution over images).

	This layer creates a convolution kernel that is convolved
	with the layer input to produce a tensor of
	outputs. If `use_bias` is True,
	a bias vector is created and added to the outputs. Finally, if
	`activation` is not `None`, it is applied to the outputs as well.

	When using this layer as the first layer in a model,
	provide the keyword argument `input_shape`
	(tuple of integers, does not include the sample axis),
	e.g. `input_shape=(128, 128, 3)` for 128x128 RGB pictures
	in `data_format="channels_last"`.

	Arguments:
		filters: Integer, the dimensionality of the output space
			(i.e. the number of output filters in the convolution).
		kernel_size: An integer or tuple/list of 2 integers, specifying the
			height and width of the 2D convolution window.
			Can be a single integer to specify the same value for
			all spatial dimensions.
		strides: An integer or tuple/list of 2 integers,
			specifying the strides of the convolution along the height and width.
			Can be a single integer to specify the same value for
			all spatial dimensions.
			Specifying any stride value != 1 is incompatible with specifying
			any `dilation_rate` value != 1.
		padding: one of `"valid"` or `"same"` (case-insensitive).
		data_format: A string,
			one of `channels_last` (default) or `channels_first`.
			The ordering of the dimensions in the inputs.
			`channels_last` corresponds to inputs with shape
			`(batch, height, width, channels)` while `channels_first`
			corresponds to inputs with shape
			`(batch, channels, height, width)`.
			It defaults to the `image_data_format` value found in your
			Keras config file at `~/.keras/keras.json`.
			If you never set it, then it will be "channels_last".
		dilation_rate: an integer or tuple/list of 2 integers, specifying
			the dilation rate to use for dilated convolution.
			Can be a single integer to specify the same value for
			all spatial dimensions.
			Currently, specifying any `dilation_rate` value != 1 is
			incompatible with specifying any stride value != 1.
		activation: Activation function to use.
			If you don't specify anything, no activation is applied
			(ie. "linear" activation: `a(x) = x`).
		use_bias: Boolean, whether the layer uses a bias vector.
		kernel_initializer: Initializer for the `kernel` weights matrix.
		bias_initializer: Initializer for the bias vector.
		kernel_regularizer: Regularizer function applied to
			the `kernel` weights matrix.
		bias_regularizer: Regularizer function applied to the bias vector.
		activity_regularizer: Regularizer function applied to
			the output of the layer (its "activation")..
		kernel_constraint: Constraint function applied to the kernel matrix.
		bias_constraint: Constraint function applied to the bias vector.

	Input shape:
		4D tensor with shape:
		`(samples, channels, rows, cols)` if data_format='channels_first'
		or 4D tensor with shape:
		`(samples, rows, cols, channels)` if data_format='channels_last'.

	Output shape:
		4D tensor with shape:
		`(samples, filters, new_rows, new_cols)` if data_format='channels_first'
		or 4D tensor with shape:
		`(samples, new_rows, new_cols, filters)` if data_format='channels_last'.
		`rows` and `cols` values might have changed due to padding.
	"""

	def call(self, input):
		kernel_rotated = []
		group_size     = math.ceil(self.kernel.shape[3] / 4)

		for k in range(4):
			output_channels_index_min = k * group_size

			if k < 3:
				output_channels_index_max = (k + 1) * group_size
			else:
				output_channels_index_max = self.kernel.shape[3]

			kernel_rotated.append(rotate_square_kernel(kernel=self.kernel[:, :, :, output_channels_index_min:output_channels_index_max], angle = k * 90))

		kernel_rotated = tf.concat(
			values = kernel_rotated,
			axis   = 3,
			name   = 'SGConv2D_kernel_rotated_concat')


		output = tf.nn.conv2d(
			input       = input,
			filters     = kernel_rotated,
			strides     = self.strides,
			padding     = self.padding,
			data_format = self.data_format,
			dilations   = self.dilation_rate,
			name        = 'SGConv2D_output_conv2d')

		output = tf.nn.bias_add(
			value       = output,
			bias        = self.bias,
			data_format = self.data_format,
			name        = 'SGConv2D_output_bias_add')

		if self.activation is not None:
			output = self.activation(
				features = output,
				name     = 'SGConv2D_output_activation')

		return output




class SPool2D(tf.keras.layers.Layer):
	"""Square pooling operation for spatial data.

	Arguments:
		pool_size: integer or tuple of 2 integers,
			factors by which to downscale (vertical, horizontal).
			`(2, 2)` will halve the input in both spatial dimension.
			If only one integer is specified, the same window length
			will be used for both dimensions.
		strides: Integer, tuple of 2 integers, or None.
			Strides values.
			If None, it will default to `pool_size`.
		padding: One of `"valid"` or `"same"` (case-insensitive).
		data_format: A string,
			one of `channels_last` (default) or `channels_first`.
			The ordering of the dimensions in the inputs.
			`channels_last` corresponds to inputs with shape
			`(batch, height, width, channels)` while `channels_first`
			corresponds to inputs with shape
			`(batch, channels, height, width)`.
			It defaults to the `image_data_format` value found in your
			Keras config file at `~/.keras/keras.json`.
			If you never set it, then it will be "channels_last".

	Input shape:
		- If `data_format='channels_last'`:
			4D tensor with shape `(batch_size, rows, cols, channels)`.
		- If `data_format='channels_first'`:
			4D tensor with shape `(batch_size, channels, rows, cols)`.

	Output shape:
		- If `data_format='channels_last'`:
			4D tensor with shape `(batch_size, pooled_rows, pooled_cols, channels)`.
		- If `data_format='channels_first'`:
			4D tensor with shape `(batch_size, channels, pooled_rows, pooled_cols)`.
	"""

	def __init__(
		self,
		pool_size   = (3, 3),
		strides     = None,
		padding     = 'SAME',
		data_format = 'NHWC',
		**kwargs):

		super().__init__(**kwargs)

		if type(pool_size) is int:
			self.pool_size = (pool_size, pool_size)
		elif type(pool_size) is not tuple:
			self.pool_size = tuple(pool_size)

			if len(pool_size) == 1:
				self.pool_size *= 2
		else:
			self.pool_size = pool_size

		if strides is None:
			self.strides = pool_size
		else:
			self.strides = strides

		self.padding     = padding
		self.data_format = data_format

	def build(self, input_shape):
		super().build(input_shape)


class SAvgPool2D(SPool2D):
	"""Square average pooling operation for spatial data.

	Arguments:
		pool_size: integer or tuple of 2 integers,
			factors by which to downscale (vertical, horizontal).
			`(2, 2)` will halve the input in both spatial dimension.
			If only one integer is specified, the same window length
			will be used for both dimensions.
		strides: Integer, tuple of 2 integers, or None.
			Strides values.
			If None, it will default to `pool_size`.
		padding: One of `"valid"` or `"same"` (case-insensitive).
		data_format: A string,
			one of `channels_last` (default) or `channels_first`.
			The ordering of the dimensions in the inputs.
			`channels_last` corresponds to inputs with shape
			`(batch, height, width, channels)` while `channels_first`
			corresponds to inputs with shape
			`(batch, channels, height, width)`.
			It defaults to the `image_data_format` value found in your
			Keras config file at `~/.keras/keras.json`.
			If you never set it, then it will be "channels_last".

	Input shape:
		- If `data_format='channels_last'`:
			4D tensor with shape `(batch_size, rows, cols, channels)`.
		- If `data_format='channels_first'`:
			4D tensor with shape `(batch_size, channels, rows, cols)`.

	Output shape:
		- If `data_format='channels_last'`:
			4D tensor with shape `(batch_size, pooled_rows, pooled_cols, channels)`.
		- If `data_format='channels_first'`:
			4D tensor with shape `(batch_size, channels, pooled_rows, pooled_cols)`.
	"""

	def call(self, input):
		output = tf.nn.avg_pool2d(
			input       = input,
			ksize       = self.pool_size,
			strides     = self.strides,
			padding     = self.padding,
			data_format = self.data_format,
			name        = 'SAvgPool2D_output_avg_pool2d')

		return output


class SMaxPool2D(SPool2D):
	"""Square max pooling operation for spatial data.

	Arguments:
		pool_size: integer or tuple of 2 integers,
			factors by which to downscale (vertical, horizontal).
			`(2, 2)` will halve the input in both spatial dimension.
			If only one integer is specified, the same window length
			will be used for both dimensions.
		strides: Integer, tuple of 2 integers, or None.
			Strides values.
			If None, it will default to `pool_size`.
		padding: One of `"valid"` or `"same"` (case-insensitive).
		data_format: A string,
			one of `channels_last` (default) or `channels_first`.
			The ordering of the dimensions in the inputs.
			`channels_last` corresponds to inputs with shape
			`(batch, height, width, channels)` while `channels_first`
			corresponds to inputs with shape
			`(batch, channels, height, width)`.
			It defaults to the `image_data_format` value found in your
			Keras config file at `~/.keras/keras.json`.
			If you never set it, then it will be "channels_last".

	Input shape:
		- If `data_format='channels_last'`:
			4D tensor with shape `(batch_size, rows, cols, channels)`.
		- If `data_format='channels_first'`:
			4D tensor with shape `(batch_size, channels, rows, cols)`.

	Output shape:
		- If `data_format='channels_last'`:
			4D tensor with shape `(batch_size, pooled_rows, pooled_cols, channels)`.
		- If `data_format='channels_first'`:
			4D tensor with shape `(batch_size, channels, pooled_rows, pooled_cols)`.
	"""

	def call(self, input):
		output = tf.nn.max_pool2d(
			input       = input,
			ksize       = self.pool_size,
			strides     = self.strides,
			padding     = self.padding,
			data_format = self.data_format,
			name        = 'SMaxPool2D_output_max_pool2d')

		return output




class SSampling2D(tf.keras.layers.Layer):
	"""Square sampling layer for 2D inputs.

	Resized images will be distorted if their original aspect ratio is not
	the same as `target_size`. To avoid distortions see
	`tf.image.resize_with_pad`.

	When 'antialias' is true, the sampling filter will anti-alias the input image
	as well as interpolate. When downsampling an image with [anti-aliasing](
	https://en.wikipedia.org/wiki/Spatial_anti-aliasing) the sampling filter
	kernel is scaled in order to properly anti-alias the input image signal.
	'antialias' has no effect when upsampling an image.

	* 	<b>`bilinear`</b>: [Bilinear interpolation.](
		https://en.wikipedia.org/wiki/Bilinear_interpolation) If 'antialias' is
		true, becomes a hat/tent filter function with radius 1 when downsampling.
	* 	<b>`lanczos3`</b>: [Lanczos kernel](
		https://en.wikipedia.org/wiki/Lanczos_resampling) with radius 3.
		High-quality practical filter but may have some ringing especially on
		synthetic images.
	* 	<b>`lanczos5`</b>: [Lanczos kernel] (
		https://en.wikipedia.org/wiki/Lanczos_resampling) with radius 5.
		Very-high-quality filter but may have stronger ringing.
	* 	<b>`bicubic`</b>: [Cubic interpolant](
		https://en.wikipedia.org/wiki/Bicubic_interpolation) of Keys. Equivalent to
		Catmull-Rom kernel. Reasonably good quality and faster than Lanczos3Kernel,
		particularly when upsampling.
	* 	<b>`gaussian`</b>: [Gaussian kernel](
		https://en.wikipedia.org/wiki/Gaussian_filter) with radius 3,
		sigma = 1.5 / 3.0.
	* 	<b>`nearest`</b>: [Nearest neighbor interpolation.](
		https://en.wikipedia.org/wiki/Nearest-neighbor_interpolation)
		'antialias' has no effect when used with nearest neighbor interpolation.
	* 	<b>`area`</b>: Anti-aliased resampling with area interpolation.
		'antialias' has no effect when used with area interpolation; it
		always anti-aliases.
	* 	<b>`mitchellcubic`</b>: Mitchell-Netravali Cubic non-interpolating filter.
		For synthetic images (especially those lacking proper prefiltering), less
		ringing than Keys cubic kernel but less sharp.

	Note that near image edges the filtering kernel may be partially outside the
	image boundaries. For these pixels, only input pixels inside the image will be
	included in the filter sum, and the output value will be appropriately
	normalized.

	The return value has the same type as `images` if `interpolation` is
	`ResizeMethod.NEAREST_NEIGHBOR`. Otherwise, the return value has type
	`float32`.

	Arguments:
		images: 4-D Tensor of shape `[batch, height, width, channels]` or 3-D Tensor
			of shape `[height, width, channels]`.
		size_factor: A 1-D int32 Tensor of 2 elements:
			`target_size = (round(size_factor[0] * height), round(size_factor[1] * width))`.
			The new size factor for the images.
		data_format: A string,
			one of `channels_last` (default) or `channels_first`.
			The ordering of the dimensions in the inputs.
			`channels_last` corresponds to inputs with shape
			`(batch, height, width, channels)` while `channels_first`
			corresponds to inputs with shape
			`(batch, channels, height, width)`.
			It defaults to the `image_data_format` value found in your
			Keras config file at `~/.keras/keras.json`.
			If you never set it, then it will be "channels_last".
		interpolation: ResizeMethod. Defaults to `nearest`.
		preserve_aspect_ratio: Whether to preserve the aspect ratio. If this is set,
			then `images` will be resized to a size that fits in `target_size` while
			preserving the aspect ratio of the original image. Scales up the image if
			`target_size` is bigger than the current size of the `image`. Defaults to False.
		antialias: Whether to use an anti-aliasing filter when downsampling an
			image.

	Raises:
		ValueError: if the shape of `images` is incompatible with the
			shape arguments to this function
		ValueError: if `target_size` has invalid shape or type.
		ValueError: if an unsupported resize method is specified.

	Returns:
		If `images` was 4-D, a 4-D float Tensor of shape
		`[batch, new_height, new_width, channels]`.
		If `images` was 3-D, a 3-D float Tensor of shape
		`[new_height, new_width, channels]`.
	"""

	def __init__(
		self,
		size_factor           = (2, 2),
		data_format           = 'NHWC',
		interpolation         = 'nearest',
		preserve_aspect_ratio = False,
		antialias             = False,
		**kwargs):

		super().__init__(**kwargs)

		if type(size_factor) is int:
			self.size_factor = (size_factor, size_factor)
		elif type(size_factor) is not tuple:
			self.size_factor = tuple(size_factor)

			if len(size_factor) == 1:
				self.size_factor *= 2
		else:
			self.size_factor = size_factor

		self.data_format           = data_format
		self.interpolation         = interpolation
		self.preserve_aspect_ratio = preserve_aspect_ratio
		self.antialias             = antialias

	def build(self, input_shape):
		super().build(input_shape)

		self.target_size = (round(self.size_factor[0] * input_shape[1]), round(self.size_factor[1] * input_shape[2]))

	def call(self, input):
		output = tf.image.resize(
			images                = input,
			size                  = self.target_size,
			method                = self.interpolation,
			preserve_aspect_ratio = self.preserve_aspect_ratio,
			antialias             = self.antialias,
			name                  = 'SSampling2D_output_resize')

		return output




class HConv2D(tf.keras.layers.Layer):
	"""Hexagonal 2D convolution layer (e.g. spatial convolution over images).

	This layer creates a convolution kernel that is convolved
	with the layer input to produce a tensor of
	outputs. If `use_bias` is True,
	a bias vector is created and added to the outputs. Finally, if
	`activation` is not `None`, it is applied to the outputs as well.

	When using this layer as the first layer in a model,
	provide the keyword argument `input_shape`
	(tuple of integers, does not include the sample axis),
	e.g. `input_shape=(128, 128, 3)` for 128x128 RGB pictures
	in `data_format="channels_last"`.

	Arguments:
		filters: Integer, the dimensionality of the output space
			(i.e. the number of output filters in the convolution).
		kernel_size: An integer or tuple/list of 2 integers, specifying the
			height and width of the 2D convolution window.
			Can be a single integer to specify the same value for
			all spatial dimensions.
		strides: An integer or tuple/list of 2 integers,
			specifying the strides of the convolution along the height and width.
			Can be a single integer to specify the same value for
			all spatial dimensions.
			Specifying any stride value != 1 is incompatible with specifying
			any `dilation_rate` value != 1.
		padding: one of `"valid"` or `"same"` (case-insensitive).
		data_format: A string,
			one of `channels_last` (default) or `channels_first`.
			The ordering of the dimensions in the inputs.
			`channels_last` corresponds to inputs with shape
			`(batch, height, width, channels)` while `channels_first`
			corresponds to inputs with shape
			`(batch, channels, height, width)`.
			It defaults to the `image_data_format` value found in your
			Keras config file at `~/.keras/keras.json`.
			If you never set it, then it will be "channels_last".
		dilation_rate: an integer or tuple/list of 2 integers, specifying
			the dilation rate to use for dilated convolution.
			Can be a single integer to specify the same value for
			all spatial dimensions.
			Currently, specifying any `dilation_rate` value != 1 is
			incompatible with specifying any stride value != 1.
		activation: Activation function to use.
			If you don't specify anything, no activation is applied
			(ie. "linear" activation: `a(x) = x`).
		use_bias: Boolean, whether the layer uses a bias vector.
		kernel_initializer: Initializer for the `kernel` weights matrix.
		bias_initializer: Initializer for the bias vector.
		kernel_regularizer: Regularizer function applied to
			the `kernel` weights matrix.
		bias_regularizer: Regularizer function applied to the bias vector.
		activity_regularizer: Regularizer function applied to
			the output of the layer (its "activation")..
		kernel_constraint: Constraint function applied to the kernel matrix.
		bias_constraint: Constraint function applied to the bias vector.
		mode: String,
			one of `hexagonal_kernel` (default), `square_kernel_square_stride`,
			or `square_kernel_hexagonal_stride`.

	Input shape:
		4D tensor with shape:
		`(samples, channels, rows, cols)` if data_format='channels_first'
		or 4D tensor with shape:
		`(samples, rows, cols, channels)` if data_format='channels_last'.

	Output shape:
		4D tensor with shape:
		`(samples, filters, new_rows, new_cols)` if data_format='channels_first'
		or 4D tensor with shape:
		`(samples, new_rows, new_cols, filters)` if data_format='channels_last'.
		`rows` and `cols` values might have changed due to padding.
	"""

	def __init__(
		self,
		filters              = 1,
		kernel_size          = (3, 3),
		strides              = (1, 1),
		padding              = 'SAME',
		data_format          = 'NHWC',
		dilation_rate        = (1, 1),
		activation           = None,
		use_bias             = True,
		kernel_initializer   = tf.initializers.glorot_uniform,
		bias_initializer     = tf.initializers.zeros,
		kernel_regularizer   = None,
		bias_regularizer     = None,
		activity_regularizer = None,
		kernel_constraint    = None,
		bias_constraint      = None,
		mode                 = 'hexagonal_kernel',
		**kwargs):

		super().__init__(**kwargs)

		self.filters = filters

		if type(kernel_size) is int:
			self.kernel_size = (kernel_size, kernel_size)
		elif type(kernel_size) is not tuple:
			self.kernel_size = tuple(kernel_size)

			if len(kernel_size) == 1:
				self.kernel_size *= 2
		else:
			self.kernel_size = kernel_size

		if type(strides) is int:
			self.strides = (strides, strides)
		elif type(strides) is not tuple:
			self.strides = tuple(strides)

			if len(strides) == 1:
				self.strides *= 2
		else:
			self.strides = strides

		self.padding       = padding
		self.data_format   = data_format
		self.dilation_rate = dilation_rate

		if type(activation) is str:
			self.activation = tf.keras.activations(activation)
		else:
			self.activation = activation

		self.use_bias             = use_bias
		self.kernel_initializer   = kernel_initializer
		self.bias_initializer     = bias_initializer
		self.kernel_regularizer   = kernel_regularizer
		self.bias_regularizer     = bias_regularizer
		self.activity_regularizer = activity_regularizer
		self.kernel_constraint    = kernel_constraint
		self.bias_constraint      = bias_constraint

		self.mode = mode

	def build_masks(self, mask_size=(3, 3)):
		return build_masks(mask_size)

	def build(self, input_shape):
		super().build(input_shape)


		kernel_shape = (self.kernel_size[0], self.kernel_size[1], input_shape[3], self.filters) # HWIOC
		bias_shape   = self.filters

		self.kernel = self.add_variable(
			name        = 'HConv2D_kernel_add_variable',
			shape       = kernel_shape,
			initializer = self.kernel_initializer,
			trainable   = True)

		self.bias = self.add_variable(
			name        = 'HConv2D_bias_add_variable',
			shape       = bias_shape,
			initializer = self.bias_initializer,
			trainable   = True)

		if self.mode == 'hexagonal_kernel':
			(kernel_mask_even_rows, _) = self.build_masks(mask_size=self.kernel_size)

			self.kernel_mask_even_rows = tf.convert_to_tensor(
				value = kernel_mask_even_rows,
				dtype = tf.float32,
				name  = 'HConv2D_kernel_mask_even_rows_convert_to_tensor')


		if self.mode == 'hexagonal_kernel' or self.mode == 'square_kernel_hexagonal_stride':
			self.strides = (2 * self.strides[0], self.strides[1])

	def call_hexagonal_kernel(self, input):
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
				else:
					kernel_masked_odd_rows.append(kernel_masked_even_rows[kernel_row])

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




		########################################################################
		# Concatenate the output tensor
		########################################################################


		# SResNet_v1: ~75s


		########################################################################
		# v0 (HResNet_v1: ~160s): tf.concat (best performance but invalid)
		########################################################################

		'''
		output = tf.concat(
			values = (output_even_rows, output_odd_rows),
			axis   = 1,
			name   = 'HConv2D_output_concat')
		'''


		########################################################################
		# v1 (HResNet_v1: ~240s): looped tf.concat
		########################################################################

		'''
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
		'''


		########################################################################
		# v2 (HResNet_v1: ~220s): list-based tf.concat
		########################################################################

		output = []

		if input.shape[1] > 1:
			output.extend([output_even_rows[:, 0:1, :, :], output_odd_rows[:, 0:1, :, :]])

			for h in range(1, min(output_even_rows.shape[1], output_odd_rows.shape[1])):
				output.extend([output_even_rows[:, h:h+1, :, :], output_odd_rows[:, h:h+1, :, :]])

			if output_even_rows.shape[1] > output_odd_rows.shape[1]:
				output.append(output_even_rows[:, -1:, :, :])
		else:
			output = output_even_rows

		output = tf.concat(
			values = output,
			axis   = 1,
			name   = 'HConv2D_output_concat')


		########################################################################
		# v3 (HResNet_v1: ~165s): tf.expand_dims + tf.concat + tf.reshape
		########################################################################

		# TODO: add names for tf.*

		# Reference: https://stackoverflow.com/questions/46431983/concatenate-two-tensors-in-alternate-fashion-tensorflow

		'''
		output_shape             = output_even_rows.shape.as_list()
		output_shape[0]          = -1
		output_shape[1]         += output_odd_rows.shape[1]
		expand_dims_concat_axis  = len(output_shape)

		output_even_rows = tf.expand_dims(output_even_rows, expand_dims_concat_axis)
		output_odd_rows  = tf.expand_dims(output_odd_rows,  expand_dims_concat_axis)

		output = tf.concat([output_even_rows, output_odd_rows], expand_dims_concat_axis)

		output = tf.reshape(output, output_shape)
		'''




		output = tf.nn.bias_add(
			value       = output,
			bias        = self.bias,
			data_format = self.data_format,
			name        = 'HConv2D_output_bias_add')

		if self.activation is not None:
			output = self.activation(
				features = output,
				name     = 'HConv2D_output_activation')

		return output

	def call_square_kernel_square_stride(self, input):
		output = tf.nn.conv2d(
			input       = input,
			filters     = self.kernel,
			strides     = self.strides,
			padding     = self.padding,
			data_format = self.data_format,
			dilations   = self.dilation_rate,
			name        = 'HConv2D_output_conv2d')

		output = tf.nn.bias_add(
			value       = output,
			bias        = self.bias,
			data_format = self.data_format,
			name        = 'HConv2D_output_bias_add')

		if self.activation is not None:
			output = self.activation(
				features = output,
				name     = 'HConv2D_output_activation')

		return output

	def call_square_kernel_hexagonal_stride(self, input):
		kernel_even_rows = self.kernel

		if input.shape[1] > 1:
			kernel_odd_rows = []

			for kernel_row in range(kernel_even_rows.shape[0]):
				if kernel_row % 2:
					kernel_odd_rows.append(
						tf.pad(
							tensor          = kernel_even_rows[kernel_row],
							paddings        = ((1, 0), (0, 0), (0, 0)),
							mode            = 'CONSTANT',
							constant_values = 0,
							name            = 'HConv2D_kernel_odd_rows_pad'))
				else:
					kernel_odd_rows.append(
						tf.pad(
							tensor          = kernel_even_rows[kernel_row],
							paddings        = ((0, 1), (0, 0), (0, 0)),
							mode            = 'CONSTANT',
							constant_values = 0,
							name            = 'HConv2D_kernel_odd_rows_pad'))

			kernel_odd_rows = tf.stack(
				values = kernel_odd_rows,
				axis   = 0,
				name   = 'HConv2D_kernel_odd_rows_stack')


		output_even_rows = tf.nn.conv2d(
			input       = input,
			filters     = kernel_even_rows,
			strides     = self.strides,
			padding     = self.padding,
			data_format = self.data_format,
			dilations   = self.dilation_rate,
			name        = 'HConv2D_output_even_rows_conv2d')

		if input.shape[1] > 1:
			output_odd_rows = tf.nn.conv2d(
				input       = input[:, 1:, :, :],
				filters     = kernel_odd_rows,
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

		if self.activation is not None:
			output = self.activation(
				features = output,
				name     = 'HConv2D_output_activation')

		return output

	def call(self, input):
		if self.mode == 'hexagonal_kernel':
			output = self.call_hexagonal_kernel(input)
		elif self.mode == 'square_kernel_square_stride':
			output = self.call_square_kernel_square_stride(input)
		else: # 'square_kernel_hexagonal_stride'
			output = self.call_square_kernel_hexagonal_stride(input)

		return output


class HConv2DTranspose(HConv2D):
	"""Hexagonal 2D transposed convolution layer (e.g. spatial convolution over images, sometimes called Deconvolution).

	This layer creates a convolution kernel that is convolved
	with the layer input to produce a tensor of
	outputs. If `use_bias` is True,
	a bias vector is created and added to the outputs. Finally, if
	`activation` is not `None`, it is applied to the outputs as well.

	When using this layer as the first layer in a model,
	provide the keyword argument `input_shape`
	(tuple of integers, does not include the sample axis),
	e.g. `input_shape=(128, 128, 3)` for 128x128 RGB pictures
	in `data_format="channels_last"`.

	Arguments:
		filters: Integer, the dimensionality of the output space
			(i.e. the number of output filters in the convolution).
		kernel_size: An integer or tuple/list of 2 integers, specifying the
			height and width of the 2D convolution window.
			Can be a single integer to specify the same value for
			all spatial dimensions.
		strides: An integer or tuple/list of 2 integers,
			specifying the strides of the convolution along the height and width.
			Can be a single integer to specify the same value for
			all spatial dimensions.
			Specifying any stride value != 1 is incompatible with specifying
			any `dilation_rate` value != 1.
		padding: one of `"valid"` or `"same"` (case-insensitive).
		data_format: A string,
			one of `channels_last` (default) or `channels_first`.
			The ordering of the dimensions in the inputs.
			`channels_last` corresponds to inputs with shape
			`(batch, height, width, channels)` while `channels_first`
			corresponds to inputs with shape
			`(batch, channels, height, width)`.
			It defaults to the `image_data_format` value found in your
			Keras config file at `~/.keras/keras.json`.
			If you never set it, then it will be "channels_last".
		dilation_rate: an integer or tuple/list of 2 integers, specifying
			the dilation rate to use for dilated convolution.
			Can be a single integer to specify the same value for
			all spatial dimensions.
			Currently, specifying any `dilation_rate` value != 1 is
			incompatible with specifying any stride value != 1.
		activation: Activation function to use.
			If you don't specify anything, no activation is applied
			(ie. "linear" activation: `a(x) = x`).
		use_bias: Boolean, whether the layer uses a bias vector.
		kernel_initializer: Initializer for the `kernel` weights matrix.
		bias_initializer: Initializer for the bias vector.
		kernel_regularizer: Regularizer function applied to
			the `kernel` weights matrix.
		bias_regularizer: Regularizer function applied to the bias vector.
		activity_regularizer: Regularizer function applied to
			the output of the layer (its "activation")..
		kernel_constraint: Constraint function applied to the kernel matrix.
		bias_constraint: Constraint function applied to the bias vector.
		mode: String,
			one of `hexagonal_kernel` (default), `square_kernel_square_stride`,
			or `square_kernel_hexagonal_stride`.

	Input shape:
		4D tensor with shape:
		`(samples, channels, rows, cols)` if data_format='channels_first'
		or 4D tensor with shape:
		`(samples, rows, cols, channels)` if data_format='channels_last'.

	Output shape:
		4D tensor with shape:
		`(samples, filters, new_rows, new_cols)` if data_format='channels_first'
		or 4D tensor with shape:
		`(samples, new_rows, new_cols, filters)` if data_format='channels_last'.
		`rows` and `cols` values might have changed due to padding.
	"""

	def build_masks(self, mask_size=(3, 3)):
		return build_masks(mask_size)

	def build(self, input_shape):
		super().build(input_shape)


		kernel_shape = (self.kernel_size[0], self.kernel_size[1], self.filters, input_shape[3]) # HWOIC
		bias_shape   = self.filters

		self.kernel = self.add_variable(
			name        = 'HConv2DTranspose_kernel_add_variable',
			shape       = kernel_shape,
			initializer = self.kernel_initializer,
			trainable   = True)

		self.bias = self.add_variable(
			name        = 'HConv2DTranspose_bias_add_variable',
			shape       = bias_shape,
			initializer = self.bias_initializer,
			trainable   = True)

		if self.mode == 'hexagonal_kernel':
			(kernel_mask_even_rows, _) = self.build_masks(mask_size=self.kernel_size)

			self.kernel_mask_even_rows = tf.convert_to_tensor(
				value = kernel_mask_even_rows,
				dtype = tf.float32,
				name  = 'HConv2DTranspose_kernel_mask_even_rows_convert_to_tensor')


		if self.mode == 'hexagonal_kernel' or self.mode == 'square_kernel_hexagonal_stride':
			self.strides = (int(self.strides[0] / 2), self.strides[1])

	def call_hexagonal_kernel(self, input):
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
							name  = 'HConv2DTranspose_kernel_masked_odd_rows_roll'))
				else:
					kernel_masked_odd_rows.append(kernel_masked_even_rows[kernel_row])

			kernel_masked_odd_rows = tf.stack(
				values = kernel_masked_odd_rows,
				axis   = 0,
				name   = 'HConv2DTranspose_kernel_masked_odd_rows_stack')


		if type(self.strides) is int:
			output_shape = (tf.shape(input)[0], self.strides * input.shape[1], self.strides * input.shape[2], self.filters)
		else:
			output_shape = (tf.shape(input)[0], self.strides[0] * input.shape[1], self.strides[1] * input.shape[2], self.filters)

		output_even_rows = tf.nn.conv2d_transpose(
			input        = input,
			filters      = kernel_masked_even_rows,
			output_shape = output_shape,
			strides      = self.strides,
			padding      = self.padding,
			data_format  = self.data_format,
			dilations    = self.dilation_rate,
			name         = 'HConv2DTranspose_output_even_rows_conv2d_transpose')

		if input.shape[1] > 1:
			output_odd_rows = tf.nn.conv2d_transpose(
				input        = input[:, 1:, :, :],
				filters      = kernel_masked_odd_rows,
				output_shape = output_shape,
				strides      = self.strides,
				padding      = self.padding,
				data_format  = self.data_format,
				dilations    = self.dilation_rate,
				name         = 'HConv2DTranspose_output_odd_rows_conv2d_transpose')


		if input.shape[1] > 1:
			output = tf.concat(
				values = (output_even_rows[:, 0:1, :, :], output_odd_rows[:, 0:1, :, :]),
				axis   = 1,
				name   = 'HConv2DTranspose_output_concat')

			for h in range(1, min(output_even_rows.shape[1], output_odd_rows.shape[1])):
				output = tf.concat(
					values = (output, output_even_rows[:, h:h+1, :, :], output_odd_rows[:, h:h+1, :, :]),
					axis   = 1,
					name   = 'HConv2DTranspose_output_concat')

			if output_even_rows.shape[1] > output_odd_rows.shape[1]:
				output = tf.concat(
					values = (output, output_even_rows[:, -1:, :, :]),
					axis   = 1,
					name   = 'HConv2DTranspose_output_concat')
		else:
			output = output_even_rows

		output = tf.nn.bias_add(
			value       = output,
			bias        = self.bias,
			data_format = self.data_format,
			name        = 'HConv2DTranspose_output_bias_add')

		if self.activation is not None:
			output = self.activation(
				features = output,
				name     = 'HConv2DTranspose_output_activation')

		return output

	def call_square_kernel_square_stride(self, input):
		if type(self.strides) is int:
			output_shape = (tf.shape(input)[0], self.strides * input.shape[1], self.strides * input.shape[2], self.filters)
		else:
			output_shape = (tf.shape(input)[0], self.strides[0] * input.shape[1], self.strides[1] * input.shape[2], self.filters)

		output = tf.nn.conv2d_transpose(
			input        = input,
			filters      = self.kernel,
			output_shape = output_shape,
			strides      = self.strides,
			padding      = self.padding,
			data_format  = self.data_format,
			dilations    = self.dilation_rate,
			name         = 'HConv2DTranspose_output_conv2d_transpose')

		output = tf.nn.bias_add(
			value       = output,
			bias        = self.bias,
			data_format = self.data_format,
			name        = 'HConv2DTranspose_output_bias_add')

		if self.activation is not None:
			output = self.activation(
				features = output,
				name     = 'HConv2DTranspose_output_activation')

		return output

	def call_square_kernel_hexagonal_stride(self, input):
		kernel_even_rows = self.kernel

		if input.shape[1] > 1:
			kernel_odd_rows = []

			for kernel_row in range(kernel_even_rows.shape[0]):
				if kernel_row % 2:
					kernel_odd_rows.append(
						tf.pad(
							tensor          = kernel_even_rows[kernel_row],
							paddings        = ((1, 0), (0, 0), (0, 0)),
							mode            = 'CONSTANT',
							constant_values = 0,
							name            = 'HConv2DTranspose_kernel_odd_rows_pad'))
				else:
					kernel_odd_rows.append(
						tf.pad(
							tensor          = kernel_even_rows[kernel_row],
							paddings        = ((0, 1), (0, 0), (0, 0)),
							mode            = 'CONSTANT',
							constant_values = 0,
							name            = 'HConv2DTranspose_kernel_odd_rows_pad'))

			kernel_odd_rows = tf.stack(
				values = kernel_odd_rows,
				axis   = 0,
				name   = 'HConv2DTranspose_kernel_odd_rows_stack')


		if type(self.strides) is int:
			output_shape = (tf.shape(input)[0], self.strides * input.shape[1], self.strides * input.shape[2], self.filters)
		else:
			output_shape = (tf.shape(input)[0], self.strides[0] * input.shape[1], self.strides[1] * input.shape[2], self.filters)

		output_even_rows = tf.nn.conv2d_transpose(
			input        = input,
			filters      = kernel_even_rows,
			output_shape = output_shape,
			strides      = self.strides,
			padding      = self.padding,
			data_format  = self.data_format,
			dilations    = self.dilation_rate,
			name         = 'HConv2DTranspose_output_even_rows_conv2d_transpose')

		if input.shape[1] > 1:
			output_odd_rows = tf.nn.conv2d_transpose(
				input        = input[:, 1:, :, :],
				filters      = kernel_odd_rows,
				output_shape = output_shape,
				strides      = self.strides,
				padding      = self.padding,
				data_format  = self.data_format,
				dilations    = self.dilation_rate,
				name         = 'HConv2DTranspose_output_odd_rows_conv2d_transpose')


		if input.shape[1] > 1:
			output = tf.concat(
				values = (output_even_rows[:, 0:1, :, :], output_odd_rows[:, 0:1, :, :]),
				axis   = 1,
				name   = 'HConv2DTranspose_output_concat')

			for h in range(1, min(output_even_rows.shape[1], output_odd_rows.shape[1])):
				output = tf.concat(
					values = (output, output_even_rows[:, h:h+1, :, :], output_odd_rows[:, h:h+1, :, :]),
					axis   = 1,
					name   = 'HConv2DTranspose_output_concat')

			if output_even_rows.shape[1] > output_odd_rows.shape[1]:
				output = tf.concat(
					values = (output, output_even_rows[:, -1:, :, :]),
					axis   = 1,
					name   = 'HConv2DTranspose_output_concat')
		else:
			output = output_even_rows

		output = tf.nn.bias_add(
			value       = output,
			bias        = self.bias,
			data_format = self.data_format,
			name        = 'HConv2DTranspose_output_bias_add')

		if self.activation is not None:
			output = self.activation(
				features = output,
				name     = 'HConv2DTranspose_output_activation')

		return output

	def call(self, input):
		if self.mode == 'hexagonal_kernel':
			output = self.call_hexagonal_kernel(input)
		elif self.mode == 'square_kernel_square_stride':
			output = self.call_square_kernel_square_stride(input)
		else: # 'square_kernel_hexagonal_stride'
			output = self.call_square_kernel_hexagonal_stride(input)

		return output


class HGConv2D(HConv2D):
	"""Hexagonal 2D group convolution layer (e.g. spatial convolution over images).

	This layer creates a convolution kernel that is convolved
	with the layer input to produce a tensor of
	outputs. If `use_bias` is True,
	a bias vector is created and added to the outputs. Finally, if
	`activation` is not `None`, it is applied to the outputs as well.

	When using this layer as the first layer in a model,
	provide the keyword argument `input_shape`
	(tuple of integers, does not include the sample axis),
	e.g. `input_shape=(128, 128, 3)` for 128x128 RGB pictures
	in `data_format="channels_last"`.

	Arguments:
		filters: Integer, the dimensionality of the output space
			(i.e. the number of output filters in the convolution).
		kernel_size: An integer or tuple/list of 2 integers, specifying the
			height and width of the 2D convolution window.
			Can be a single integer to specify the same value for
			all spatial dimensions.
		strides: An integer or tuple/list of 2 integers,
			specifying the strides of the convolution along the height and width.
			Can be a single integer to specify the same value for
			all spatial dimensions.
			Specifying any stride value != 1 is incompatible with specifying
			any `dilation_rate` value != 1.
		padding: one of `"valid"` or `"same"` (case-insensitive).
		data_format: A string,
			one of `channels_last` (default) or `channels_first`.
			The ordering of the dimensions in the inputs.
			`channels_last` corresponds to inputs with shape
			`(batch, height, width, channels)` while `channels_first`
			corresponds to inputs with shape
			`(batch, channels, height, width)`.
			It defaults to the `image_data_format` value found in your
			Keras config file at `~/.keras/keras.json`.
			If you never set it, then it will be "channels_last".
		dilation_rate: an integer or tuple/list of 2 integers, specifying
			the dilation rate to use for dilated convolution.
			Can be a single integer to specify the same value for
			all spatial dimensions.
			Currently, specifying any `dilation_rate` value != 1 is
			incompatible with specifying any stride value != 1.
		activation: Activation function to use.
			If you don't specify anything, no activation is applied
			(ie. "linear" activation: `a(x) = x`).
		use_bias: Boolean, whether the layer uses a bias vector.
		kernel_initializer: Initializer for the `kernel` weights matrix.
		bias_initializer: Initializer for the bias vector.
		kernel_regularizer: Regularizer function applied to
			the `kernel` weights matrix.
		bias_regularizer: Regularizer function applied to the bias vector.
		activity_regularizer: Regularizer function applied to
			the output of the layer (its "activation")..
		kernel_constraint: Constraint function applied to the kernel matrix.
		bias_constraint: Constraint function applied to the bias vector.

	Input shape:
		4D tensor with shape:
		`(samples, channels, rows, cols)` if data_format='channels_first'
		or 4D tensor with shape:
		`(samples, rows, cols, channels)` if data_format='channels_last'.

	Output shape:
		4D tensor with shape:
		`(samples, filters, new_rows, new_cols)` if data_format='channels_first'
		or 4D tensor with shape:
		`(samples, new_rows, new_cols, filters)` if data_format='channels_last'.
		`rows` and `cols` values might have changed due to padding.
	"""

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
							name  = 'HGConv2D_kernel_masked_odd_rows_roll'))
				else:
					kernel_masked_odd_rows.append(kernel_masked_even_rows[kernel_row])

			kernel_masked_odd_rows = tf.stack(
				values = kernel_masked_odd_rows,
				axis   = 0,
				name   = 'HGConv2D_kernel_masked_odd_rows_stack')


		kernel_masked_even_rows_rotated = []
		kernel_masked_odd_rows_rotated  = []

		group_size = math.ceil(self.kernel.shape[3] / 6)

		for k in range(6):
			output_channels_index_min = k * group_size

			if k < 5:
				output_channels_index_max = (k + 1) * group_size
			else:
				output_channels_index_max = self.kernel.shape[3]

			kernels_masked = (kernel_masked_even_rows[:, :, :, output_channels_index_min:output_channels_index_max], kernel_masked_odd_rows[:, :, :, output_channels_index_min:output_channels_index_max])

			kernels_masked_rotated = rotate_hexagonal_kernels(kernels=kernels_masked, angle = k * 60)

			kernel_masked_even_rows_rotated.append(kernels_masked_rotated[0])
			kernel_masked_odd_rows_rotated.append(kernels_masked_rotated[1])

		kernel_masked_even_rows_rotated = tf.concat(
			values = kernel_masked_even_rows_rotated,
			axis   = 3,
			name   = 'HGConv2D_kernel_masked_even_rows_rotated_concat')

		kernel_masked_odd_rows_rotated = tf.concat(
			values = kernel_masked_odd_rows_rotated,
			axis   = 3,
			name   = 'HGConv2D_kernel_masked_odd_rows_rotated_concat')


		output_even_rows = tf.nn.conv2d(
			input       = input,
			filters     = kernel_masked_even_rows_rotated,
			strides     = self.strides,
			padding     = self.padding,
			data_format = self.data_format,
			dilations   = self.dilation_rate,
			name        = 'HGConv2D_output_even_rows_conv2d')

		if input.shape[1] > 1:
			output_odd_rows = tf.nn.conv2d(
				input       = input[:, 1:, :, :],
				filters     = kernel_masked_odd_rows_rotated,
				strides     = self.strides,
				padding     = self.padding,
				data_format = self.data_format,
				dilations   = self.dilation_rate,
				name        = 'HGConv2D_output_odd_rows_conv2d')


		if input.shape[1] > 1:
			output = tf.concat(
				values = (output_even_rows[:, 0:1, :, :], output_odd_rows[:, 0:1, :, :]),
				axis   = 1,
				name   = 'HGConv2D_output_concat')

			for h in range(1, min(output_even_rows.shape[1], output_odd_rows.shape[1])):
				output = tf.concat(
					values = (output, output_even_rows[:, h:h+1, :, :], output_odd_rows[:, h:h+1, :, :]),
					axis   = 1,
					name   = 'HGConv2D_output_concat')

			if output_even_rows.shape[1] > output_odd_rows.shape[1]:
				output = tf.concat(
					values = (output, output_even_rows[:, -1:, :, :]),
					axis   = 1,
					name   = 'HGConv2D_output_concat')
		else:
			output = output_even_rows

		output = tf.nn.bias_add(
			value       = output,
			bias        = self.bias,
			data_format = self.data_format,
			name        = 'HGConv2D_output_bias_add')

		if self.activation is not None:
			output = self.activation(
				features = output,
				name     = 'HGConv2D_output_activation')

		return output




class HPool2D(tf.keras.layers.Layer):
	"""Hexagonal pooling operation for spatial data.

	Arguments:
		pool_size: integer or tuple of 2 integers,
			factors by which to downscale (vertical, horizontal).
			`(2, 2)` will halve the input in both spatial dimension.
			If only one integer is specified, the same window length
			will be used for both dimensions.
		strides: Integer, tuple of 2 integers, or None.
			Strides values.
			If None, it will default to `pool_size`.
		padding: One of `"valid"` or `"same"` (case-insensitive).
		data_format: A string,
			one of `channels_last` (default) or `channels_first`.
			The ordering of the dimensions in the inputs.
			`channels_last` corresponds to inputs with shape
			`(batch, height, width, channels)` while `channels_first`
			corresponds to inputs with shape
			`(batch, channels, height, width)`.
			It defaults to the `image_data_format` value found in your
			Keras config file at `~/.keras/keras.json`.
			If you never set it, then it will be "channels_last".
		mode: String,
			one of `hexagonal_kernel` (default), `square_kernel_square_stride`,
			or `square_kernel_hexagonal_stride`.

	Input shape:
		- If `data_format='channels_last'`:
			4D tensor with shape `(batch_size, rows, cols, channels)`.
		- If `data_format='channels_first'`:
			4D tensor with shape `(batch_size, channels, rows, cols)`.

	Output shape:
		- If `data_format='channels_last'`:
			4D tensor with shape `(batch_size, pooled_rows, pooled_cols, channels)`.
		- If `data_format='channels_first'`:
			4D tensor with shape `(batch_size, channels, pooled_rows, pooled_cols)`.
	"""

	def __init__(
		self,
		pool_size   = (3, 3),
		strides     = None,
		padding     = 'SAME',
		data_format = 'NHWC',
		mode        = 'hexagonal_kernel',
		**kwargs):

		super().__init__(**kwargs)

		if type(pool_size) is int:
			self.pool_size = (pool_size, pool_size)
		elif type(pool_size) is not tuple:
			self.pool_size = tuple(pool_size)

			if len(pool_size) == 1:
				self.pool_size *= 2
		else:
			self.pool_size = pool_size

		if strides is None:
			self.strides = pool_size
		else:
			self.strides = strides

		self.padding     = padding
		self.data_format = data_format

		self.mode = mode

	def build_masks(self, mask_size=(3, 3)):
		return build_masks(mask_size)

	def build_offsets(self, mask_size=(3, 3)):
		return build_offsets(mask_size)

	def build(self, input_shape):
		super().build(input_shape)


		if self.mode == 'hexagonal_kernel':
			self.pool_size2 = (int((self.pool_size[0] - 1) / 2), int((self.pool_size[1] - 1) / 2))

			(kernel_mask_even_rows, kernel_mask_odd_rows)                 = self.build_masks(mask_size=self.pool_size)
			(self.kernel_offsets_even_rows, self.kernel_offsets_odd_rows) = self.build_offsets(mask_size=self.pool_size)

			self.kernel_mask_even_rows = tf.convert_to_tensor(
				value = kernel_mask_even_rows,
				dtype = tf.float32,
				name  = 'HPool2D_kernel_mask_even_rows_convert_to_tensor')

			self.kernel_mask_odd_rows = tf.convert_to_tensor(
				value = kernel_mask_odd_rows,
				dtype = tf.float32,
				name  = 'HPool2D_kernel_mask_odd_rows_convert_to_tensor')




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

			input_offsets_padded = [(offset[0] + input_offsets_padding_h[0], offset[1] + input_offsets_padding_w[0]) for offset in input_offsets]
			input_offsets_center = (sum(input_offsets_padded[:][0]) / len(input_offsets_padded[:][0]), sum(input_offsets_padded[:][1]) / len(input_offsets_padded[:][1]))

			input_offsets_centered = []

			for offset in input_offsets_padded:
				if not (offset[0] - input_offsets_padding_h[0]) % 2:
					input_offsets_centered.append(((offset[0] - input_offsets_center[0]) * 1.5, (offset[1] - input_offsets_center[1]) * math.sqrt(3)))
				else:
					input_offsets_centered.append(((offset[0] - input_offsets_center[0]) * 1.5, (offset[1] + 0.5 - input_offsets_center[1]) * math.sqrt(3)))

			output_shape_base = math.sqrt(len(input_offsets_padded))
			output_shape_h    = math.floor((input_shape_h / input_shape_w) * output_shape_base)
			output_shape_w    = math.floor((input_shape_w / input_shape_h) * output_shape_base)
			output_shape      = (input_shape[0], output_shape_h, output_shape_w, input_shape[3])

			output_offsets = []

			for h in range(output_shape_h):
				for w in range(output_shape_w):
					if not h % 2:
						output_offsets.append((h * 1.5, w * math.sqrt(3)))
					else:
						output_offsets.append((h * 1.5, (w + 0.5) * math.sqrt(3)))

			output_offsets_scale    = (input_offsets_shape[0] / output_shape_h, input_offsets_shape[1] / output_shape_w)
			output_offsets_scaled   = [(output_offsets_scale[0] * offset[0], output_offsets_scale[1] * offset[1]) for offset in output_offsets]
			output_offsets_center   = (sum(output_offsets_scaled[:][0]) / len(output_offsets_scaled[:][0]), sum(output_offsets_scaled[:][1]) / len(output_offsets_scaled[:][1]))
			output_offsets_centered = [(offset[0] - output_offsets_center[0], offset[1] - output_offsets_center[1]) for offset in output_offsets_scaled]




			cost_matrix = np.full(shape=(len(input_offsets_centered), len(output_offsets_centered)), fill_value=np.inf)

			for h in range(cost_matrix.shape[0]):
				for w in range(cost_matrix.shape[1]):
					cost_matrix[h][w] = np.linalg.norm(np.asarray(input_offsets_centered[h]) - np.asarray(output_offsets_centered[w]))

			row_indices, col_indices = linear_sum_assignment(cost_matrix)

			if _ENABLE_DEBUGGING:
				costs     = cost_matrix[row_indices, col_indices]
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
		elif self.mode == 'square_kernel_hexagonal_stride':
			self.strides = (2 * self.strides[0], self.strides[1])


class HAvgPool2D(HPool2D):
	"""Hexagonal average pooling operation for spatial data.

	Arguments:
		pool_size: integer or tuple of 2 integers,
			factors by which to downscale (vertical, horizontal).
			`(2, 2)` will halve the input in both spatial dimension.
			If only one integer is specified, the same window length
			will be used for both dimensions.
		strides: Integer, tuple of 2 integers, or None.
			Strides values.
			If None, it will default to `pool_size`.
		padding: One of `"valid"` or `"same"` (case-insensitive).
		data_format: A string,
			one of `channels_last` (default) or `channels_first`.
			The ordering of the dimensions in the inputs.
			`channels_last` corresponds to inputs with shape
			`(batch, height, width, channels)` while `channels_first`
			corresponds to inputs with shape
			`(batch, channels, height, width)`.
			It defaults to the `image_data_format` value found in your
			Keras config file at `~/.keras/keras.json`.
			If you never set it, then it will be "channels_last".
		mode: String,
			one of `hexagonal_kernel` (default), `square_kernel_square_stride`,
			or `square_kernel_hexagonal_stride`.

	Input shape:
		- If `data_format='channels_last'`:
			4D tensor with shape `(batch_size, rows, cols, channels)`.
		- If `data_format='channels_first'`:
			4D tensor with shape `(batch_size, channels, rows, cols)`.

	Output shape:
		- If `data_format='channels_last'`:
			4D tensor with shape `(batch_size, pooled_rows, pooled_cols, channels)`.
		- If `data_format='channels_first'`:
			4D tensor with shape `(batch_size, channels, pooled_rows, pooled_cols)`.
	"""

	def call_hexagonal_kernel(self, input):
		input = tf.pad(
			tensor          = input,
			paddings        = self.paddings,
			mode            = 'CONSTANT',
			constant_values = 0,
			name            = 'HAvgPool2D_input_pad')


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
					name   = 'HAvgPool2D_input_pooling_list_h_concat'))

		input_pooling_list = tf.concat(
			values = input_pooling_list_h,
			axis   = 1,
			name   = 'HAvgPool2D_input_pooling_list_concat')


		output = tf.nn.avg_pool2d(
			input       = input_pooling_list,
			ksize       = self.pool_size,
			strides     = self.strides,
			padding     = 'VALID',
			data_format = self.data_format,
			name        = 'HAvgPool2D_output_avg_pool2d')

		return output

	def call_square_kernel_square_stride(self, input):
		output = tf.nn.avg_pool2d(
			input       = input,
			ksize       = self.pool_size,
			strides     = self.strides,
			padding     = self.padding,
			data_format = self.data_format,
			name        = 'HAvgPool2D_output_avg_pool2d')

		return output

	def call_square_kernel_hexagonal_stride(self, input):
		if input.shape[1] > self.pool_size[0]:
			input_even_rows = tf.pad(
				tensor          = input,
				paddings        = ((0, 0), (0, 0), (int(self.pool_size[0] / 2), 0), (0, 0)),
				mode            = 'CONSTANT',
				constant_values = 0,
				name            = 'HAvgPool2D_input_even_rows_pad')

			input_odd_rows = input[:, self.pool_size[0]:, :, :]
		else:
			input_even_rows = input


		output_even_rows = tf.nn.max_pool2d(
			input       = input_even_rows,
			ksize       = self.pool_size,
			strides     = self.strides,
			padding     = self.padding,
			data_format = self.data_format,
			name        = 'HAvgPool2D_output_even_rows_max_pool2d')

		if input.shape[1] > self.pool_size[0]:
			output_odd_rows = tf.nn.max_pool2d(
				input       = input_odd_rows,
				ksize       = self.pool_size,
				strides     = self.strides,
				padding     = self.padding,
				data_format = self.data_format,
				name        = 'HAvgPool2D_output_odd_rows_max_pool2d')


		if input.shape[1] > self.pool_size[0]:
			output = tf.concat(
				values = (output_even_rows[:, 0:1, :, :], output_odd_rows[:, 0:1, :, :]),
				axis   = 1,
				name   = 'HAvgPool2D_output_concat')

			for h in range(1, min(output_even_rows.shape[1], output_odd_rows.shape[1])):
				output = tf.concat(
					values = (output, output_even_rows[:, h:h+1, :, :], output_odd_rows[:, h:h+1, :, :]),
					axis   = 1,
					name   = 'HAvgPool2D_output_concat')

			if output_even_rows.shape[1] > output_odd_rows.shape[1]:
				output = tf.concat(
					values = (output, output_even_rows[:, -1:, :, :]),
					axis   = 1,
					name   = 'HAvgPool2D_output_concat')
		else:
			output = output_even_rows

		return output

	def call(self, input):
		if self.mode == 'hexagonal_kernel':
			output = self.call_hexagonal_kernel(input)
		elif self.mode == 'square_kernel_square_stride':
			output = self.call_square_kernel_square_stride(input)
		else: # 'square_kernel_hexagonal_stride'
			output = call_square_kernel_hexagonal_stride(input)

		return output


class HMaxPool2D(HPool2D):
	"""Hexagonal max pooling operation for spatial data.

	Arguments:
		pool_size: integer or tuple of 2 integers,
			factors by which to downscale (vertical, horizontal).
			`(2, 2)` will halve the input in both spatial dimension.
			If only one integer is specified, the same window length
			will be used for both dimensions.
		strides: Integer, tuple of 2 integers, or None.
			Strides values.
			If None, it will default to `pool_size`.
		padding: One of `"valid"` or `"same"` (case-insensitive).
		data_format: A string,
			one of `channels_last` (default) or `channels_first`.
			The ordering of the dimensions in the inputs.
			`channels_last` corresponds to inputs with shape
			`(batch, height, width, channels)` while `channels_first`
			corresponds to inputs with shape
			`(batch, channels, height, width)`.
			It defaults to the `image_data_format` value found in your
			Keras config file at `~/.keras/keras.json`.
			If you never set it, then it will be "channels_last".
		mode: String,
			one of `hexagonal_kernel` (default), `square_kernel_square_stride`,
			or `square_kernel_hexagonal_stride`.

	Input shape:
		- If `data_format='channels_last'`:
			4D tensor with shape `(batch_size, rows, cols, channels)`.
		- If `data_format='channels_first'`:
			4D tensor with shape `(batch_size, channels, rows, cols)`.

	Output shape:
		- If `data_format='channels_last'`:
			4D tensor with shape `(batch_size, pooled_rows, pooled_cols, channels)`.
		- If `data_format='channels_first'`:
			4D tensor with shape `(batch_size, channels, pooled_rows, pooled_cols)`.
	"""

	def call_hexagonal_kernel(self, input):
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

	def call_square_kernel_square_stride(self, input):
		output = tf.nn.max_pool2d(
			input       = input,
			ksize       = self.pool_size,
			strides     = self.strides,
			padding     = self.padding,
			data_format = self.data_format,
			name        = 'HMaxPool2D_output_max_pool2d')

		return output

	def call_square_kernel_hexagonal_stride(self, input):
		if input.shape[1] > self.pool_size[0]:
			input_even_rows = tf.pad(
				tensor          = input,
				paddings        = ((0, 0), (0, 0), (int(self.pool_size[0] / 2), 0), (0, 0)),
				mode            = 'CONSTANT',
				constant_values = 0,
				name            = 'HMaxPool2D_input_even_rows_pad')

			input_odd_rows = input[:, self.pool_size[0]:, :, :]
		else:
			input_even_rows = input


		output_even_rows = tf.nn.max_pool2d(
			input       = input_even_rows,
			ksize       = self.pool_size,
			strides     = self.strides,
			padding     = self.padding,
			data_format = self.data_format,
			name        = 'HMaxPool2D_output_even_rows_max_pool2d')

		if input.shape[1] > self.pool_size[0]:
			output_odd_rows = tf.nn.max_pool2d(
				input       = input_odd_rows,
				ksize       = self.pool_size,
				strides     = self.strides,
				padding     = self.padding,
				data_format = self.data_format,
				name        = 'HMaxPool2D_output_odd_rows_max_pool2d')


		if input.shape[1] > self.pool_size[0]:
			output = tf.concat(
				values = (output_even_rows[:, 0:1, :, :], output_odd_rows[:, 0:1, :, :]),
				axis   = 1,
				name   = 'HMaxPool2D_output_concat')

			for h in range(1, min(output_even_rows.shape[1], output_odd_rows.shape[1])):
				output = tf.concat(
					values = (output, output_even_rows[:, h:h+1, :, :], output_odd_rows[:, h:h+1, :, :]),
					axis   = 1,
					name   = 'HMaxPool2D_output_concat')

			if output_even_rows.shape[1] > output_odd_rows.shape[1]:
				output = tf.concat(
					values = (output, output_even_rows[:, -1:, :, :]),
					axis   = 1,
					name   = 'HMaxPool2D_output_concat')
		else:
			output = output_even_rows

		return output

	def call(self, input):
		if self.mode == 'hexagonal_kernel':
			output = self.call_hexagonal_kernel(input)
		elif self.mode == 'square_kernel_square_stride':
			output = self.call_square_kernel_square_stride(input)
		else: # 'square_kernel_hexagonal_stride'
			output = self.call_square_kernel_hexagonal_stride(input)

		return output




class HSampling2D(tf.keras.layers.Layer):
	"""Hexagonal sampling layer for 2D inputs.

	Resized images will be distorted if their original aspect ratio is not
	the same as `target_size`. To avoid distortions see
	`tf.image.resize_with_pad`.

	When 'antialias' is true, the sampling filter will anti-alias the input image
	as well as interpolate. When downsampling an image with [anti-aliasing](
	https://en.wikipedia.org/wiki/Spatial_anti-aliasing) the sampling filter
	kernel is scaled in order to properly anti-alias the input image signal.
	'antialias' has no effect when upsampling an image.

	* 	<b>`bilinear`</b>: [Bilinear interpolation.](
		https://en.wikipedia.org/wiki/Bilinear_interpolation) If 'antialias' is
		true, becomes a hat/tent filter function with radius 1 when downsampling.
	* 	<b>`lanczos3`</b>: [Lanczos kernel](
		https://en.wikipedia.org/wiki/Lanczos_resampling) with radius 3.
		High-quality practical filter but may have some ringing especially on
		synthetic images.
	* 	<b>`lanczos5`</b>: [Lanczos kernel] (
		https://en.wikipedia.org/wiki/Lanczos_resampling) with radius 5.
		Very-high-quality filter but may have stronger ringing.
	* 	<b>`bicubic`</b>: [Cubic interpolant](
		https://en.wikipedia.org/wiki/Bicubic_interpolation) of Keys. Equivalent to
		Catmull-Rom kernel. Reasonably good quality and faster than Lanczos3Kernel,
		particularly when upsampling.
	* 	<b>`gaussian`</b>: [Gaussian kernel](
		https://en.wikipedia.org/wiki/Gaussian_filter) with radius 3,
		sigma = 1.5 / 3.0.
	* 	<b>`nearest`</b>: [Nearest neighbor interpolation.](
		https://en.wikipedia.org/wiki/Nearest-neighbor_interpolation)
		'antialias' has no effect when used with nearest neighbor interpolation.
	* 	<b>`area`</b>: Anti-aliased resampling with area interpolation.
		'antialias' has no effect when used with area interpolation; it
		always anti-aliases.
	* 	<b>`mitchellcubic`</b>: Mitchell-Netravali Cubic non-interpolating filter.
		For synthetic images (especially those lacking proper prefiltering), less
		ringing than Keys cubic kernel but less sharp.

	Note that near image edges the filtering kernel may be partially outside the
	image boundaries. For these pixels, only input pixels inside the image will be
	included in the filter sum, and the output value will be appropriately
	normalized.

	The return value has the same type as `images` if `interpolation` is
	`ResizeMethod.NEAREST_NEIGHBOR`. Otherwise, the return value has type
	`float32`.

	Arguments:
		images: 4-D Tensor of shape `[batch, height, width, channels]` or 3-D Tensor
			of shape `[height, width, channels]`.
		size_factor: A 1-D int32 Tensor of 2 elements:
			`target_size = (round(size_factor[0] * height), round(size_factor[1] * width))`.
			The new size factor for the images.
		data_format: A string,
			one of `channels_last` (default) or `channels_first`.
			The ordering of the dimensions in the inputs.
			`channels_last` corresponds to inputs with shape
			`(batch, height, width, channels)` while `channels_first`
			corresponds to inputs with shape
			`(batch, channels, height, width)`.
			It defaults to the `image_data_format` value found in your
			Keras config file at `~/.keras/keras.json`.
			If you never set it, then it will be "channels_last".
		interpolation: ResizeMethod. Defaults to `nearest`.
		preserve_aspect_ratio: Whether to preserve the aspect ratio. If this is set,
			then `images` will be resized to a size that fits in `target_size` while
			preserving the aspect ratio of the original image. Scales up the image if
			`target_size` is bigger than the current size of the `image`. Defaults to False.
		antialias: Whether to use an anti-aliasing filter when downsampling an
			image.
		mode: String,
			one of `hexagonal_interpolation` (default) or
			`square_interpolation`.

	Raises:
		ValueError: if the shape of `images` is incompatible with the
			shape arguments to this function
		ValueError: if `target_size` has invalid shape or type.
		ValueError: if an unsupported resize method is specified.

	Returns:
		If `images` was 4-D, a 4-D float Tensor of shape
		`[batch, new_height, new_width, channels]`.
		If `images` was 3-D, a 3-D float Tensor of shape
		`[new_height, new_width, channels]`.
	"""

	def __init__(
		self,
		size_factor           = (2, 2),
		data_format           = 'NHWC',
		interpolation         = 'nearest',
		preserve_aspect_ratio = False,
		antialias             = False,
		mode                  = 'hexagonal_interpolation',
		**kwargs):

		super().__init__(**kwargs)

		if type(size_factor) is int:
			self.size_factor = (size_factor, size_factor)
		elif type(size_factor) is not tuple:
			self.size_factor = tuple(size_factor)

			if len(size_factor) == 1:
				self.size_factor *= 2
		else:
			self.size_factor = size_factor

		self.data_format           = data_format
		self.interpolation         = interpolation
		self.preserve_aspect_ratio = preserve_aspect_ratio
		self.antialias             = antialias

		self.mode = mode

	def build(self, input_shape):
		super().build(input_shape)

		self.target_size = (round(self.size_factor[0] * input_shape[1]), round(self.size_factor[1] * input_shape[2]))

	def call(self, input):
		if self.mode == 'hexagonal_interpolation':
			method = 0 if self.interpolation == 'bilinear' else 1 # TODO: add more interpolation methods

			h2_s = tf.identity(input)[:, :self.target_size[0], :self.target_size[1], :] # TODO: initialize hexarray without identity

			output = Hexsamp_h2h(input, h2_s, method)
		else: # 'square_interpolation'
			output = tf.image.resize(
				images                = input,
				size                  = self.target_size,
				method                = self.interpolation,
				preserve_aspect_ratio = self.preserve_aspect_ratio,
				antialias             = self.antialias,
				name                  = 'HSampling2D_output_resize')

		return output

