'''****************************************************************************
 * losses.py: Loss Functions
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

import tensorflow as tf

from tools.compare import _COMPARE_MSE, _compare_s2s, _compare_s2h


################################################################################
# Loss "MSE_test": mean squared error (MSE) loss function
################################################################################

class loss_MSE_test(tf.losses.Loss):
	def __init__(self, **kwargs):
		super().__init__(**kwargs)

	def call(self, y_true, y_pred):
		return tf.reduce_mean(tf.square(y_true - y_pred))




################################################################################
# Loss "MSE_s2s": MSE loss based on Array to Array comparison
################################################################################

class loss_MSE_s2s(tf.losses.Loss):
	def __init__(self, input_shape, output_shape, **kwargs):
		super().__init__(**kwargs)

		self.input_shape  = input_shape
		self.output_shape = output_shape

	def call(self, y_true, y_pred):
		loss = _compare_s2s(
			s1       = y_true,
			s2       = y_pred,
			method   = _COMPARE_MSE,
			s1_shape = self.input_shape,
			s2_shape = self.output_shape)

		return loss


################################################################################
# Loss "MSE_s2h": MSE loss based on Array to Hexarray comparison
################################################################################

class loss_MSE_s2h(tf.losses.Loss):
	def __init__(self, input_shape, output_shape, **kwargs):
		super().__init__(**kwargs)

		self.input_shape  = input_shape
		self.output_shape = output_shape

	def call(self, y_true, y_pred):
		loss = _compare_s2h(
			s       = y_true,
			h       = y_pred,
			method  = _COMPARE_MSE,
			s_shape = self.input_shape,
			h_shape = self.output_shape)

		return loss

