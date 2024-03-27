#!/usr/bin/env python3.7


'''****************************************************************************
 * _Transformation_Efficiency_Performance_all_evaluation.py: Transformation
 *  Efficiency and Performance Evaluation for All Datasets and Metrics
 ******************************************************************************
 * v0.1 - 01.09.2020
 *
 * Copyright (c) 2020 Tobias Schlosser (tobias@tobias-schlosser.net)
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

import os

import numpy as np

from glob import glob


################################################################################
# Default parameters
################################################################################

testset = 'testset'


################################################################################
# Transformation Efficiency and Performance evaluation
################################################################################

for dat in glob(os.path.join(testset, '*.dat')):
	current_dataset = os.path.basename(dat)

	with open(dat) as dat_file:
		current_data = np.loadtxt(dat_file)

	current_data_min    = current_data.min()
	current_data_max    = current_data.max()
	current_data_mean   = current_data.mean()
	current_data_median = np.median(current_data)

	print(
		f'>  current_dataset     = {current_dataset}         \n' \
		f'>> current_data_min    = {current_data_min:.3f}    \n' \
		f'>> current_data_max    = {current_data_max:.3f}    \n' \
		f'>> current_data_mean   = {current_data_mean:.3f}   \n' \
		f'>> current_data_median = {current_data_median:.3f} \n')

