#!/usr/bin/env python3.7


'''****************************************************************************
 * geometric_primitives_dataset.py: Dataset Generation
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

import itertools
import os
import random

import numpy as np

from joblib import delayed, Parallel
from tqdm   import tqdm

from geometric_primitives import plot_function, plot_function_hexagonal, plot_function_square


################################################################################
# Parameters
################################################################################

enable_randomized_augmentation = False

enable_test_mode = False

output_dir = 'geometric_primitives_dataset'

max_filename_length = 50


################################################################################
# Dataset class for primitive generation
################################################################################

class dataset:
	def __init__(
		self,
		plot_function_function,
		functions,
		symbols,
		figure_size        = 60,
		window_size        =  1,
		step_sizes         = [0.005],
		linewidth_factors  = [0.02],
		rad_o              = 0.63,
		rotation_degrees   = [0],
		function_modifiers = [None],
		translations       = [None],
		output_dir         = output_dir):

		self.plot_function_function = plot_function_function
		self.functions              = list(functions.values())
		self.function_labels        = list(functions.keys())
		self.symbols                = symbols
		self.figure_size            = figure_size
		self.window_size            = window_size
		self.step_sizes             = step_sizes
		self.linewidth_factors      = linewidth_factors
		self.rad_o                  = rad_o
		self.rotation_degrees       = rotation_degrees
		self.function_modifiers     = function_modifiers
		self.translations           = translations
		self.output_dir             = output_dir

		if self.plot_function_function is plot_function_square:
			self.type = 'square'
		else: # plot_function_hexagonal
			self.type = 'hexagonal'

		self.title      = f'{self.type}_{self.figure_size}x{self.figure_size}'
		self.output_dir = os.path.join(self.output_dir, self.title)


	def _generate(self, function_s, symbol_s, function_label):
		function_s_replace_what_with = [['*', ''], ['/', '_div_']]

		image_index = 0

		output_dir_class = os.path.join(self.output_dir, function_label)
		os.makedirs(output_dir_class, exist_ok=True)

		data_file = open(os.path.join(self.output_dir, f'{function_label}.csv'), 'w')
		print('filename,function(s),symbol(s),figure size,window size,step size,linewidth,rotation,pixels in bound', file=data_file)

		for function_modifier in tqdm(self.function_modifiers, desc='function modifiers'):
			if function_modifier:
				modified_function_s = [f'{function_modifier}({function})' for function in function_s]
			else:
				modified_function_s = function_s.copy()

			for translation in tqdm(self.translations, desc='translations'):
				if translation:
					translated_function_s = [f'{function.replace("x", f"(x-({translation[0]}))")}+({translation[1]})' for \
						function in modified_function_s]
				else:
					translated_function_s = modified_function_s.copy()

				filename_translation = ','.join(translated_function_s)

				for replace_what, replace_with in function_s_replace_what_with:
					filename_translation = filename_translation.replace(replace_what, replace_with)

				if len(filename_translation) > max_filename_length:
					filename_translation = f'{filename_translation[:max_filename_length]}_'

				for step_size in tqdm(self.step_sizes, desc = 'step sizes'):
					filename_step_size = f'{filename_translation}_ss{step_size:.4f}'

					for linewidth_factor in tqdm(self.linewidth_factors, desc = 'linewidth factors'):
						filename_linewidth_factor = f'{filename_step_size}_lf{linewidth_factor:.3f}'

						for rotation_degrees in tqdm(self.rotation_degrees, desc = 'rotation degrees'):
							filename = f'{image_index}_{filename_linewidth_factor}_deg{rotation_degrees:.2f}'

							_, pixels_in_bound_cnt, filename = plot_function(
								self.plot_function_function,
								translated_function_s,
								symbol_s,
								self.figure_size,
								self.window_size,
								step_size,
								linewidth_factor,
								self.rad_o,
								rotation_degrees,
								output_dir = output_dir_class,
								filename   = filename)

							print(
								f'"{filename}",'
								f'{translated_function_s},'
								f'{symbol_s},'
								f'{self.figure_size},'
								f'{self.window_size},'
								f'{step_size},'
								f'{linewidth_factor},'
								f'{rotation_degrees},'
								f'{pixels_in_bound_cnt}',
								file=data_file)

							image_index += 1

		data_file.close()


	def generate(self):
		os.makedirs(self.output_dir, exist_ok=True)

		Parallel(n_jobs=-1, verbose=11)(delayed(self._generate)(function_s, symbol_s, function_label) for \
			function_s, symbol_s, function_label in tqdm(zip(self.functions, self.symbols, self.function_labels), desc = 'functions and symbols'))


def random_uniform(a, b, format_string='.2f'):
	return float(format(random.uniform(a, b), format_string))


################################################################################
# main
################################################################################

if __name__ == '__main__':
	plot_functions_functions = (plot_function_square, plot_function_hexagonal)


	if not enable_test_mode:
		functions = {
			'1_lines'                 : ['x'],
			'2_curves'                : ['sqrt(x)'],
			'3_ellipses'              : ['1/4*(2-sqrt(-3+16*x-16*x^2))', '1/4*(2+sqrt(-3+16*x-16*x^2))'],
			'4_line-based_grids'      : [f'-x+1+({i:.2f})' for i in np.arange(-0.4, 0.5, 0.1)] + [f'x+({i:.2f})' for i in np.arange(-0.4, 0.5, 0.1)],
			'5_curve-based_grids'     : [f'-sqrt(x)+1+({i:.2f})' for i in np.arange(-0.4, 0.5, 0.1)] + [f'sqrt(x)+({i:.2f})' for i in np.arange(-0.4, 0.5, 0.1)],
			'6_lines_curves_ellipses' : ['x'] + ['sqrt(x)'] + ['1/4*(2-sqrt(-3+16*x-16*x^2))', '1/4*(2+sqrt(-3+16*x-16*x^2))']
		}

		symbols      = len(functions) * [['x']]
		figure_sizes = [60, 80, 100, 200]
		window_size  = 1

		if not enable_randomized_augmentation:
			step_sizes         = np.arange(0.001, 0.006, 0.001)
			linewidth_factors  = np.arange(0.01, 0.06, 0.01)
			rotation_degrees   = range(0, 301, 60)
			function_modifiers = [None, '1.2*']
			translations       = [(f'{i[0]:.2f}', f'{i[1]:.2f}') for i in itertools.product(np.arange(-0.25, 0.3, 0.25), repeat=2)]
		else:
			step_sizes         = np.arange(random_uniform(0.0007, 0.0013, '.4f'), random_uniform(0.003, 0.009, '.4f'), random_uniform(0.0007, 0.0013, '.4f'))
			linewidth_factors  = np.arange(random_uniform(0.007, 0.013, '.3f'), random_uniform(0.03, 0.09, '.3f'), random_uniform(0.007, 0.013, '.3f'))
			rotation_degrees   = range(random_uniform(0, 59), random_uniform(301, 359), random_uniform(30, 90))
			function_modifiers = [f'{i:.2f}*' for i in np.arange(random_uniform(0.6, 1.0), random_uniform(1.1, 1.5), random_uniform(0.07, 0.13))]

			translations = [(f'{i[0]:.2f}', f'{i[1]:.2f}') for \
				i in itertools.product(np.arange(random_uniform(-0.7, -0.3), random_uniform(0.4, 0.8), random_uniform(0.3, 0.7)), repeat=2)]
	else:
		functions = {
			'1_lines'                 : ['x'],
			'2_curves'                : ['sqrt(x)'],
			'3_ellipses'              : ['1/4*(2-sqrt(-3+16*x-16*x^2))', '1/4*(2+sqrt(-3+16*x-16*x^2))'],
			'4_line-based_grids'      : [f'-x+1+({i:.2f})' for i in np.arange(-0.4, 0.5, 0.1)] + [f'x+({i:.2f})' for i in np.arange(-0.4, 0.5, 0.1)],
			'5_curve-based_grids'     : [f'-sqrt(x)+1+({i:.2f})' for i in np.arange(-0.4, 0.5, 0.1)] + [f'sqrt(x)+({i:.2f})' for i in np.arange(-0.4, 0.5, 0.1)],
			'6_lines_curves_ellipses' : ['x'] + ['sqrt(x)'] + ['1/4*(2-sqrt(-3+16*x-16*x^2))', '1/4*(2+sqrt(-3+16*x-16*x^2))']
		}

		symbols            = len(functions) * [['x']]
		figure_sizes       = [60, 80]
		window_size        = 1
		step_sizes         = [0.001, 0.002]
		linewidth_factors  = [0.01, 0.02]
		rotation_degrees   = [0, 90]
		function_modifiers = [None, '1.2*']
		translations       = [None, (0.25, 0.25)]


	datasets     = len(figure_sizes) * len(plot_functions_functions) * [None]
	datasets_cnt = 0


	print('> Dataset initialization')

	for figure_size in figure_sizes:
		print(f'\t> figure_size={figure_size}')

		for plot_function_function in plot_functions_functions:
			print(f'\t\t> plot_function_function={plot_function_function.__name__}')

			datasets[datasets_cnt] = dataset(
				plot_function_function,
				functions,
				symbols,
				figure_size,
				window_size,
				step_sizes,
				linewidth_factors,
				rotation_degrees   = rotation_degrees,
				function_modifiers = function_modifiers,
				translations       = translations,
				output_dir         = output_dir)

			datasets_cnt += 1


	print('> Dataset generation')

	for dataset in datasets:
		print(f'\t> dataset.output_dir={dataset.output_dir}')

		dataset.generate()

