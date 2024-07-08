#!/usr/bin/env python3.7


'''****************************************************************************
 * geometric_primitives.py: Primitive Generation
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
import math
import os

import numpy as np
import sympy as sp

from matplotlib.pyplot import imsave
from tqdm              import tqdm


################################################################################
# Parameters
################################################################################

output_dir = 'geometric_primitives'

window_size_factor =  0.1

max_filename_length = 50

populated_variables_dict = {}


################################################################################
# Rotate point by a given angle around a coordinate of the image
################################################################################

def rotate_point(point, angle = 90, origin = (0, 0), convert_to_radians = True):
	if convert_to_radians:
		angle = math.radians(angle)

	px, py = point
	ox, oy = origin

	qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
	qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)

	return (qx, qy)


################################################################################
# Calculate coordinates for function sampling
################################################################################

def populate_variables(function, symbol, window_size, step_size):
	populated_variables = []

	current_variable = -window_size_factor * window_size

	function_diff = sp.diff(function, symbol)

	while current_variable < window_size + window_size_factor * window_size:
		subs = {str(symbol): current_variable}

		result_diff = function_diff.evalf(subs=subs)

		if result_diff.is_imaginary:
			current_variable += step_size

			continue
		else:
			populated_variables.append(current_variable)

			result_diff = abs(result_diff)

			if result_diff > 1:
				current_variable += (step_size / result_diff / 2)
			else:
				current_variable += step_size

	return populated_variables


################################################################################
# Plot function via the square lattice format
################################################################################

def plot_function_square(
	function_s,
	symbol_s,
	figure_size      = 60,
	window_size      =  1,
	step_size        =  0.005,
	linewidth_factor =  0.01,
	rotation_degrees =  0,
	interpolation    = 'nearest neighbor'):

	nearest_neighbor_interpolation = ['nearest neighbor', 'nn']
	linear_interpolation           = ['linear',           'l']
	cubic_interpolation            = ['cubic',            'c']


	multiple_symbols = len(symbol_s) > 1

	function_s = [sp.sympify(function) for function in function_s]
	symbol_s   = sp.symbols(','.join(symbol_s))


	image = np.full(fill_value = 255, shape = (figure_size, figure_size))

	if multiple_symbols:
		variables = np.arange(-window_size_factor * window_size, window_size + window_size_factor * window_size, step_size).tolist()

		variables = itertools.product(variables, repeat=len(symbol_s))

	sampling_size = max(1, round(linewidth_factor * window_size * figure_size))


	pixels_in_bound = []

	for function in function_s:
		if not multiple_symbols:
			if f'{function}_{symbol_s}_{step_size}' not in populated_variables_dict:
				populated_variables = populate_variables(function, symbol_s, window_size, step_size)
				populated_variables_dict[f'{function}_{symbol_s}_{step_size}'] = populated_variables
			else:
				populated_variables = populated_variables_dict[f'{function}_{symbol_s}_{step_size}']
		else:
			populated_variables = variables

		for variable_s in populated_variables:
			if multiple_symbols:
				subs = {str(symbol): variable for symbol, variable in zip(symbol_s, variable_s)}
			else:
				subs = {str(symbol_s): variable_s}

			result = function.evalf(subs=subs)

			if multiple_symbols:
				x = variable_s[0]
			else:
				x = variable_s

			y = result

			if rotation_degrees:
				x, y = rotate_point(point = (x, y), angle = rotation_degrees, origin = (window_size / 2, window_size / 2))

			x = float(figure_size * x)
			y = float(figure_size * y)

			xr = round(x)
			yr = round(y)

			for yi in range(yr - sampling_size, yr + sampling_size + 1):
				y_diff = (y - yi)**2

				for xi in range(xr - sampling_size, xr + sampling_size + 1):
					x_diff = (x - xi)**2

					diff = math.sqrt(x_diff + y_diff)

					if diff <= sampling_size and \
					   xi >= 0 and xi < figure_size and yi >= 0 and yi < figure_size:
						if interpolation in nearest_neighbor_interpolation:
							v = 0
						elif interpolation in linear_interpolation:
							v = (diff / sampling_size) * 255
						else: # cubic_interpolation
							k = diff / sampling_size
							l = 2 * k**3 - 3 * k**2 + 1
							v = 255 - (l * 255)

						if v < image[figure_size - 1 - yi][xi]:
							image[figure_size - 1 - yi][xi] = v

						pixels_in_bound.append((yi, xi))


	return image, len(set(pixels_in_bound))


################################################################################
# Plot function via the hexagonal lattice format
################################################################################

def plot_function_hexagonal(
	function_s,
	symbol_s,
	figure_size      = 60,
	window_size      =  1,
	step_size        =  0.005,
	linewidth_factor =  0.01,
	rad_o            =  0.63,
	rotation_degrees =  0,
	interpolation    = 'nearest neighbor'):

	nearest_neighbor_interpolation = ['nearest neighbor', 'nn']
	linear_interpolation           = ['linear',           'l']
	cubic_interpolation            = ['cubic',            'c']


	s_height = figure_size
	s_width  = figure_size


	if s_height > rad_o:
		h_width  = math.ceil(s_width  / (math.sqrt(3) * rad_o))
		h_height = math.ceil(s_height / (1.5 * rad_o))
	else:
		h_width  = math.ceil(s_width / (math.sqrt(3) * rad_o))
		h_height = 1


	width_sq  = s_width
	height_sq = s_height


	h_rad_i  = (math.sqrt(3) / 2) * rad_o
	h_dia_o  = 2 * rad_o
	h_dia_i  = 2 * h_rad_i
	h_dist_w = h_dia_i
	h_dist_h = 1.5 * rad_o

	if h_height > 1:
		width_hex  = h_width * h_dia_i + h_rad_i
		height_hex = h_dia_o + (h_height - 1) * h_dist_h
	else:
		width_hex  = h_width * h_dia_i
		height_hex = h_dia_o


	wb = (width_hex  - width_sq)  / 2
	hb = (height_hex - height_sq) / 2




	multiple_symbols = len(symbol_s) > 1

	function_s = [sp.sympify(function) for function in function_s]
	symbol_s   = sp.symbols(','.join(symbol_s))


	image = np.full(fill_value = 255, shape = (h_height, h_width))

	if multiple_symbols:
		variables = np.arange(
			-wb / figure_size - window_size_factor * window_size,
			window_size + wb / figure_size + window_size_factor * window_size,
			step_size).tolist()

		variables = itertools.product(variables, repeat=len(symbol_s))

	sampling_size = max(1, round(linewidth_factor * window_size * figure_size))


	pixels_in_bound = []

	for function in function_s:
		if not multiple_symbols:
			if f'{function}_{symbol_s}_{step_size}' not in populated_variables_dict:
				populated_variables = populate_variables(function, symbol_s, window_size, step_size)
				populated_variables_dict[f'{function}_{symbol_s}_{step_size}'] = populated_variables
			else:
				populated_variables = populated_variables_dict[f'{function}_{symbol_s}_{step_size}']
		else:
			populated_variables = variables

		for variable_s in populated_variables:
			if multiple_symbols:
				subs = {str(symbol): variable for symbol, variable in zip(symbol_s, variable_s)}
			else:
				subs = {str(symbol_s): variable_s}

			result = function.evalf(subs=subs)

			if multiple_symbols:
				x = variable_s[0]
			else:
				x = variable_s

			y = result

			if rotation_degrees:
				x, y = rotate_point(point = (x, y), angle = rotation_degrees, origin = (window_size / 2, window_size / 2))

			x = float(width_hex  * x)
			y = float(height_hex * y)


			if 1 - (h_height - 1 - y) % 2:
				xi = round(x / h_dist_w)
			else:
				xi = round((x - h_rad_i) / h_dist_w)

			yi = round(y / h_dist_h)


			for yik in range(yi - sampling_size, yi + sampling_size + 1):
				yiks   = -hb + yik * h_dist_h
				y_diff = (y - yiks)**2

				for xik in range(xi - sampling_size, xi + sampling_size + 1):
					if 1 - (h_height - 1 - yik) % 2:
						xiks = -wb + xik * h_dist_w
					else:
						xiks = -wb + h_rad_i + xik * h_dist_w

					x_diff = (x - xiks)**2

					diff = math.sqrt(x_diff + y_diff)

					if diff <= sampling_size and \
					   xik >= 0 and xik < h_width and yik >= 0 and yik < h_height:
						if interpolation in nearest_neighbor_interpolation:
							v = 0
						elif interpolation in linear_interpolation:
							v = (diff / sampling_size) * 255
						else: # cubic_interpolation
							k = diff / sampling_size
							l = 2 * k**3 - 3 * k**2 + 1
							v = 255 - (l * 255)

						if v < image[h_height - 1 - yik][xik]:
							image[h_height - 1 - yik][xik] = v

						pixels_in_bound.append((yik, xik))


	return image, len(set(pixels_in_bound))


################################################################################
# Plot function via the square and the hexagonal lattice format
################################################################################

def plot_function(
	plot_function_function,
	function_s,
	symbol_s,
	figure_size      = 60,
	window_size      =  1,
	step_size        =  0.005,
	linewidth_factor =  0.01,
	rad_o            =  0.63,
	rotation_degrees =  0,
	interpolation    = 'nearest neighbor',
	output_dir       = output_dir,
	filename         = None):

	if plot_function_function is plot_function_square:
		image, pixels_in_bound_cnt = plot_function_function(
			function_s,
			symbol_s,
			figure_size,
			window_size,
			step_size,
			linewidth_factor,
			rotation_degrees,
			interpolation)
	else: # plot_function_hexagonal
		image, pixels_in_bound_cnt = plot_function_function(
			function_s,
			symbol_s,
			figure_size,
			window_size,
			step_size,
			linewidth_factor,
			rad_o,
			rotation_degrees,
			interpolation)


	if not filename:
		function_s_replace_what_with = [['*', ''], ['/', '_div_']]

		filename = ','.join(function_s)

		for replace_what, replace_with in function_s_replace_what_with:
			filename = filename.replace(replace_what, replace_with)

		if len(filename) > max_filename_length:
			filename = f'{filename[:max_filename_length]}'

	filename = f'{filename}_pib{pixels_in_bound_cnt}'

	if plot_function_function is plot_function_square:
		image_filename = os.path.join(output_dir, f'{filename}_sq.png')
	else: # plot_function_hexagonal
		image_filename = os.path.join(output_dir, f'{filename}_hex.png')


	imsave(image_filename, image, vmin=0, vmax=255, cmap='gray')


	return image, pixels_in_bound_cnt, filename


################################################################################
# Test the square and the hexagonal plot function
################################################################################

def test_plot_functions():
	plot_functions_functions = (plot_function_square, plot_function_hexagonal)


	functions = [
		['x'],
		['sqrt(x)'],
		['1/4*(2-sqrt(-3+16*x-16*x^2))', '1/4*(2+sqrt(-3+16*x-16*x^2))'],
		[f'-x+1+({i:.2f})' for i in np.arange(-0.4, 0.5, 0.1)] + [f'x+({i:.2f})' for i in np.arange(-0.4, 0.5, 0.1)],
		[f'-sqrt(x)+1+({i:.2f})' for i in np.arange(-0.4, 0.5, 0.1)] + [f'sqrt(x)+({i:.2f})' for i in np.arange(-0.4, 0.5, 0.1)],
		['x'] + ['sqrt(x)'] + ['1/4*(2-sqrt(-3+16*x-16*x^2))', '1/4*(2+sqrt(-3+16*x-16*x^2))']
	]

	symbols           = [['x']]
	figure_sizes      = [60]
	window_sizes      = [1]
	step_sizes        = [0.005]
	linewidth_factors = [0.01]

	function_modifiers = [f'{i:.2f}*' for i in np.arange(0.1, 2.1, 0.1)]


	functions_len = len(functions)

	if len(symbols)           < functions_len: symbols           += (functions_len - len(symbols))           * [symbols[0]]
	if len(figure_sizes)      < functions_len: figure_sizes      += (functions_len - len(figure_sizes))      * [figure_sizes[0]]
	if len(window_sizes)      < functions_len: window_sizes      += (functions_len - len(window_sizes))      * [window_sizes[0]]
	if len(step_sizes)        < functions_len: step_sizes        += (functions_len - len(step_sizes))        * [step_sizes[0]]
	if len(linewidth_factors) < functions_len: linewidth_factors += (functions_len - len(linewidth_factors)) * [linewidth_factors[0]]


	for function_s, symbol_s, figure_size, window_size, step_size, linewidth_factor in \
	    zip(functions, symbols, figure_sizes, window_sizes, step_sizes, linewidth_factors):

		print(f'> function_s={function_s}')

		for plot_function_function in plot_functions_functions:
			print(f'\t> plot_function_function={plot_function_function.__name__}')

			pixels_in_bound_cnt_list = []

			if function_modifiers:
				for function_modifier in tqdm(function_modifiers):
					modified_function_s = [f'{function_modifier}({function})' for function in function_s]

					_, pixels_in_bound_cnt, _ = plot_function(
						plot_function_function,
						modified_function_s,
						symbol_s,
						figure_size,
						window_size,
						step_size,
						linewidth_factor,
						output_dir = output_dir)

					pixels_in_bound_cnt_list.append(pixels_in_bound_cnt)
			else:
				_, pixels_in_bound_cnt, _ = plot_function(
					plot_function_function,
					modified_function_s,
					symbol_s,
					figure_size,
					window_size,
					step_size,
					linewidth_factor,
					output_dir = output_dir)

				pixels_in_bound_cnt_list.append(pixels_in_bound_cnt)

			mean_total_pixels = sum(pixels_in_bound_cnt_list) / len(pixels_in_bound_cnt_list)
			print(f'\t\t> mean total pixels: {mean_total_pixels}')


################################################################################
# main
################################################################################

if __name__ == '__main__':
	os.makedirs(output_dir, exist_ok=True)

	test_plot_functions()


