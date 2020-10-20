#!/usr/bin/env python3.7


import itertools
import math
import os

import numpy as np
import sympy as sp

from matplotlib.pyplot import imsave
from tqdm              import tqdm


output_dir = 'geometric_primitives_image_generation'

function_s_in_bound_threshold = 128


os.makedirs(output_dir, exist_ok=True)


def plot_function_square(
	function_s,
	symbol_s,
	figure_size      = 64,
	window_size      =  1,
	step_size        =  0.001,
	linewidth_factor =  0.01):

	multiple_symbols = len(symbol_s) > 1

	function_s = [sp.sympify(function) for function in function_s]
	symbol_s   = sp.symbols(','.join(symbol_s), real=True)


	image = np.full(fill_value = 255, shape = (figure_size, figure_size))

	variables = np.arange(0, window_size, step_size)

	if multiple_symbols:
		variables = itertools.product(variables, repeat=len(symbol_s))

	sampling_size = int(linewidth_factor * window_size * figure_size)


	function_s_in_bound_cnt = 0

	for function in function_s:
		for variable_s in variables:
			if multiple_symbols:
				subs = {str(symbol): variable for symbol, variable in zip(symbol_s, variable_s)}
			else:
				subs = {str(symbol_s): variable_s}

			result = function.evalf(subs=subs)

			if result.has(sp.I):
				continue

			if multiple_symbols:
				x = int(figure_size * variable_s[0])
			else:
				x = int(figure_size * variable_s)

			y = int(figure_size * result)

			for yi in range(y - sampling_size, y + sampling_size + 1):
				y_diff = (y - yi)**2

				for xi in range(x - sampling_size, x + sampling_size + 1):
					x_diff = (x - xi)**2

					if math.sqrt(x_diff + y_diff) <= sampling_size and \
					   xi >= 0 and xi < figure_size and yi >= 0 and yi < figure_size:
						image[figure_size - 1 - yi][xi]  = 0
						function_s_in_bound_cnt         += 1


	return image, function_s_in_bound_cnt


def plot_function_hexagonal(
	function_s,
	symbol_s,
	figure_size      = 64,
	window_size      =  1,
	step_size        =  0.001,
	linewidth_factor =  0.01,
	rad_o            =  0.63):

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
	symbol_s   = sp.symbols(','.join(symbol_s), real=True)


	image = np.full(fill_value = 255, shape = (h_height, h_width))

	variables = np.arange(-wb / figure_size, window_size + wb / figure_size, step_size)

	if multiple_symbols:
		variables = itertools.product(variables, repeat=len(symbol_s))

	sampling_size = int(linewidth_factor * window_size * figure_size)


	function_s_in_bound_cnt = 0

	for function in function_s:
		for variable_s in variables:
			if multiple_symbols:
				subs = {str(symbol): variable for symbol, variable in zip(symbol_s, variable_s)}
			else:
				subs = {str(symbol_s): variable_s}

			result = function.evalf(subs=subs)

			if result.has(sp.I):
				continue

			if multiple_symbols:
				x = float(width_hex * variable_s[0])
			else:
				x = float(width_hex * variable_s)

			y = float(height_hex * result)


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

					if math.sqrt(x_diff + y_diff) <= sampling_size and \
					   xik >= 0 and xik < h_width and yik >= 0 and yik < h_height:
						image[h_height - 1 - yik][xik]  = 0
						function_s_in_bound_cnt        += 1


	return image, function_s_in_bound_cnt


def plot_function(
	plot_function_function,
	function_s,
	symbol_s,
	figure_size      = 64,
	window_size      =  1,
	step_size        =  0.001,
	linewidth_factor =  0.01):

	function_s_replace_what_with = [['*', '']]


	image, function_s_in_bound_cnt = plot_function_function(
		function_s,
		symbol_s,
		figure_size,
		window_size,
		step_size,
		linewidth_factor)

	if function_s_in_bound_cnt > function_s_in_bound_threshold:
		function_s = ','.join(function_s)

		for replace_what, replace_with in function_s_replace_what_with:
			function_s = function_s.replace(replace_what, replace_with)

		if plot_function_function is plot_function_square:
			image_filename = os.path.join(output_dir, f'{function_s}.png')
		else: # plot_function_hexagonal
			image_filename = os.path.join(output_dir, f'{function_s}_hex.png')

		imsave(image_filename, image, vmin=0, vmax=255, cmap='gray')


def test_plot_functions():
	plot_functions_functions = (plot_function_square, plot_function_hexagonal)


	functions = [
		['x'], ['x+0.5'],
		['sqrt(x)'], ['sqrt(x)+0.5'],
		['x^2'], ['x^2+0.5'],
		['-sqrt(1-x**2)', 'sqrt(1-x**2)'], ['-sqrt(1-x**2)+0.5', 'sqrt(1-x**2)+0.5']
	]

	symbols           = [['x']]
	figure_sizes      = [64]
	window_sizes      = [1]
	step_sizes        = [0.001]
	linewidth_factors = [0.02]

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

			if function_modifiers:
				for function_modifier in tqdm(function_modifiers):
					modified_function_s = [f'{function_modifier}({function})' for function in function_s]

					plot_function(
						plot_function_function,
						modified_function_s,
						symbol_s,
						figure_size,
						window_size,
						step_size,
						linewidth_factor)
			else:
				plot_function(
					plot_function_function,
					modified_function_s,
					symbol_s,
					figure_size,
					window_size,
					step_size,
					linewidth_factor)


test_plot_functions()


