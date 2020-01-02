'''****************************************************************************
 * visualization.py: TODO
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


import os

import matplotlib.pyplot as plt

from matplotlib.pyplot import imsave
from tensorflow.keras  import Input, Model

from misc.misc import Hexnet_print


def visualize_filters(model, output_dir, verbosity_level=2):
	filter_to_visualize = 0

	os.makedirs(output_dir, exist_ok=True)

	for layer in model.layers:
		if not 'conv' in layer.name:
			continue

		filters = layer.get_weights()[0]

		if verbosity_level >= 2:
			Hexnet_print(f'(visualize_filters) layer={layer} (layer.name={layer.name}): filters.shape={filters.shape}')

		for channel in range(filters.shape[2]):
			filter_fname = os.path.join(output_dir, f'{layer.name}_filter{filter_to_visualize}_channel{channel}.png')
			filter       = filters[:, :, channel, filter_to_visualize]
			imsave(filter_fname, filter, cmap='viridis')


def visualize_feature_maps(
	model,
	test_classes,
	test_data,
	test_labels,
	output_dir,
	max_images_per_class = 10,
	verbosity_level      =  2):

	feature_map_to_visualize = 0

	os.makedirs(output_dir, exist_ok=True)

	sequential_model = model
	functional_model = None

	input_layer      = Input(batch_shape=sequential_model.layers[0].input_shape)
	following_layers = input_layer

	for layer in sequential_model.layers:
		following_layers = layer(following_layers)

		if not 'conv' in layer.name:
			continue

		functional_model = Model(input_layer, following_layers)

		if verbosity_level >= 3:
			functional_model.summary()

		feature_maps = functional_model.predict(test_data)

		if verbosity_level >= 2:
			Hexnet_print(f'(visualize_feature_maps) layer={layer} (layer.name={layer.name}): feature_maps.shape={feature_maps.shape}')

		class_counter_dict = dict.fromkeys(test_classes, 0)

		for image_counter, (feature_map, test_label) in enumerate(zip(feature_maps, test_labels)):
			if class_counter_dict[test_label] < max_images_per_class:
				feature_map_fname = os.path.join(output_dir, f'{layer.name}_fm{feature_map_to_visualize}_label{test_label}_image{image_counter}.png')
				feature_map       = feature_map[:, :, feature_map_to_visualize]
				imsave(feature_map_fname, feature_map, cmap='viridis')
				class_counter_dict[test_label] += 1


def visualize_model(
	model,
	test_classes,
	test_data,
	test_labels,
	output_dir,
	max_images_per_class = 10,
	verbosity_level      =  2):

	os.makedirs(output_dir, exist_ok=True)

	visualize_filters(model, output_dir, verbosity_level)

	visualize_feature_maps(
		model,
		test_classes,
		test_data,
		test_labels,
		output_dir,
		max_images_per_class,
		verbosity_level)


def visualize_results(history, title, output_dir, show_results):
	for key in history.history.keys():
		title_key = f'{title}_{key}'

		if output_dir is not None:
			with open(os.path.join(output_dir, f'{title_key}.dat'), 'w') as results_dat:
				for value in history.history[key]:
					print(value, file=results_dat)

		fig_title = f'model train {key}'
		plt.figure(fig_title)
		plt.title(fig_title)
		plt.xlabel('epoch')
		plt.ylabel(key)

		plt.plot(history.history[key])

		if output_dir is not None:
			results_fig = os.path.join(output_dir, title_key)
			plt.savefig(f'{results_fig}.png')
			plt.savefig(f'{results_fig}.pdf')

		if show_results:
			plt.show()


