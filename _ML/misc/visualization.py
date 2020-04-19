'''****************************************************************************
 * visualization.py: Filters, Feature Maps, and Results Visualization
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
from tensorflow.keras  import Model

from misc.misc import Hexnet_print


def visualize_filters(model, output_dir, verbosity_level=2):
	filter_to_visualize = 0

	os.makedirs(output_dir, exist_ok=True)

	for layer_counter, layer in enumerate(model.layers):
		if not 'conv' in layer.name:
			continue

		filters = layer.get_weights()[0]

		if verbosity_level >= 2:
			Hexnet_print(f'(visualize_filters) layer={layer} (layer_counter={layer_counter}, layer.name={layer.name}): filters.shape={filters.shape}')

		for channel in range(filters.shape[2]):
			filter_filename = os.path.join(output_dir, f'layer{str(layer_counter).zfill(3)}_{layer.name}_filter{filter_to_visualize}_channel{channel}.png')
			filter          = filters[:, :, channel, filter_to_visualize]
			imsave(filter_filename, filter, cmap='viridis')


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

	layers_outputs = [layer.output for layer in model.layers]

	if verbosity_level >= 3:
		Hexnet_print(f'(visualize_feature_maps) layers_outputs={layers_outputs}')

	model_outputs = Model(inputs=model.input, outputs=layers_outputs)

	if verbosity_level >= 3:
		Hexnet_print(f'(visualize_feature_maps) model_outputs={model_outputs}')

	predictions = model_outputs.predict(test_data)

	if verbosity_level >= 3:
		Hexnet_print(f'(visualize_feature_maps) predictions={predictions}')

	for layer_counter, (layer, feature_maps) in enumerate(zip(model.layers, predictions)):
		if feature_maps.ndim != 4:
			continue

		if verbosity_level >= 2:
			Hexnet_print(f'(visualize_feature_maps) layer={layer} (layer_counter={layer_counter}, layer.name={layer.name}): feature_maps.shape={feature_maps.shape}')

		class_counter_dict = dict.fromkeys(test_classes, 0)

		for feature_map_counter, (feature_map, label) in enumerate(zip(feature_maps, test_labels)):
			if class_counter_dict[label] < max_images_per_class:
				feature_map_filename = os.path.join(output_dir, f'layer{str(layer_counter).zfill(3)}_{layer.name}_fm{feature_map_to_visualize}_label{label}_image{feature_map_counter}.png')
				feature_map          = feature_map[:, :, feature_map_to_visualize]
				imsave(feature_map_filename, feature_map, cmap='viridis')
				class_counter_dict[label] += 1


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
	if output_dir is not None:
		os.makedirs(output_dir, exist_ok=True)

	keys = history.history.keys()

	nrows = 1
	ncols = len([None for key in keys if not 'val_' in key])
	index = 1

	plt.figure('Test results')
	plt.subplots_adjust(wspace=0.5)

	for key in keys:
		title_key = f'{title}_{key}'

		if output_dir is not None:
			with open(os.path.join(output_dir, f'{title_key}.dat'), 'w') as results_dat:
				for value in history.history[key]:
					print(value, file=results_dat)

		if 'val_' in key:
			continue

		plt.subplot(nrows, ncols, index)
		plt.title(f'model train {key}')
		plt.xlabel('epoch')
		plt.ylabel(key)

		for key_to_plot in keys:
			if key in key_to_plot:
				plt.plot(history.history[key_to_plot], label=key_to_plot)

		plt.legend()

		index += 1

	if output_dir is not None:
		results_fig = os.path.join(output_dir, title)
		plt.savefig(f'{results_fig}.png')
		plt.savefig(f'{results_fig}.pdf')

	if show_results:
		plt.show()

	plt.close()

