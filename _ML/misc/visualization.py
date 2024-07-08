'''****************************************************************************
 * visualization.py: Model (Filters, etc.) and Results Visualization
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

import cv2
import math
import os
import sklearn.metrics

import matplotlib.pyplot as plt
from   matplotlib.collections import PatchCollection
from   matplotlib.patches     import RegularPolygon

import numpy      as np
import pandas     as pd
import seaborn    as sn
import tensorflow as tf

from matplotlib.pyplot import imsave
from pprint            import pprint
from tensorflow.keras  import Model
from tqdm              import tqdm

from misc.misc import Hexnet_print, normalize_array


################################################################################
# Visualize array as raster and vector graphics
################################################################################

def visualize_array(array, title, colormap=None, visualize_axes=True, show_array=False, rad_o=0.71):
	axes_border_width = 0.05


	# Construct array

	array_width  = array.shape[1]
	array_height = array.shape[0]

	array_x = [w for h in range(array_height) for w in range(array_width)]
	array_y = [h for h in range(array_height) for w in range(array_width)]

	array_x = [rad_o**2 + x for x in array_x]
	array_y = [rad_o**2 + y for y in array_y]

	if colormap is not None:
		array = plt.cm.get_cmap(colormap)(array)

	array_colors = np.reshape(array, newshape = (array.shape[0] * array.shape[1], array.shape[2]))


	# Set Matplotlib window title

	fig = plt.figure()
	man = plt.get_current_fig_manager()
	man.set_window_title(os.path.basename(title))


	# Adjust axes

	ax = plt.subplot(aspect='equal')

	ax.axis(
		(min(array_x) - 1,
		 max(array_x) + 1,
		 min(array_y) - 1,
		 max(array_y) + 1)
	)

	ax.invert_yaxis()


	# Calculate squares

	orientation = math.radians(45)

	patches_list = [RegularPolygon((x, y), numVertices=4, radius=rad_o, color=c, linewidth=0.1, orientation=orientation) for x, y, c in zip(array_x, array_y, array_colors)]


	# Visualize squares

	patch_collection = PatchCollection(patches_list, match_original=True)
	patch_collection = ax.add_collection(patch_collection)
	ax.axis('off')

	array_fig = f'{title}_sq'
	plt.savefig(f'{array_fig}.png', bbox_inches='tight')
	plt.savefig(f'{array_fig}.pdf', bbox_inches='tight')

	if show_array and not visualize_axes:
		plt.show()


	# Visualize squares with axes

	if visualize_axes:
		patch_collection.remove()


		axes_border_width_x = axes_border_width * max(array_x)
		axes_border_width_y = axes_border_width * max(array_y)

		ax.axis(
			(min(array_x) - axes_border_width_x,
			 max(array_x) + axes_border_width_x,
			 min(array_y) - axes_border_width_y,
			 max(array_y) + axes_border_width_y)
		)

		ax.invert_yaxis()


		patch_collection = PatchCollection(patches_list, match_original=True, edgecolor='black')
		ax.add_collection(patch_collection)
		ax.axis('on')

		array_fig = f'{array_fig}_with_axes'
		plt.savefig(f'{array_fig}.png', bbox_inches='tight')
		plt.savefig(f'{array_fig}.pdf', bbox_inches='tight')

		if show_array:
			plt.show()


	# Close

	plt.close()


################################################################################
# Visualize Hexarray as raster and vector graphics
################################################################################

def visualize_hexarray(hexarray, title, colormap=None, visualize_axes=True, show_hexarray=False, rad_o=1.0, figure_size=60):
	axes_border_width = 0.05


	# Construct Hexarray

	hexarray_width  = hexarray.shape[1]
	hexarray_height = hexarray.shape[0]

	hexarray_x = [w * math.sqrt(3) if not h % 2 else math.sqrt(3) / 2 + w * math.sqrt(3) for h in range(hexarray_height) for w in range(hexarray_width)]
	hexarray_y = [1.5 * h for h in range(hexarray_height) for w in range(hexarray_width)]

	# hexarray_x = [(figure_size - rad_o * max(hexarray_x)) / 2 + rad_o * x for x in hexarray_x]
	# hexarray_y = [(figure_size - rad_o * max(hexarray_y)) / 2 + rad_o * y for y in hexarray_y]

	if colormap is not None:
		hexarray = plt.cm.get_cmap(colormap)(hexarray)

	hexarray_colors = np.reshape(hexarray, newshape = (hexarray.shape[0] * hexarray.shape[1], hexarray.shape[2]))


	# Set Matplotlib window title

	fig = plt.figure()
	man = plt.get_current_fig_manager()
	man.set_window_title(os.path.basename(title))


	# Adjust axes

	ax = plt.subplot(aspect='equal')

	ax.axis(
		(min(hexarray_x) - 1,
		 max(hexarray_x) + 1,
		 min(hexarray_y) - 1,
		 max(hexarray_y) + 1)
	)

	ax.invert_yaxis()


	# Calculate hexagons

	patches_list = [RegularPolygon((x, y), numVertices=6, radius=rad_o, color=c, linewidth=0.1) for x, y, c in zip(hexarray_x, hexarray_y, hexarray_colors)]


	# Visualize hexagons

	patch_collection = PatchCollection(patches_list, match_original=True)
	patch_collection = ax.add_collection(patch_collection)
	ax.axis('off')

	hexarray_fig = f'{title}_hex'
	plt.savefig(f'{hexarray_fig}.png', bbox_inches='tight')
	plt.savefig(f'{hexarray_fig}.pdf', bbox_inches='tight')

	if show_hexarray and not visualize_axes:
		plt.show()


	# Visualize hexagons with axes

	if visualize_axes:
		patch_collection.remove()


		axes_border_width_x = axes_border_width * max(hexarray_x)
		axes_border_width_y = axes_border_width * max(hexarray_y)

		ax.axis(
			(min(hexarray_x) - axes_border_width_x,
			 max(hexarray_x) + axes_border_width_x,
			 min(hexarray_y) - axes_border_width_y,
			 max(hexarray_y) + axes_border_width_y)
		)

		ax.invert_yaxis()


		patch_collection = PatchCollection(patches_list, match_original=True, edgecolor='black')
		ax.add_collection(patch_collection)
		ax.axis('on')

		hexarray_fig = f'{hexarray_fig}_with_axes'
		plt.savefig(f'{hexarray_fig}.png', bbox_inches='tight')
		plt.savefig(f'{hexarray_fig}.pdf', bbox_inches='tight')

		if show_hexarray:
			plt.show()


	# Close

	plt.close()


################################################################################
# Visualize filters via function visualize_hexarray()
################################################################################

def visualize_filters(
	model,
	colormap,
	visualize_hexagonal,
	output_dir,
	verbosity_level = 2):

	filter_to_visualize = 0

	os.makedirs(output_dir, exist_ok=True)

	for layer_counter, layer in enumerate(model.layers):
		if 'conv' not in layer.name:
			continue

		filters = layer.get_weights()[0]

		if verbosity_level >= 2:
			Hexnet_print(f'(visualize_filters) layer={layer} (layer_counter={layer_counter}, layer.name={layer.name}): filters.shape={filters.shape}')

		for channel in tqdm(range(filters.shape[2])):
			filter_filename = os.path.join(output_dir, f'layer{str(layer_counter).zfill(3)}_{layer.name}_channel{str(channel).zfill(3)}_filter{filter_to_visualize}')
			filter          = normalize_array(filters[:, :, channel, filter_to_visualize])

			imsave(f'{filter_filename}.png', filter, cmap=colormap)

			if visualize_hexagonal:
				visualize_hexarray(filter, filter_filename, colormap)


################################################################################
# Visualize feature maps via function visualize_hexarray()
################################################################################

def visualize_feature_maps(
	model,
	test_classes,
	test_data,
	test_filenames,
	test_labels,
	colormap,
	visualize_hexagonal,
	output_dir,
	max_images_per_class = 10,
	verbosity_level      =  2):

	feature_map_to_visualize = 0

	os.makedirs(output_dir, exist_ok=True)

	test_data_for_prediction   = []
	test_labels_for_prediction = []

	class_counter_dict = dict.fromkeys(test_classes, 0)

	for image, filename, label in zip(test_data, test_filenames, test_labels):
		if class_counter_dict[label] < max_images_per_class:
			test_data_for_prediction.append(image)
			test_labels_for_prediction.append(f'{label}_{filename}')
			class_counter_dict[label] += 1

	test_data_for_prediction = np.asarray(test_data_for_prediction)

	layers_outputs = [layer.output for layer in model.layers]

	if verbosity_level >= 3:
		Hexnet_print(f'(visualize_feature_maps) layers_outputs={layers_outputs}')

	model_outputs = Model(inputs=model.input, outputs=layers_outputs)

	if verbosity_level >= 3:
		Hexnet_print(f'(visualize_feature_maps) model_outputs={model_outputs}')

	predictions = model_outputs.predict(test_data_for_prediction)

	if verbosity_level >= 3:
		Hexnet_print(f'(visualize_feature_maps) predictions={predictions}')

	for layer_counter, (layer, feature_maps) in enumerate(zip(model.layers, predictions)):
		if feature_maps.ndim != 4:
			continue

		if verbosity_level >= 2:
			Hexnet_print(f'(visualize_feature_maps) layer={layer} (layer_counter={layer_counter}, layer.name={layer.name}): feature_maps.shape={feature_maps.shape}')

		for feature_map_counter, (feature_map, label) in enumerate(zip(tqdm(feature_maps), test_labels_for_prediction)):
			title = f'layer{str(layer_counter).zfill(3)}_{layer.name}_label{str(label).zfill(3)}_image{str(feature_map_counter).zfill(3)}_featuremap{feature_map_to_visualize}'
			feature_map_filename = os.path.join(output_dir, title)
			feature_map          = normalize_array(feature_map[:, :, feature_map_to_visualize])

			imsave(f'{feature_map_filename}.png', feature_map, cmap=colormap)

			if visualize_hexagonal:
				visualize_hexarray(feature_map, feature_map_filename, colormap)


################################################################################
# Visualize activations via function visualize_hexarray()
################################################################################

def visualize_activations(
	model,
	test_classes,
	test_data,
	test_data_orig,
	test_filenames,
	test_labels,
	colormap,
	visualize_hexagonal,
	output_dir,
	max_images_per_class = 10,
	verbosity_level      =  2):

	heatmap_intensity_factor = 0.66


	os.makedirs(output_dir, exist_ok=True)


	test_data_for_prediction    = []
	test_data_for_visualization = []
	_test_labels_for_prediction = []
	test_labels_for_prediction  = []

	class_counter_dict = dict.fromkeys(test_classes, 0)

	for image, image_orig, filename, label in zip(test_data, test_data_orig, test_filenames, test_labels):
		if class_counter_dict[label] < max_images_per_class:
			test_data_for_prediction.append(image)
			test_data_for_visualization.append(image_orig)
			_test_labels_for_prediction.append(label)
			test_labels_for_prediction.append(f'{label}_{filename}')
			class_counter_dict[label] += 1

	test_data_for_prediction = np.asarray(test_data_for_prediction, dtype=np.float32)


	for layer_counter, layer in enumerate(model.layers):
		if 'conv' not in layer.name:
			continue

		if verbosity_level >= 2:
			Hexnet_print(f'(visualize_activations) layer={layer} (layer_counter={layer_counter}, layer.name={layer.name}): layer.output.shape={layer.output.shape}')


		model_outputs = Model(inputs = model.inputs, outputs = (model.output, layer.output))

		with tf.GradientTape() as gradient_tape:
			predictions, layer_output = model_outputs(test_data_for_prediction)
			# predictions = tf.reduce_max(predictions, axis=1)
			predictions_indices = np.stack((np.arange(0, len(test_labels_for_prediction)), _test_labels_for_prediction), axis=1)
			predictions = tf.gather_nd(predictions, indices=predictions_indices)
			gradients   = gradient_tape.gradient(predictions, layer_output)
			gradients   = tf.reduce_mean(gradients, axis = (1, 2))

		activations = tf.einsum('ijkl,il->ijkl', layer_output, gradients)
		# activations = tf.reduce_mean(activations, axis=3)
		activations = tf.reduce_sum(activations, axis=3)
		activations = np.maximum(activations, 0)


		for image_counter, (image, label, activation) in enumerate(zip(tqdm(test_data_for_visualization), test_labels_for_prediction, activations)):
			activation_max = activation.max()

			if activation_max:
				activation /= activation_max

			activation = cv2.resize(activation, (image.shape[1], image.shape[0]))
			heatmap    = plt.cm.get_cmap(colormap)(activation)[:, :, :3]

			image_heatmapped = image.astype(np.float32) / 255 + heatmap_intensity_factor * heatmap
			image_heatmapped = (image_heatmapped - image_heatmapped.min()) / (image_heatmapped.max() - image_heatmapped.min())

			title = f'layer{str(layer_counter).zfill(3)}_{layer.name}_label{str(label).zfill(3)}_image{str(image_counter).zfill(3)}'
			image_heatmapped_filename = os.path.join(output_dir, f'{title}_image_heatmapped')
			heatmap_filename          = os.path.join(output_dir, f'{title}_activations_heatmap')

			imsave(f'{image_heatmapped_filename}.png', image_heatmapped)
			imsave(f'{heatmap_filename}.png',          heatmap)

			if visualize_hexagonal:
				# heatmap = heatmap / 255

				visualize_hexarray(image_heatmapped, image_heatmapped_filename)
				visualize_hexarray(heatmap,          heatmap_filename)


################################################################################
# Visualize filters, feature maps, and activations
################################################################################

def visualize_model(
	model,
	test_classes,
	test_data,
	test_data_orig,
	test_filenames,
	test_labels,
	colormap,
	visualize_hexagonal,
	output_dir,
	max_images_per_class = 10,
	verbosity_level      =  2):

	os.makedirs(output_dir, exist_ok=True)

	visualize_filters(
		model,
		colormap,
		visualize_hexagonal,
		os.path.join(output_dir, 'filters'),
		verbosity_level)

	visualize_feature_maps(
		model,
		test_classes,
		test_data,
		test_filenames,
		test_labels,
		colormap,
		visualize_hexagonal,
		os.path.join(output_dir, 'feature_maps'),
		max_images_per_class,
		verbosity_level)

	visualize_activations(
		model,
		test_classes,
		test_data,
		test_data_orig,
		test_filenames,
		test_labels,
		colormap,
		visualize_hexagonal,
		os.path.join(output_dir, 'activations'),
		max_images_per_class,
		verbosity_level)


################################################################################
# Output training results and visualize as plot
################################################################################

def visualize_training_results(history, title, output_dir, show_results):
	if output_dir is not None:
		os.makedirs(output_dir, exist_ok=True)

	keys = history.history.keys()

	nrows = 1
	ncols = len([None for key in keys if 'val_' not in key])
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


################################################################################
# Output test results as classification report and visualize confusion
#  matrices
################################################################################

# TODO: multi-label classification
def visualize_test_results(
	predictions,
	test_classes,
	test_classes_orig,
	test_filenames,
	test_labels,
	test_labels_orig,
	title,
	output_dir):

	if output_dir is not None:
		os.makedirs(output_dir, exist_ok=True)

	set_is_multilabel_set = test_labels.ndim > 1

	if not set_is_multilabel_set:
		predictions_classes = [test_classes[prediction_class_index] for prediction_class_index in predictions.argmax(axis=-1)]
	else:
		predictions_classes = (predictions > 0.5).astype(np.uint8) # TODO: label translation

	classification_report = sklearn.metrics.classification_report(test_labels, predictions_classes, target_names=test_classes_orig, output_dict=True)

	classification_report['macro avg']['jaccard']     = sklearn.metrics.jaccard_score(test_labels, predictions_classes, average='macro')
	classification_report['weighted avg']['jaccard']  = sklearn.metrics.jaccard_score(test_labels, predictions_classes, average='weighted')
	classification_report['macro avg']['f0-score']    = sklearn.metrics.fbeta_score(test_labels, predictions_classes, beta=0, average='macro')
	classification_report['weighted avg']['f0-score'] = sklearn.metrics.fbeta_score(test_labels, predictions_classes, beta=0, average='weighted')
	classification_report['macro avg']['f2-score']    = sklearn.metrics.fbeta_score(test_labels, predictions_classes, beta=2, average='macro')
	classification_report['weighted avg']['f2-score'] = sklearn.metrics.fbeta_score(test_labels, predictions_classes, beta=2, average='weighted')

	if not set_is_multilabel_set:
		confusion_matrix                    = sklearn.metrics.confusion_matrix(test_labels, predictions_classes)
		confusion_matrix_normalized_columns = confusion_matrix / confusion_matrix.sum(axis=0)
		confusion_matrix_normalized_rows    = confusion_matrix / confusion_matrix.sum(axis=1, keepdims=True)
		confusion_matrix_normalized_global  = confusion_matrix / confusion_matrix.sum()


		predictions_classes_unique = {list(test_classes).index(prediction_class): test_classes_orig[list(test_classes).index(prediction_class)] for prediction_class in list(set(predictions_classes))}

		for prediction_class_index, prediction_class in predictions_classes_unique.items():
			TP = confusion_matrix[prediction_class_index][prediction_class_index] # true positive (TP)
			TN = confusion_matrix.trace() - TP                                    # true negative (TN)
			FP = confusion_matrix[:, prediction_class_index].sum() - TP           # false positive (FP)
			FN = confusion_matrix[prediction_class_index, :].sum() - TP           # false negative (FN)

			P = TP + FN # condition positive (P)
			N = TN + FP # condition negative (N)
			T = P  + N  # total

			sensitivity_recall_TPR      = TP / P            # sensitivity, recall, hit rate, or true positive rate (TPR)
			specificity_selectivity_TNR = TN / N            # specificity, selectivity, or true negative rate (TNR)
			precision_PPV               = TP / (TP + FP)    # precision or positive predictive value (PPV)
			NPV                         = TN / (TN + FN)    # negative predictive value (NPV)
			FNR                         = FN / P            # miss rate or false negative rate (FNR)
			FPR                         = FP / N            # fall-out or false positive rate (FPR)
			FDR                         = 1 - precision_PPV # false discovery rate (FDR)
			FOR                         = 1 - NPV           # false omission rate (FOR)

			prevalence        = P / T
			accuracy          = (TP + TN) / T
			balanced_accuracy = (sensitivity_recall_TPR + specificity_selectivity_TNR) / 2


			classification_report[prediction_class]['Hexnet_metric__TP']                          = TP
			classification_report[prediction_class]['Hexnet_metric__TN']                          = TN
			classification_report[prediction_class]['Hexnet_metric__FP']                          = FP
			classification_report[prediction_class]['Hexnet_metric__FN']                          = FN
			classification_report[prediction_class]['Hexnet_metric__P']                           = P
			classification_report[prediction_class]['Hexnet_metric__N']                           = N
			classification_report[prediction_class]['Hexnet_metric__T']                           = T
			classification_report[prediction_class]['Hexnet_metric__sensitivity_recall_TPR']      = sensitivity_recall_TPR
			classification_report[prediction_class]['Hexnet_metric__specificity_selectivity_TNR'] = specificity_selectivity_TNR
			classification_report[prediction_class]['Hexnet_metric__precision_PPV']               = precision_PPV
			classification_report[prediction_class]['Hexnet_metric__NPV']                         = NPV
			classification_report[prediction_class]['Hexnet_metric__FNR']                         = FNR
			classification_report[prediction_class]['Hexnet_metric__FPR']                         = FPR
			classification_report[prediction_class]['Hexnet_metric__FDR']                         = FDR
			classification_report[prediction_class]['Hexnet_metric__FOR']                         = FOR
			classification_report[prediction_class]['Hexnet_metric__prevalence']                  = prevalence
			classification_report[prediction_class]['Hexnet_metric__accuracy']                    = accuracy
			classification_report[prediction_class]['Hexnet_metric__balanced_accuracy']           = balanced_accuracy

		classification_report['Hexnet_metric__TP']                          = np.mean([classification_report[pc]['Hexnet_metric__TP']                          for pc in predictions_classes_unique.values()])
		classification_report['Hexnet_metric__TN']                          = np.mean([classification_report[pc]['Hexnet_metric__TN']                          for pc in predictions_classes_unique.values()])
		classification_report['Hexnet_metric__FP']                          = np.mean([classification_report[pc]['Hexnet_metric__FP']                          for pc in predictions_classes_unique.values()])
		classification_report['Hexnet_metric__FN']                          = np.mean([classification_report[pc]['Hexnet_metric__FN']                          for pc in predictions_classes_unique.values()])
		classification_report['Hexnet_metric__P']                           = np.mean([classification_report[pc]['Hexnet_metric__P']                           for pc in predictions_classes_unique.values()])
		classification_report['Hexnet_metric__N']                           = np.mean([classification_report[pc]['Hexnet_metric__N']                           for pc in predictions_classes_unique.values()])
		classification_report['Hexnet_metric__T']                           = np.mean([classification_report[pc]['Hexnet_metric__T']                           for pc in predictions_classes_unique.values()])
		classification_report['Hexnet_metric__sensitivity_recall_TPR']      = np.mean([classification_report[pc]['Hexnet_metric__sensitivity_recall_TPR']      for pc in predictions_classes_unique.values()])
		classification_report['Hexnet_metric__specificity_selectivity_TNR'] = np.mean([classification_report[pc]['Hexnet_metric__specificity_selectivity_TNR'] for pc in predictions_classes_unique.values()])
		classification_report['Hexnet_metric__precision_PPV']               = np.mean([classification_report[pc]['Hexnet_metric__precision_PPV']               for pc in predictions_classes_unique.values()])
		classification_report['Hexnet_metric__NPV']                         = np.mean([classification_report[pc]['Hexnet_metric__NPV']                         for pc in predictions_classes_unique.values()])
		classification_report['Hexnet_metric__FNR']                         = np.mean([classification_report[pc]['Hexnet_metric__FNR']                         for pc in predictions_classes_unique.values()])
		classification_report['Hexnet_metric__FPR']                         = np.mean([classification_report[pc]['Hexnet_metric__FPR']                         for pc in predictions_classes_unique.values()])
		classification_report['Hexnet_metric__FDR']                         = np.mean([classification_report[pc]['Hexnet_metric__FDR']                         for pc in predictions_classes_unique.values()])
		classification_report['Hexnet_metric__FOR']                         = np.mean([classification_report[pc]['Hexnet_metric__FOR']                         for pc in predictions_classes_unique.values()])
		classification_report['Hexnet_metric__prevalence']                  = np.mean([classification_report[pc]['Hexnet_metric__prevalence']                  for pc in predictions_classes_unique.values()])
		classification_report['Hexnet_metric__accuracy']                    = np.mean([classification_report[pc]['Hexnet_metric__accuracy']                    for pc in predictions_classes_unique.values()])
		classification_report['Hexnet_metric__balanced_accuracy']           = np.mean([classification_report[pc]['Hexnet_metric__balanced_accuracy']           for pc in predictions_classes_unique.values()])


		output_dir_confusion_matrix                    = os.path.join(output_dir, f'{title}_confusion_matrix')
		output_dir_confusion_matrix_normalized_columns = f'{output_dir_confusion_matrix}_normalized_columns'
		output_dir_confusion_matrix_normalized_rows    = f'{output_dir_confusion_matrix}_normalized_rows'
		output_dir_confusion_matrix_normalized_global  = f'{output_dir_confusion_matrix}_normalized_global'

	with open(os.path.join(output_dir, f'{title}_predictions.csv'), 'w') as predictions_file:
		print('label_orig,filename,label,prediction_class,prediction', file=predictions_file)

		for label_orig, filename, label, prediction_class, prediction in zip(test_labels_orig, test_filenames, test_labels, predictions_classes, predictions):
			prediction = [float(format(class_confidence, '.8f')) for class_confidence in prediction]
			print(f'{label_orig},{filename},{label},{prediction_class},{prediction}', file=predictions_file)

	with open(os.path.join(output_dir, f'{title}_classification_report.dat'), 'w') as classification_report_file:
		pprint(classification_report, stream=classification_report_file)

	if not set_is_multilabel_set:
		np.savetxt(f'{output_dir_confusion_matrix}.csv',                    confusion_matrix,                    fmt='%d',   delimiter=',')
		np.savetxt(f'{output_dir_confusion_matrix_normalized_columns}.csv', confusion_matrix_normalized_columns, fmt='%.8f', delimiter=',')
		np.savetxt(f'{output_dir_confusion_matrix_normalized_rows}.csv',    confusion_matrix_normalized_rows,    fmt='%.8f', delimiter=',')
		np.savetxt(f'{output_dir_confusion_matrix_normalized_global}.csv',  confusion_matrix_normalized_global,  fmt='%.8f', delimiter=',')

		confusion_matrix_dataframe                    = pd.DataFrame(confusion_matrix,                    test_classes_orig, test_classes_orig)
		confusion_matrix_normalized_columns_dataframe = pd.DataFrame(confusion_matrix_normalized_columns, test_classes_orig, test_classes_orig)
		confusion_matrix_normalized_rows_dataframe    = pd.DataFrame(confusion_matrix_normalized_rows,    test_classes_orig, test_classes_orig)
		confusion_matrix_normalized_global_dataframe  = pd.DataFrame(confusion_matrix_normalized_global,  test_classes_orig, test_classes_orig)

		ax = sn.heatmap(confusion_matrix_dataframe, cmap='viridis', annot=True, fmt='d')
		ax.set(title='model test confusion matrix', xlabel='predicted label', ylabel='true label')
		plt.savefig(f'{output_dir_confusion_matrix}.png')
		plt.savefig(f'{output_dir_confusion_matrix}.pdf')
		plt.clf()

		ax = sn.heatmap(confusion_matrix_normalized_columns_dataframe, cmap='viridis', annot=True, fmt='.2f')
		ax.set(title='model test column-wise normalized confusion matrix', xlabel='predicted label', ylabel='true label')
		plt.savefig(f'{output_dir_confusion_matrix_normalized_columns}.png')
		plt.savefig(f'{output_dir_confusion_matrix_normalized_columns}.pdf')
		plt.clf()

		ax = sn.heatmap(confusion_matrix_normalized_rows_dataframe, cmap='viridis', annot=True, fmt='.2f')
		ax.set(title='model test row-wise normalized confusion matrix', xlabel='predicted label', ylabel='true label')
		plt.savefig(f'{output_dir_confusion_matrix_normalized_rows}.png')
		plt.savefig(f'{output_dir_confusion_matrix_normalized_rows}.pdf')
		plt.clf()

		ax = sn.heatmap(confusion_matrix_normalized_global_dataframe, cmap='viridis', annot=True, fmt='.2f')
		ax.set(title='model test globally normalized confusion matrix', xlabel='predicted label', ylabel='true label')
		plt.savefig(f'{output_dir_confusion_matrix_normalized_global}.png')
		plt.savefig(f'{output_dir_confusion_matrix_normalized_global}.pdf')

	plt.close()

	return classification_report


################################################################################
# Create summarized classification report: calculate mean and std per metric
################################################################################

def summarize_classification_reports(classification_reports):
	summary = {}

	for key in classification_reports[0].keys():
		if key.startswith('Hexnet_metric__') or 'accuracy' in key:
			key_values = [report[key] for report in classification_reports]
			key_mean   = np.mean(key_values)
			key_std    = np.sqrt(((key_values - key_mean)**2).mean())

			summary[f'{key}_mean'] = key_mean
			summary[f'{key}_std']  = key_std
		else:
			key_dict = {}

			for subkey in classification_reports[0][key].keys():
				subkey_values = [report[key][subkey] for report in classification_reports if subkey in report[key]]
				subkey_mean   = np.mean(subkey_values)
				subkey_std    = np.sqrt(((subkey_values - subkey_mean)**2).mean())

				key_dict[f'{subkey}_mean'] = subkey_mean
				key_dict[f'{subkey}_std']  = subkey_std

			summary[key] = key_dict

	return summary


################################################################################
# Create summarized classification report for all training and test runs
################################################################################

def visualize_global_test_results(classification_reports, title, output_dir):
	if output_dir is not None:
		os.makedirs(output_dir, exist_ok=True)

	with open(os.path.join(output_dir, f'{title}_classification_reports_summary.dat'), 'w') as classification_report_file:
		pprint(summarize_classification_reports(classification_reports), stream=classification_report_file)

