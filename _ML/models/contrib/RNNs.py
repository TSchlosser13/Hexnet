#!/usr/bin/env python3.7


'''****************************************************************************
 * RNNs.py: RNN Models Test Script
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
# Default parameters
################################################################################


# TODO: arguments

add_additional_data      = True
balancing_factor         = 100   # TODO: dataset balancing with additional data
enable_logMAR_evaluation = True
generate_plots           = False
growing_window_size      = True
look_back                =   4
max_window_size          =  10


model      = 'MLP'
dataset    = 'D:/Data/_Ophthalmology/data/treatment_prediction_dataset_20211101'
output_dir = 'D:/Data/_Ophthalmology/data/treatment_prediction_dataset_20211101_results_20220329'

data_train_size_fraction = 0.9
unchanged_factor         = 0.1

batch_size       = 32
epochs           =  2
validation_split =  0.1

disable_tensorflow_warnings = True




################################################################################
# Disable TensorFlow warnings
################################################################################

import os

if disable_tensorflow_warnings:
	os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

	import tensorflow.python.util.deprecation as deprecation
	deprecation._PRINT_DEPRECATION_WARNINGS = False


################################################################################
# Imports
################################################################################

import argparse
import copy
import inspect
import math
import random
import shutil
import sys

import tensorflow as tf

from datetime                import datetime
from glob                    import glob
from pprint                  import pprint
from sklearn.ensemble        import BaggingRegressor, RandomForestRegressor
from tensorflow.keras        import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, GRU, LSTM
from tqdm                    import tqdm

import matplotlib.pyplot as plt
import numpy             as np
import pandas            as pd

sys.path[0] = '../..'
import Hexnet




################################################################################
# Models
################################################################################


############################################################################
# Statistical estimators
############################################################################

class statistical_estimator():
	def __init__(self, *args, **kwargs):
		return None

	def compile(self, *args, **kwargs):
		return None

	def fit(self, *args, **kwargs):
		return None

	def predict(self, data, **kwargs):
		predictions = []

		for current_data in data:
			current_data = current_data.flatten()

			prediction = random.uniform(0, 1)

			# unchanged / increased / decreased = 0.8 / 0.1 / 0.1
			if prediction <= 0.8:
				prediction = current_data[-1]
			elif prediction <= 0.9:
				prediction = current_data[-1] + 0.2
			else:
				prediction = current_data[-1] - 0.2

			predictions.append([prediction])

		return np.asarray(predictions)

def model_statistical_estimator(*args, **kwargs):
	return statistical_estimator(*args, **kwargs)


############################################################################
# (Weighted) moving average models
############################################################################

class moving_average():
	def __init__(self, *args, look_back=max_window_size, enable_weighting=False, **kwargs):
		self.look_back        = look_back
		self.enable_weighting = enable_weighting

		return None

	def compile(self, *args, **kwargs):
		return None

	def fit(self, *args, **kwargs):
		return None

	def predict(self, data, **kwargs):
		predictions = []

		if self.enable_weighting:
			weights = np.arange(0.5, 1.001, 0.5 / (self.look_back - 1)) # weight range = [0.5, 1.0]

		for current_data in data:
			current_data = current_data.flatten()

			if self.enable_weighting:
				current_data *= weights

			prediction = current_data[-self.look_back:].mean()

			if self.enable_weighting:
				current_data /= weights.sum()

			predictions.append([prediction])

		return np.asarray(predictions)

def model_moving_average(*args, **kwargs):
	return moving_average(*args, **kwargs)

def model_moving_average_look_back_4(*args, **kwargs):
	return moving_average(*args, look_back=4, **kwargs)

def model_moving_average_look_back_10(*args, **kwargs):
	return moving_average(*args, look_back=10, **kwargs)

def model_weighted_moving_average(*args, **kwargs):
	return moving_average(*args, enable_weighting=True, **kwargs)

def model_weighted_moving_average_look_back_4(*args, **kwargs):
	return moving_average(*args, look_back=4, enable_weighting=True, **kwargs)

def model_weighted_moving_average_look_back_10(*args, **kwargs):
	return moving_average(*args, look_back=10, enable_weighting=True, **kwargs)


############################################################################
# Regressors
############################################################################

class sklearn_BaggingRegressor():
	def __init__(self, *args, n_jobs=-1, verbose=0, **kwargs):
		self.regressor = BaggingRegressor(n_jobs=n_jobs, verbose=verbose)

		return None

	def compile(self, *args, **kwargs):
		return None

	def fit(self, X, y, *args, **kwargs):
		X = X.reshape((X.shape[0], X.shape[1]))

		self.regressor.fit(X, y)

		return None

	def predict(self, data, **kwargs):
		data = data.reshape((data.shape[0], data.shape[1]))

		predictions = self.regressor.predict(data).reshape(-1, 1)

		return predictions

class sklearn_RandomForestRegressor():
	def __init__(self, *args, n_jobs=-1, verbose=0, **kwargs):
		self.regressor = RandomForestRegressor(n_jobs=n_jobs, verbose=verbose)

		return None

	def compile(self, *args, **kwargs):
		return None

	def fit(self, X, y, *args, **kwargs):
		X = X.reshape((X.shape[0], X.shape[1]))

		self.regressor.fit(X, y)

		return None

	def predict(self, data, **kwargs):
		data = data.reshape((data.shape[0], data.shape[1]))

		predictions = self.regressor.predict(data).reshape(-1, 1)

		return predictions

def model_sklearn_BaggingRegressor(*args, **kwargs):
	return sklearn_BaggingRegressor(*args, **kwargs)

def model_sklearn_RandomForestRegressor(*args, **kwargs):
	return sklearn_RandomForestRegressor(*args, **kwargs)


############################################################################
# Multilayer perceptrons (MLP)
############################################################################

def model_MLP(input_shape):
	model = Sequential()

	model.add(Dense(units=32, activation=tf.nn.relu, input_shape=input_shape))
	model.add(Dropout(rate=0.25))
	model.add(Dense(units=64, activation=tf.nn.relu))
	model.add(Dropout(rate=0.25))
	model.add(Flatten())
	model.add(Dense(units=128, activation=tf.nn.relu))
	model.add(Dropout(rate=0.25))
	model.add(Dense(units=1))

	return model


############################################################################
# Recurrent neural networks (RNN)
############################################################################

def model_RNN_GRU(GRU_units=10, Dense_units=1):
	model = Sequential()

	model.add(GRU(units=GRU_units))
	model.add(Dense(units=Dense_units))

	return model

def model_RNN_LSTM(LSTM_units=10, Dense_units=1):
	model = Sequential()

	model.add(LSTM(units=LSTM_units))
	model.add(Dense(units=Dense_units))

	return model




################################################################################
# Visualization
################################################################################


# https://github.com/TSchlosser13/Hexnet/blob/master/_ML/misc/visualization.py

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


def visualize_prediction_results(
	dataset,
	ground_truth,
	train_predictions,
	test_predictions,
	title,
	output_dir,
	show_results):

	if output_dir is not None:
		os.makedirs(output_dir, exist_ok=True)

	if train_predictions is not None:
		train_predictions_x = range(dataset['look_back'], dataset['look_back'] + len(train_predictions))

	if test_predictions is not None:
		test_predictions_x = range(len(ground_truth) - len(test_predictions), len(ground_truth))

	plt.figure('Test results')

	plt.title(dataset['plot_title'])
	plt.xlabel(dataset['plot_xlabel'])
	plt.ylabel(dataset['plot_ylabel'])

	range_xticks_axvlines = range(0, len(ground_truth), math.ceil(len(ground_truth) / 10))

	plt.xticks(range_xticks_axvlines)

	for x in range_xticks_axvlines:
		plt.axvline(x, color='gray', linestyle='dashed', linewidth=1.0)

	if ground_truth is not None:
		plt.plot(ground_truth, label = 'ground truth (gt)')

	if train_predictions is not None:
		plt.plot(train_predictions_x, train_predictions, label = 'predictions train set (ptrain)')

	if test_predictions is not None:
		plt.plot(test_predictions_x, test_predictions, label = 'predictions test set (ptest)')

	if ground_truth is not None and train_predictions is not None:
		abs_gt_ptrain = [abs(gt - ptrain) for gt, ptrain in zip(ground_truth[train_predictions_x[0]:], train_predictions)]
		plt.plot(train_predictions_x, abs_gt_ptrain, label = 'abs(gt - ptrain)')

	if ground_truth is not None and test_predictions is not None:
		abs_gt_ptest = [abs(gt - ptest) for gt, ptest in zip(ground_truth[test_predictions_x[0]:], test_predictions)]
		plt.plot(test_predictions_x, abs_gt_ptest, label = 'abs(gt - ptest)')

	plt.legend()

	if output_dir is not None:
		results_fig = os.path.join(output_dir, title)
		plt.savefig(f'{results_fig}.png')
		plt.savefig(f'{results_fig}.pdf')

	if show_results:
		plt.show()

	plt.close()




################################################################################
# Evaluation
################################################################################


# https://github.com/TSchlosser13/Hexnet/blob/master/_ML/tools/compare.py

def ae(p1, p2):
	return np.sum(np.abs(p1 - p2), axis=-1)

def se(p1, p2):
	diff = p1 - p2

	return np.sum(np.multiply(diff, diff), axis=-1)

def mae(p1, p2):
	return ae(p1, p2) / p1.size

def mse(p1, p2):
	return se(p1, p2) / p1.size

def rmse(p1, p2):
	return math.sqrt(mse(p1, p2))


def visus_to_logMAR(visus):
	return -math.log10(max(0.001, float(visus)))


def calculate_prediction_accuracy(ground_truth, predictions, unchanged_factor, evaluation_metrics):
	ground_truth_IDU = []
	predictions_IDU  = []

	ground_truth_len = len(ground_truth)

	steps = ground_truth_len - 1


	if enable_logMAR_evaluation:
		ground_truth = np.asarray([visus_to_logMAR(value) for value in ground_truth])
		predictions  = np.asarray([visus_to_logMAR(value) for value in predictions])


	for i in range(1, ground_truth_len):
		if abs(ground_truth[i - 1] - ground_truth[i]) <= unchanged_factor:
			ground_truth_IDU.append('unchanged')
		elif (ground_truth[i] > ground_truth[i - 1] if not enable_logMAR_evaluation else ground_truth[i] < ground_truth[i - 1]):
			ground_truth_IDU.append('increased')
		else:
			ground_truth_IDU.append('decreased')

		if abs(ground_truth[i - 1] - predictions[i]) <= unchanged_factor:
			predictions_IDU.append('unchanged')
		elif (predictions[i] > ground_truth[i - 1] if not enable_logMAR_evaluation else predictions[i] < ground_truth[i - 1]):
			predictions_IDU.append('increased')
		else:
			predictions_IDU.append('decreased')

	prediction_accuracy = sum(1 for gt, p in zip(ground_truth_IDU, predictions_IDU) if gt == p) / steps


	ground_truth_delta = ground_truth[-1] - ground_truth[0]
	predictions_delta  = predictions[-1]  - ground_truth[0]

	if ground_truth_delta <= unchanged_factor:
		ground_truth_delta_class = 'unchanged'
	elif (ground_truth_delta > 0 if not enable_logMAR_evaluation else ground_truth_delta < 0):
		ground_truth_delta_class = 'increased'
	else:
		ground_truth_delta_class = 'decreased'

	if predictions_delta <= unchanged_factor:
		predictions_delta_class = 'unchanged'
	elif (predictions_delta > 0 if not enable_logMAR_evaluation else predictions_delta < 0):
		predictions_delta_class = 'increased'
	else:
		predictions_delta_class = 'decreased'


	prediction_accuracy_dict = {
		'ground_truth_IDU'           : ground_truth_IDU,
		'predictions_IDU'            : predictions_IDU,
		'prediction_accuracy'        : prediction_accuracy,
		'ground_truth_delta'         : ground_truth_delta,
		'ground_truth_delta_class'   : ground_truth_delta_class,
		'predictions_delta'          : predictions_delta,
		'predictions_delta_class'    : predictions_delta_class,
		'ground_truth_increased_cnt' : ground_truth_IDU.count('increased'),
		'ground_truth_decreased_cnt' : ground_truth_IDU.count('decreased'),
		'ground_truth_unchanged_cnt' : ground_truth_IDU.count('unchanged'),
		'predictions_increased_cnt'  : predictions_IDU.count('increased'),
		'predictions_decreased_cnt'  : predictions_IDU.count('decreased'),
		'predictions_unchanged_cnt'  : predictions_IDU.count('unchanged'),
		'steps'                      : steps
	}

	for metric in evaluation_metrics:
		prediction_accuracy_dict[f'metric_{metric}'] = globals()[metric](ground_truth, predictions)


	return prediction_accuracy_dict








################################################################################
# Load the dataset
################################################################################

def load_treatment_prediction_dataset(args):
	dataset = {}

	dataset['title'] = 'treatment_prediction_dataset'

	dataset['filenames'] = []
	dataset['data']      = []

	if add_additional_data:
		dataset['additional_data'] = []

		dataset['additional_data_fields'] = [
			'V_Datum',
			'Geburtsdatum', 'Geschlecht',
			'O_ZentrNetzhDicke', 'O_IntraretFlk', 'O_SubretFlk', 'O_RPE_Abhebg', 'O_SubretFibrose', 'O_RPE', 'O_ELM', 'O_Ellipsoid', 'O_FovDepr', 'O_Narben',
			'D_AMD', 'D_Cataracta', 'D_Pseudophakie', 'D_RVV', 'D_DMOE', 'D_DiabRetino', 'D_Gliose',
			'T_Medikament', 'T_Check_Apoplex', 'T_Check_Blutverd', 'T_Check_Herzinfarkt',
			'SAP_Medikament'
		]

	dataset['look_back'] = look_back

	dataset['evaluation_metrics'] = ['ae', 'se', 'mae', 'mse', 'rmse']

	dataset['plot_title']  = 'model visus prediction'
	dataset['plot_xlabel'] = 'visus sample'
	dataset['plot_ylabel'] = 'visus value'


	for CSV in glob(os.path.join(args.dataset, '*.csv')):
		CSV_dataframe = pd.read_csv(CSV)


		dates         = []
		dates_indices = []

		for date_index, date in enumerate(CSV_dataframe['V_Datum'].to_list()):
			if not dates or date != dates[-1]:
				dates.append(date)
				dates_indices.append(date_index)


		if len(dates) > dataset['look_back']:
			dataset['filenames'].append(os.path.basename(CSV)[:-len('.csv')])


			if 'V_Vis_L' in CSV_dataframe:
				current_data = CSV_dataframe['V_Vis_L'].to_list()
				current_data = [current_data[i] for i in dates_indices]
				dataset['data'].append(current_data)

				if add_additional_data:
					current_data = []

					for field in dataset['additional_data_fields']:
						d = CSV_dataframe[f'{field}_L'].to_list() if f'{field}_L' in CSV_dataframe else CSV_dataframe[f'{field}'].to_list()
						current_data.extend([d[i] / 1000000 for i in dates_indices]) # TODO: additional data normalization

					dataset['additional_data'].append(current_data)
			else: # 'V_Vis_R'
				current_data = CSV_dataframe['V_Vis_R'].to_list()
				current_data = [current_data[i] for i in dates_indices]
				dataset['data'].append(current_data)

				if add_additional_data:
					current_data = []

					for field in dataset['additional_data_fields']:
						d = CSV_dataframe[f'{field}_R'].to_list() if f'{field}_R' in CSV_dataframe else CSV_dataframe[f'{field}'].to_list()
						current_data.extend([d[i] / 1000000 for i in dates_indices]) # TODO: additional data normalization

					dataset['additional_data'].append(current_data)




	# Dataset balancing
	if args.balance_dataset:
		# WLS statistics

		WLS_counter_dict = {
			'winners_cnt_local'     : 0,
			'losers_cnt_local'      : 0,
			'stabilizers_cnt_local' : 0,
		}

		for current_data in dataset['data']:
			for i in range(dataset['look_back'], len(current_data)):
				visus_0 = current_data[i - 1]
				visus_1 = current_data[i]

				if abs(visus_0 - visus_1) <= args.unchanged_factor:
					WLS_counter_dict['stabilizers_cnt_local'] += 1
				elif visus_1 > visus_0:
					WLS_counter_dict['winners_cnt_local'] += 1
				else:
					WLS_counter_dict['losers_cnt_local'] += 1

		print('WLS_counter_dict =')
		pprint(WLS_counter_dict)


		# Dataset balancing
		if args.balance_dataset:
			balanced_filenames = []
			balanced_data      = []

			WLS_counter_dict = {
				'winners_cnt_local'     : 0,
				'losers_cnt_local'      : 0,
				'stabilizers_cnt_local' : 0,
			}

			added_sequence  = True
			added_sequences = []

			while added_sequence:
				added_sequence = False

				for current_filename, current_data in zip(dataset['filenames'], dataset['data']):
					current_winners_cnt_local     = 0
					current_losers_cnt_local      = 0
					current_stabilizers_cnt_local = 0

					for i in range(dataset['look_back'], len(current_data)):
						visus_0 = current_data[i - 1]
						visus_1 = current_data[i]

						if abs(visus_0 - visus_1) <= args.unchanged_factor:
							current_stabilizers_cnt_local += 1
						elif visus_1 > visus_0:
							current_winners_cnt_local += 1
						else: # visus_1 < visus_0
							current_losers_cnt_local += 1

					# Assess current dataset balance before adding more sequences
					if abs((WLS_counter_dict['winners_cnt_local']     + current_winners_cnt_local) -                           \
					       (WLS_counter_dict['losers_cnt_local']      + current_losers_cnt_local))      < balancing_factor and \
					   abs((WLS_counter_dict['winners_cnt_local']     + current_winners_cnt_local) -                           \
					       (WLS_counter_dict['stabilizers_cnt_local'] + current_stabilizers_cnt_local)) < balancing_factor and \
					   abs((WLS_counter_dict['losers_cnt_local']      + current_losers_cnt_local)  -                           \
					       (WLS_counter_dict['stabilizers_cnt_local'] + current_stabilizers_cnt_local)) < balancing_factor and \
					   current_filename not in added_sequences:

						balanced_filenames.append(current_filename)
						balanced_data.append(current_data)

						WLS_counter_dict['winners_cnt_local']     += current_winners_cnt_local
						WLS_counter_dict['losers_cnt_local']      += current_losers_cnt_local
						WLS_counter_dict['stabilizers_cnt_local'] += current_stabilizers_cnt_local

						added_sequence = True
						added_sequences.append(current_filename)

			dataset['filenames'] = balanced_filenames
			dataset['data']      = balanced_data


		print('>> after the dataset balancing')


		# WLS statistics

		WLS_counter_dict = {
			'winners_cnt_local'     : 0,
			'losers_cnt_local'      : 0,
			'stabilizers_cnt_local' : 0,
		}

		for current_data in dataset['data']:
			for i in range(dataset['look_back'], len(current_data)):
				visus_0 = current_data[i - 1]
				visus_1 = current_data[i]

				if abs(visus_0 - visus_1) <= args.unchanged_factor:
					WLS_counter_dict['stabilizers_cnt_local'] += 1
				elif visus_1 > visus_0:
					WLS_counter_dict['winners_cnt_local'] += 1
				else:
					WLS_counter_dict['losers_cnt_local'] += 1

		print('WLS_counter_dict =')
		pprint(WLS_counter_dict)




	dataset['filenames_len'] = len(dataset['filenames'])
	dataset['data_len']      = len(dataset['data'])

	if add_additional_data:
		dataset['additional_data_len'] = len(dataset['additional_data'])


	return dataset








################################################################################
# Prepare the dataset
################################################################################

'''
	Dataset preparation example without additional data (look_back = 3)


	Input

	x	y
	112	118
	118	132
	132	129
	129	121
	121	135


	Output

	x1	x2	x3	y
	112	118	132	129
	118	132	129	121
	132	129	121	135
	129	121	135	148
	121	135	148	148
'''

def prepare_dataset(data, additional_data=None, look_back=4):
	data_x = []
	data_y = []

	if not growing_window_size:
		if additional_data is None:
			for current_data in data:
				for i in range(len(current_data) - look_back):
					data_x.append(current_data[i : i + look_back, 0])
					data_y.append(current_data[i + look_back, 0])
		# else:
			# TODO: fixed window size with additional data
	else:
		if additional_data is None:
			for current_data in data:
				for i in range(len(current_data) - look_back):
					data_i           = max(0, i - (max_window_size - look_back))
					current_data_x_i = max_window_size - (i + look_back - data_i)

					current_data_x                                     = max_window_size * [-1]
					current_data_x[current_data_x_i : max_window_size] = current_data[data_i : i + look_back, 0]

					current_data_y = current_data[i + look_back, 0]

					data_x.append(current_data_x)
					data_y.append(current_data_y)
		else:
			data_factor = int(len(additional_data[0]) / len(data[0])) + 1

			for current_data, current_additional_data in zip(data, additional_data):
				for i in range(len(current_data) - look_back):
					data_i           = max(0, i - (max_window_size - look_back))
					current_data_x_i = max_window_size - (i + look_back - data_i)

					current_data_x                                     = data_factor * max_window_size * [-1]
					current_data_x[current_data_x_i : max_window_size] = current_data[data_i : i + look_back, 0]

					for j in range(data_factor - 1):
						data_i           = max(0, i - (max_window_size - look_back))
						current_data_x_i = max_window_size - (i + look_back - data_i)

						current_data_x[(j + 1) * max_window_size + current_data_x_i : (j + 1) * max_window_size + max_window_size] = \
							current_additional_data[j * len(current_data) + data_i : j * len(current_data) + i + look_back, 0]

					current_data_y = current_data[i + look_back, 0]

					data_x.append(current_data_x)
					data_y.append(current_data_y)

	return (np.asarray(data_x), np.asarray(data_y))








################################################################################
# Test the model
################################################################################

def test_model(dataset, args):
	timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
	run_title = f'{args.model}__{dataset["title"]}__{timestamp}'




	############################################################################
	# Prepare the dataset
	############################################################################


	print('>> Dataset preparation')


	dataset['data_train_size_fraction'] = args.data_train_size_fraction

	dataset['data_reshaped']       = [np.asarray(current_data).reshape(-1, 1) for current_data in dataset['data']]
	dataset['data_train_size']     = int(dataset['data_train_size_fraction'] * dataset['data_len'])
	dataset['data_test_size']      = dataset['data_len'] - dataset['data_train_size']
	dataset['data_train_set']      = dataset['data_reshaped'][:dataset['data_train_size']]
	dataset['data_test_set']       = dataset['data_reshaped'][dataset['data_train_size']:]
	dataset['filenames_train_set'] = dataset['filenames'][:dataset['data_train_size']]
	dataset['filenames_test_set']  = dataset['filenames'][dataset['data_train_size']:]

	if 'additional_data' in dataset:
		dataset['additional_data_reshaped']  = [np.asarray(current_data).reshape(-1, 1) for current_data in dataset['additional_data']]
		dataset['additional_data_train_set'] = dataset['additional_data_reshaped'][:dataset['data_train_size']]
		dataset['additional_data_test_set']  = dataset['additional_data_reshaped'][dataset['data_train_size']:]

	if 'additional_data' not in dataset:
		train_set_x, train_set_y = prepare_dataset(dataset['data_train_set'], look_back=dataset['look_back'])
		test_set_x,  test_set_y  = prepare_dataset(dataset['data_test_set'],  look_back=dataset['look_back'])
	else:
		train_set_x, train_set_y = prepare_dataset(dataset['data_train_set'], dataset['additional_data_train_set'], look_back=dataset['look_back'])
		test_set_x,  test_set_y  = prepare_dataset(dataset['data_test_set'],  dataset['additional_data_test_set'],  look_back=dataset['look_back'])

	train_set_x = train_set_x.reshape( train_set_x.shape[0], train_set_x.shape[1], 1 )
	test_set_x  = test_set_x.reshape(  test_set_x.shape[0],  test_set_x.shape[1],  1 )

	dataset_logMAR = copy.deepcopy(dataset)

	dataset_logMAR['plot_title']  = f'{dataset_logMAR["plot_title"]} (logMAR)'
	dataset_logMAR['plot_ylabel'] = f'{dataset_logMAR["plot_ylabel"]} (logMAR)'


	output_dir_dataset = os.path.join(args.output_dir, 'dataset')

	for current_set in ['train_set', 'test_set']:
		output_dir_current_set = os.path.join(output_dir_dataset, current_set)

		os.makedirs(output_dir_current_set, exist_ok=True)

		for current_filename in dataset[f'filenames_{current_set}']:
			shutil.copyfile(os.path.join(args.dataset, f'{current_filename}.csv'), os.path.join(output_dir_current_set, f'{current_filename}.csv'))




	############################################################################
	# Initialize the model
	############################################################################

	print('>> Model initialization')

	model_string = args.model

	model = vars(sys.modules[__name__])[f'model_{model_string}']

	if 'RNN' not in model_string:
		model = model(train_set_x.shape[1:3])
	else:
		model = model()

	model.compile(optimizer='adam', loss='mean_squared_error')


	############################################################################
	# Train the model
	############################################################################

	print('>> Model training')

	history = model.fit(train_set_x, train_set_y, args.batch_size, args.epochs, validation_split=args.validation_split)

	if history is not None and generate_plots:
		visualize_training_results(history, f'{run_title}_training_results', args.output_dir, show_results=False)




	############################################################################
	# Evaluate the model
	############################################################################


	print('>> Model evaluation')


	prediction_accuracies_dict = {
		'ground_truth_IDU'                 : [],
		'predictions_IDU'                  : [],
		'prediction_accuracies_local'      : [],
		'ground_truth_deltas'              : [],
		'ground_truth_deltas_classes'      : [],
		'predictions_deltas'               : [],
		'predictions_deltas_classes'       : [],
		'ground_truth_increased_cnt_local' : 0,
		'ground_truth_decreased_cnt_local' : 0,
		'ground_truth_unchanged_cnt_local' : 0,
		'predictions_increased_cnt_local'  : 0,
		'predictions_decreased_cnt_local'  : 0,
		'predictions_unchanged_cnt_local'  : 0,
		'steps'                            : []
	}

	for metric in dataset['evaluation_metrics']:
		prediction_accuracies_dict[f'metric_{metric}'] = []

	plot_title_base = dataset['plot_title']


	output_dir_meta_dataset = os.path.join(args.output_dir, 'meta_dataset')

	for IDU in ['increased', 'decreased', 'unchanged']:
		os.makedirs(os.path.join(output_dir_meta_dataset, IDU), exist_ok=True)


	for index, (filename, data) in enumerate(tqdm(zip(dataset['filenames_test_set'], dataset['data_test_set']), total=len(dataset['filenames_test_set']))):

		########################################################################
		# Prepare the current data
		########################################################################

		data_normalized = np.asarray(data).reshape(-1, 1)

		if 'additional_data' in dataset:
			additional_data_normalized = np.asarray(dataset['additional_data_test_set'][index]).reshape(-1, 1)

		if 'additional_data' not in dataset:
			test_set_x, test_set_y = prepare_dataset([data_normalized], look_back=dataset['look_back'])
		else:
			test_set_x, test_set_y = prepare_dataset([data_normalized], [additional_data_normalized], look_back=dataset['look_back'])

		test_set_x = test_set_x.reshape(test_set_x.shape[0], test_set_x.shape[1], 1)


		########################################################################
		# Model prediction
		########################################################################

		test_predictions = model.predict(test_set_x)




		########################################################################
		# Save test results
		########################################################################


		with open(os.path.join(args.output_dir, f'{run_title}_prediction_results__{filename}.csv'), 'w') as test_predictions_CSV:
			print('sample,value', file=test_predictions_CSV)

			for visus_value_index, visus_value in enumerate(test_predictions.flatten()):
				print(f'{visus_value_index + dataset["look_back"]},{visus_value}', file=test_predictions_CSV)


		prediction_accuracy_dict = calculate_prediction_accuracy(
			ground_truth       = data_normalized[dataset['look_back'] - 1:],
			predictions        = np.concatenate(([data_normalized[dataset['look_back'] - 1]], test_predictions)),
			unchanged_factor   = args.unchanged_factor,
			evaluation_metrics = dataset['evaluation_metrics'])

		prediction_accuracies_dict['ground_truth_IDU'] += prediction_accuracy_dict['ground_truth_IDU']
		prediction_accuracies_dict['predictions_IDU']  += prediction_accuracy_dict['predictions_IDU']
		prediction_accuracies_dict['prediction_accuracies_local'].append(prediction_accuracy_dict['prediction_accuracy'])
		prediction_accuracies_dict['ground_truth_deltas'].append(prediction_accuracy_dict['ground_truth_delta'])
		prediction_accuracies_dict['ground_truth_deltas_classes'].append(prediction_accuracy_dict['ground_truth_delta_class'])
		prediction_accuracies_dict['predictions_deltas'].append(prediction_accuracy_dict['predictions_delta'])
		prediction_accuracies_dict['predictions_deltas_classes'].append(prediction_accuracy_dict['predictions_delta_class'])

		for IDU in ['increased', 'decreased', 'unchanged']:
			prediction_accuracies_dict[f'ground_truth_{IDU}_cnt_local'] += prediction_accuracy_dict[f'ground_truth_{IDU}_cnt']
			prediction_accuracies_dict[f'predictions_{IDU}_cnt_local']  += prediction_accuracy_dict[f'predictions_{IDU}_cnt']

		prediction_accuracies_dict['steps'].append(prediction_accuracy_dict['steps'])

		for metric in dataset['evaluation_metrics']:
			prediction_accuracies_dict[f'metric_{metric}'].append(prediction_accuracy_dict[f'metric_{metric}'])


		if generate_plots:
			dataset['plot_title']        = f'{plot_title_base}: "{filename}"'
			dataset_logMAR['plot_title'] = f'{plot_title_base} (logMAR): "{filename}"'

			data_normalized_logMAR  = [visus_to_logMAR(visus) for visus in data_normalized]
			test_predictions_logMAR = [visus_to_logMAR(visus) for visus in test_predictions]

			visualize_prediction_results(
				dataset           = dataset,
				ground_truth      = data_normalized,
				train_predictions = None,
				test_predictions  = test_predictions,
				title             = f'{run_title}_prediction_results__{filename}',
				output_dir        = args.output_dir,
				show_results      = False)

			visualize_prediction_results(
				dataset           = dataset_logMAR,
				ground_truth      = data_normalized_logMAR,
				train_predictions = None,
				test_predictions  = test_predictions_logMAR,
				title             = f'{run_title}_prediction_results_logMAR__{filename}',
				output_dir        = args.output_dir,
				show_results      = False)


		for i, (p, ps, d) in enumerate(zip(test_predictions, prediction_accuracy_dict['predictions_IDU'], test_set_x)):
			with open(os.path.join(output_dir_meta_dataset, ps, f'{filename}_window{i}.csv'), 'w') as p_CSV:
				print(f'{p[0]},{str(d.flatten().tolist()).replace(" ", "").replace("[", "").replace("]", "")}', file=p_CSV)




	############################################################################
	# Hexnet test results visualization
	############################################################################

	classification_report = Hexnet.visualization.visualize_test_results(
		predictions       = np.asarray([{'increased': [1, 0, 0], 'decreased': [0, 1, 0], 'unchanged': [0, 0, 1]}[p] for p in prediction_accuracies_dict['predictions_IDU']]),
		test_classes      = np.asarray([0, 1, 2]),
		test_classes_orig = np.asarray(['increased', 'decreased', 'unchanged']),
		test_filenames    = np.asarray([]),
		test_labels       = np.asarray([{'increased': 0, 'decreased': 1, 'unchanged': 2}[p] for p in prediction_accuracies_dict['ground_truth_IDU']]),
		test_labels_orig  = np.asarray(prediction_accuracies_dict['ground_truth_IDU']),
		title             = run_title,
		output_dir        = os.path.join(args.output_dir, 'Hexnet_test_results_visualization'))

	with open(os.path.join(args.output_dir, f'{run_title}_classification_report.dat'), 'w') as classification_report_file:
		pprint(classification_report, stream=classification_report_file)

	print('classification_report =')
	pprint(classification_report)


	############################################################################
	# Save global test results
	############################################################################

	for metric in dataset['evaluation_metrics']:
		prediction_accuracies_dict[f'metric_{metric}_unweighted'] = \
			sum(prediction_accuracies_dict[f'metric_{metric}']) / \
			len(prediction_accuracies_dict[f'metric_{metric}'])

		prediction_accuracies_dict[f'metric_{metric}_weighted'] = \
			sum(steps * m for m, steps in zip(prediction_accuracies_dict[f'metric_{metric}'], prediction_accuracies_dict['steps'])) / \
			sum(prediction_accuracies_dict['steps'])

	for IDU in ['increased', 'decreased', 'unchanged']:
		prediction_accuracies_dict[f'ground_truth_{IDU}_cnt'] = prediction_accuracies_dict['ground_truth_deltas_classes'].count(IDU)
		prediction_accuracies_dict[f'predictions_{IDU}_cnt']  = prediction_accuracies_dict['predictions_deltas_classes'].count(IDU)

	del prediction_accuracies_dict['ground_truth_IDU']
	del prediction_accuracies_dict['predictions_IDU']
	del prediction_accuracies_dict['prediction_accuracies_local']
	del prediction_accuracies_dict['ground_truth_deltas']
	del prediction_accuracies_dict['ground_truth_deltas_classes']
	del prediction_accuracies_dict['predictions_deltas']
	del prediction_accuracies_dict['predictions_deltas_classes']
	del prediction_accuracies_dict['steps']

	for metric in dataset['evaluation_metrics']:
		del prediction_accuracies_dict[f'metric_{metric}']

	print('prediction_accuracies_dict =')
	pprint(prediction_accuracies_dict)








################################################################################
# parse_args
################################################################################

def parse_args():
	parser = argparse.ArgumentParser(description='Hexnet: The Hexagonal Machine Learning Module - RNN Models Test Script')


	model_choices = [model[0][len('model_'):] for model in inspect.getmembers(sys.modules[__name__], inspect.isfunction) if model[0].startswith('model_')]


	parser.add_argument(
		'--model',
		default = model,
		choices = model_choices,
		help    = 'model for training and testing: choices are generated from RNNs.py')

	parser.add_argument('--dataset',                                default = dataset,                  help = 'load dataset from directory')
	parser.add_argument('--output-dir',                             default = output_dir,               help = 'training and test results\' output directory')

	parser.add_argument('--balance-dataset',                        action  = 'store_true',             help = 'balance the dataset')
	parser.add_argument('--data-train-size-fraction', type = float, default = data_train_size_fraction, help = 'fraction of the data to be used as training data')
	parser.add_argument('--unchanged-factor',         type = float, default = unchanged_factor,         help = 'unchanged threshold (unchanged / increased / decreased classification)')

	parser.add_argument('--batch-size',               type = int,   default = batch_size,               help = 'batch size for training')
	parser.add_argument('--epochs',                   type = int,   default = epochs,                   help = 'epochs')
	parser.add_argument('--validation-split',         type = float, default = validation_split,         help = 'fraction of the training data to be used as validation data')


	return parser.parse_args()




################################################################################
# main
################################################################################

if __name__ == '__main__':
	args = parse_args()

	print(f'args={args}\n')


	print('> Loading dataset')

	test_dataset = load_treatment_prediction_dataset(args)

	Hexnet.print_newline()


	print('> Starting a new training and test run')

	test_model(test_dataset, args)

