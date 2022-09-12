#!/usr/bin/env python3.7


'''****************************************************************************
 * ophthalmology_evaluation.py: Ophthalmology Models Test Script Evaluation
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

enable_logMAR_evaluation = True
evaluation_metrics       = ['ae', 'se', 'mae', 'mse', 'rmse']
look_back                = 4

predictions_to_evaluate = [(1, 'model'), (2, 'annotator')] # 0 = ground truth
dashboard_data_dir      = 'D:/Data/_Ophthalmology/data/dashboard_data_20220525'
output_dir              = 'D:/Data/_Ophthalmology/data/dashboard_data_20220525_evaluated_20220525'

unchanged_factor = 0.1




################################################################################
# Imports
################################################################################

import argparse
import math
import os
import sys

from datetime import datetime
from glob     import glob
from pprint   import pprint

import numpy as np

sys.path[0] = '../..'
import Hexnet




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
		if (not enable_logMAR_evaluation and predictions[i] < 0.01) or (enable_logMAR_evaluation and predictions[i] > 2):
			continue

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
# Load and evaluate the ophthalmic predictions
################################################################################


############################################################################
# Load the ophthalmic predictions
############################################################################

def load_dashboard_data(dashboard_data_dir):
	dashboard_data = []

	for csv in glob(os.path.join(dashboard_data_dir, '*.csv')):
		with open(csv) as csv_file:
			lines = csv_file.readlines()[look_back + 1:]

		lines = [line.rstrip().replace(';', ',').split(',')[2:] for line in lines]

		dashboard_data.append(lines)

	return dashboard_data


############################################################################
# Evaluate the ophthalmic predictions
############################################################################

def evaluate_dashboard_data(dashboard_data, predictions_to_evaluate, output_dir):
	print('> Evaluation')

	for predictions_index, predictions in predictions_to_evaluate:
		timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
		run_title = f'{predictions_index}__{predictions}__{timestamp}'


		print(f'>> Evaluation for predictions "{run_title}"')


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

		for metric in evaluation_metrics:
			prediction_accuracies_dict[f'metric_{metric}'] = []


		for lines in dashboard_data:
			if len(lines) < 2:
				continue


			ground_truth         =                        [float(line[0])                                     for line in lines]
			selected_predictions = [float(lines[0][0])] + [float(line[predictions_index].replace('<0', '-1')) for line in lines[1:]]


			prediction_accuracy_dict = calculate_prediction_accuracy(ground_truth, selected_predictions, unchanged_factor, evaluation_metrics)

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

			for metric in evaluation_metrics:
				prediction_accuracies_dict[f'metric_{metric}'].append(prediction_accuracy_dict[f'metric_{metric}'])


		# Hexnet's test results visualization

		classification_report = Hexnet.visualization.visualize_test_results(
			predictions       = np.asarray([{'increased': [1, 0, 0], 'decreased': [0, 1, 0], 'unchanged': [0, 0, 1]}[p] for p in prediction_accuracies_dict['predictions_IDU']]),
			test_classes      = np.asarray([0, 1, 2]),
			test_classes_orig = np.asarray(['increased', 'decreased', 'unchanged']),
			test_filenames    = np.asarray([]),
			test_labels       = np.asarray([{'increased': 0, 'decreased': 1, 'unchanged': 2}[p] for p in prediction_accuracies_dict['ground_truth_IDU']]),
			test_labels_orig  = np.asarray(prediction_accuracies_dict['ground_truth_IDU']),
			title             = run_title,
			output_dir        = os.path.join(output_dir, 'Hexnet\'s_test_results_visualization'))

		with open(os.path.join(output_dir, f'{run_title}_classification_report.dat'), 'w') as classification_report_file:
			pprint(classification_report, stream=classification_report_file)

		print('classification_report =')
		pprint(classification_report)


		for metric in evaluation_metrics:
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

		for metric in evaluation_metrics:
			del prediction_accuracies_dict[f'metric_{metric}']

		print('prediction_accuracies_dict =')
		pprint(prediction_accuracies_dict)


################################################################################
# parse_args
################################################################################

def parse_args():
	parser = argparse.ArgumentParser(description = 'Ophthalmology Models Test Script Evaluation')

	parser.add_argument('--dashboard-data-dir', default = dashboard_data_dir, help = 'input directory')
	parser.add_argument('--output-dir',         default = output_dir,         help = 'output directory')

	return parser.parse_args()


############################################################################
# Load and evaluate the ophthalmic predictions
############################################################################

if __name__ == '__main__':
	args = parse_args()

	print(f'args={args}')


	# Dataset is directory of files
	if glob(os.path.join(args.dashboard_data_dir, '*'))[0].lower().endswith('.csv'):
		dashboard_data = load_dashboard_data(args.dashboard_data_dir)
		evaluate_dashboard_data(dashboard_data, predictions_to_evaluate, args.output_dir)

	# Dataset is directory of directories
	else:
		for data_dir in glob(os.path.join(args.dashboard_data_dir, '*')):
			print('>>>')
			print(f'Current input directory = "{data_dir}":')
			dashboard_data = load_dashboard_data(data_dir)
			evaluate_dashboard_data(dashboard_data, predictions_to_evaluate, os.path.join(args.output_dir, os.path.basename(data_dir)))
			print('<<<')


