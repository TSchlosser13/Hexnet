#!/usr/bin/env python3.7


'''****************************************************************************
 * Model_Comparison.py: Model Comparison Test Script
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

model_s            = ['CNN', 'SCNN']
dataset_s          = ['../datasets/MNIST/MNIST.h5']
output_dir         = 'Model_Comparison'

batch_size         = 32
epochs             =  2
runs               =  2
validation_split   =  0.1

resnet_stacks      = 3
resnet_n           = 3
resnet_filter_size = 1.0


################################################################################
# Imports
################################################################################

import argparse
import ast
import copy
import os
import sys

from datetime import datetime
from glob     import glob
from natsort  import natsorted

sys.path[0] = '..'
import Hexnet


################################################################################
# Miscellaneous
################################################################################

separator_string = 80 * '#'

def visualize_results_LaTeX(output_dir, compiler='pdflatex'):
	timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')

	report_dats   = natsorted(glob(os.path.join(output_dir, '*_classification_report.dat')))
	accuracy_dats = natsorted(glob(os.path.join(output_dir, '*_accuracy.dat')))
	loss_dats     = natsorted(glob(os.path.join(output_dir, '*_loss.dat')))


	# Test report

	if report_dats:
		report_dats_len = len(report_dats)
		report_tex      = os.path.join(output_dir, f'_Model_Comparison_Report_{timestamp}.tex')
		report_tex_file = open(report_tex, 'w')

		print(
			'\\documentclass[border=1pt]{standalone}\n'
			'\n'
			'\n'
			'\\usepackage{multirow}\n'
			'\n'
			'\n'
			'\\begin{document}\n'
			'    \\begin{tabular}{|c|c|c|c|c|c|c|c|c|}\n'
			'        \\hline\n'
			'        Dataset & Model & Run & Date & Class / Entry & F1 Score & Precision & Recall & Classification Report \\\\\n'
			'        \\hline\n'
			'        \\noalign{\\vskip 2pt}\n'
			'\n'
			'        \\hline',
			 file=report_tex_file)

		for dat_index, dat in enumerate(report_dats):
			dat_basename = os.path.basename(dat)

			dataset               = dat_basename.split('__')[1].split('__')[0].replace('_', '\_')
			model                 = dat_basename.split('__')[0].replace('_', '\_')
			run                   = dat_basename.split('_run')[1].split('_')[0]
			date                  = dat_basename.split('_run')[0].split('__')[-1]
			classification_report = dat_basename.replace('_', '\_')

			with open(dat) as dat_file:
				dat_data = dat_file.read()

			dat_data     = ast.literal_eval(dat_data)
			dat_data_len = len(dat_data)

			for i, (key, value) in enumerate(dat_data.items()):
				print_end_condition = i < dat_data_len - 1 or dat_index == report_dats_len - 1

				if type(value) is dict:
					key       = key.replace('_', '\_')
					f1_score  = float(format(value['f1-score'],  '.8f'))
					precision = float(format(value['precision'], '.8f'))
					recall    = float(format(value['recall'],    '.8f'))

					print(
						f'        {dataset} & {model} & {run} & {date} & {key} & '
							f'{f1_score} & {precision} & {recall} & {classification_report} \\\\\n'
						 '        \\hline\n',
						 end  = ('' if print_end_condition else '        \\noalign{\\vskip 2pt}\n\n        \\hline\n'),
						 file = report_tex_file)
				elif not key.startswith('Hexnet_metric__'):
					key   = key.replace('_', '\_')
					value = float(format(value, '.8f'))

					print(
						f'        {dataset} & {model} & {run} & {date} & {key} & '
							f'\\multicolumn{{3}}{{c|}}{{{value}}} & {classification_report} \\\\\n'
						 '        \\hline\n',
						 end  = ('' if print_end_condition else '        \\noalign{\\vskip 2pt}\n\n        \\hline\n'),
						 file = report_tex_file)

		print(
			'    \\end{tabular}\n'
			'\\end{document}\n',
			 file=report_tex_file)

		report_tex_file.close()


	# Training accuracy

	if accuracy_dats:
		accuracy_dats_len = len(accuracy_dats)
		accuracy_tex      = os.path.join(output_dir, f'_Model_Comparison_Accuracy_{timestamp}.tex')
		accuracy_tex_file = open(accuracy_tex, 'w')

		print(
			'\\documentclass[border=1pt]{standalone}\n'
			'\n'
			'\n'
			'\\usepackage{pgfplots}\n'
			'\n'
			'\n'
			'\\begin{document}\n'
			'    \\begin{tikzpicture}\n'
			'        \\begin{axis}[xlabel={Epoch}, ylabel={Accuracy}, legend pos=outer north east, legend cell align=left]',
			 file=accuracy_tex_file)

		for dat_index, dat in enumerate(accuracy_dats):
			dat          = dat.replace('\\', '/')
			legend_entry = os.path.basename(dat).replace('_', '\_')

			print(
				f'            \\addplot table[x expr=\coordindex, y index=0] {{{dat}}};\n'
				f'            \\addlegendentry{{{legend_entry}}}\n',
				 end = ('\n' if dat_index < accuracy_dats_len - 1 else ''), file = accuracy_tex_file)

		print(
			'        \\end{axis}\n'
			'    \\end{tikzpicture}\n'
			'\\end{document}\n',
			 file=accuracy_tex_file)

		accuracy_tex_file.close()


	# Training loss

	if loss_dats:
		loss_dats_len = len(loss_dats)
		loss_tex      = os.path.join(output_dir, f'_Model_Comparison_Loss_{timestamp}.tex')
		loss_tex_file = open(loss_tex, 'w')

		print(
			'\\documentclass[border=1pt]{standalone}\n'
			'\n'
			'\n'
			'\\usepackage{pgfplots}\n'
			'\n'
			'\n'
			'\\begin{document}\n'
			'    \\begin{tikzpicture}\n'
			'        \\begin{axis}[xlabel={Epoch}, ylabel={Loss}, legend pos=outer north east, legend cell align=left]',
			 file=loss_tex_file)

		for dat_index, dat in enumerate(loss_dats):
			dat          = dat.replace('\\', '/')
			legend_entry = os.path.basename(dat).replace('_', '\_')

			print(
				f'            \\addplot table[x expr=\coordindex, y index=0] {{{dat}}};\n'
				f'            \\addlegendentry{{{legend_entry}}}\n',
				 end = ('\n' if dat_index < loss_dats_len - 1 else ''), file = loss_tex_file)

		print(
			'        \\end{axis}\n'
			'    \\end{tikzpicture}\n'
			'\\end{document}\n',
			 file=loss_tex_file)

		loss_tex_file.close()


	if report_dats or accuracy_dats or loss_dats:
		output_dir = output_dir.replace('\\', '/')

		if report_dats:
			report_tex = report_tex.replace('\\', '/')
			os.system(f'{compiler} -output-directory {output_dir} {report_tex}')

		if accuracy_dats:
			accuracy_tex = accuracy_tex.replace('\\', '/')
			os.system(f'{compiler} -output-directory {output_dir} {accuracy_tex}')

		if loss_dats:
			loss_tex = loss_tex.replace('\\', '/')
			os.system(f'{compiler} -output-directory {output_dir} {loss_tex}')


################################################################################
# Start model comparison
################################################################################

def run(args):
	status = 0


	Hexnet_args = copy.deepcopy(args)
	Hexnet_args.model   = None
	Hexnet_args.dataset = None
	Hexnet_args = Hexnet.parse_args(args=[], namespace=Hexnet_args)


	for dataset in args.dataset:
		Hexnet_args.dataset = dataset

		for model in args.model:
			Hexnet_args.model = model

			Hexnet.Hexnet_print(f'args={Hexnet_args}')
			Hexnet.print_newline()

			status |= Hexnet.run(Hexnet_args)

			print(separator_string)

		print(separator_string)


	visualize_results_LaTeX(args.output_dir)


	return status


################################################################################
# parse_args
################################################################################

def parse_args():
	parser = argparse.ArgumentParser(description='Hexnet: The Hexagonal Machine Learning Module - Model Comparison Test Script')


	parser.add_argument('--model',                            nargs = '+', default = model_s,            help = 'model(s) for training and testing (providing no argument disables training and testing)')
	parser.add_argument('--dataset',                          nargs = '+', default = dataset_s,          help = 'load dataset(s) from HDF5 or directory')
	parser.add_argument('--output-dir',                       nargs = '?', default = output_dir,         help = 'training and test results\' output directory (providing no argument disables the output)')

	parser.add_argument('--batch-size',         type = int,                default = batch_size,         help = 'batch size for training and testing')
	parser.add_argument('--epochs',             type = int,                default = epochs,             help = 'epochs for training')
	parser.add_argument('--runs',               type = int,                default = runs,               help = 'number of training and test runs')
	parser.add_argument('--validation-split',   type = float,              default = validation_split,   help = 'fraction of the training data to be used as validation data')

	parser.add_argument('--resnet-stacks',      type = int,                default = resnet_stacks,      help = 'ResNet models\' number of stacks')
	parser.add_argument('--resnet-n',           type = int,                default = resnet_n,           help = 'ResNet models\' number of residual blocks\' n')
	parser.add_argument('--resnet-filter-size', type = float,              default = resnet_filter_size, help = 'ResNet models\' filter size factor (convolutional layers)')


	return parser.parse_args()


################################################################################
# main
################################################################################

if __name__ == '__main__':
	args = parse_args()

	print(f'args={args}')
	print(separator_string)

	status = run(args)

	sys.exit(status)

