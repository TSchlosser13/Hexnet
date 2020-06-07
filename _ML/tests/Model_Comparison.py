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


model_s    = ['SResNet_v2', 'HResNet_v2']
dataset_s  = ['../datasets/CIFAR/CIFAR-10_s2s.h5', '../datasets/CIFAR/CIFAR-10_s2h.h5']
output_dir = 'tmp'

batch_size       = 32
epochs           =  2
runs             =  2
validation_split =  0.1


import argparse
import copy
import os
import sys

from glob    import glob
from natsort import natsorted

sys.path.append('..')
import Hexnet


separator_string = 80 * '#'

status = 0


def parse_args():
	parser = argparse.ArgumentParser(description='Hexnet: The Hexagonal Machine Learning Module - Model Comparison Test Script')


	parser.add_argument('--model',                          nargs = '+', default = model_s,          help = 'model(s) for training and testing (providing no argument disables training and testing)')
	parser.add_argument('--dataset',                        nargs = '+', default = dataset_s,        help = 'load dataset(s) from file or directory')
	parser.add_argument('--output-dir',                     nargs = '?', default = output_dir,       help = 'training and test results\' output directory (providing no argument disables the output)')

	parser.add_argument('--batch-size',       type = int,                default = batch_size,       help = 'training batch size')
	parser.add_argument('--epochs',           type = int,                default = epochs,           help = 'training epochs')
	parser.add_argument('--runs',             type = int,                default = runs,             help = 'training runs')
	parser.add_argument('--validation-split', type = float,              default = validation_split, help = 'fraction of the training data to be used as validation data')


	return parser.parse_args()


args = parse_args()
print(f'args={args}')
print(separator_string)

Hexnet_args = copy.deepcopy(args)
Hexnet_args.model   = ''
Hexnet_args.dataset = ''
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




accuracy_dats = natsorted(glob(os.path.join(output_dir, '*_accuracy.dat')))
loss_dats     = natsorted(glob(os.path.join(output_dir, '*_loss.dat')))

accuracy_dats_len = len(accuracy_dats)
loss_dats_len     = len(loss_dats)

accuracy_tex = 'Model_Comparison_Accuracy.tex'
loss_tex     = 'Model_Comparison_Loss.tex'

accuracy_tex_file = open(accuracy_tex, 'w')
loss_tex_file     = open(loss_tex,     'w')


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


accuracy_tex_file.close()
loss_tex_file.close()

os.system(f'pdflatex {accuracy_tex}')
os.system(f'pdflatex {loss_tex}')




sys.exit(status)


