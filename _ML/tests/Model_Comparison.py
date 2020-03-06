#!/usr/bin/env python3.7


model_s    = ['SResNet_v2', 'HResNet_v2']
dataset_s  = ['../datasets/CIFAR/CIFAR-10_s2s.h5', '../datasets/CIFAR/CIFAR-10_s2h.h5']
tests_dir  = 'tmp'

runs       =  2
epochs     =  2
batch_size = 32


import argparse
import os

import sys
sys.path.append('..')

import Hexnet

from glob import glob


separator_string = 80 * '#'

status = 1


def parse_args():
	parser = argparse.ArgumentParser(description='Hexnet: The Hexagonal Machine Learning Module - Model Comparison')


	parser.add_argument('--model',                  nargs = '+', default = model_s,    help = 'model(s) to train and test (providing no argument disables training and testing)')
	parser.add_argument('--dataset',                nargs = '+', default = dataset_s,  help = 'load dataset(s) from file or directory')
	parser.add_argument('--tests-dir',              nargs = '?', default = tests_dir,  help = 'tests output directory (providing no argument disables the tests output)')

	parser.add_argument('--runs',       type = int,              default = runs,       help = 'training runs')
	parser.add_argument('--epochs',     type = int,              default = epochs,     help = 'training epochs')
	parser.add_argument('--batch-size', type = int,              default = batch_size, help = 'training batch size')


	return parser.parse_args()


args = parse_args()
print(f'args={args}')
print(separator_string)

Hexnet_args = Hexnet.parse_args()
Hexnet_args.tests_dir  = args.tests_dir
Hexnet_args.runs       = args.runs
Hexnet_args.epochs     = args.epochs
Hexnet_args.batch_size = args.batch_size


for dataset in args.dataset:
	Hexnet_args.dataset = dataset

	for model in args.model:
		Hexnet_args.model = model

		Hexnet.Hexnet_print(f'args={Hexnet_args}')
		Hexnet.print_newline()

		status &= Hexnet.run(Hexnet_args)

		print(separator_string)

	print(separator_string)




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

for dat in sorted(glob(os.path.join(tests_dir, '*_accuracy.dat'))):
	dat          = dat.replace('\\', '/')
	legend_entry = os.path.basename(dat).replace('_', '\_')

	print(
		f'            \\addplot table[x expr=\coordindex, y index=0] {{{dat}}};\n'
		f'            \\addlegendentry{{{legend_entry}}}\n',
		 file=accuracy_tex_file)

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

for dat in sorted(glob(os.path.join(tests_dir, '*_loss.dat'))):
	dat          = dat.replace('\\', '/')
	legend_entry = os.path.basename(dat).replace('_', '\_')

	print(
		f'            \\addplot table[x expr=\coordindex, y index=0] {{{dat}}};\n'
		f'            \\addlegendentry{{{legend_entry}}}\n',
		 file=loss_tex_file)

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


