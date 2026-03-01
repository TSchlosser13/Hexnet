#!/usr/bin/env python3.7


'''****************************************************************************
 * Model_Prediction_GUI.py: Model Prediction GUI
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

dataset = 'data/MNIST/MNIST.json'

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
import json
import os
import sys

import numpy      as np
import tensorflow as tf
import tkinter    as tk

from datetime import datetime
from glob     import glob
from PIL      import Image, ImageTk


################################################################################
# Helper functions
################################################################################

def timestamp():
	return datetime.now().strftime('%Y%m%d-%H%M%S')


################################################################################
# Model Prediction GUI
################################################################################

class Model_Prediction_GUI:
	def __init__(self, dataset, args):
		self.dataset = dataset
		self.args    = args


		images = []

		for current_images in self.args['images']:
			images.extend(glob(os.path.join(os.path.dirname(self.dataset), current_images)))

		self.args['images'] = images


		self.title = f'Model Prediction GUI - {self.args["title"]}'

		self.font     = 'Courier'
		self.label_bg = '#dddddd'


		self.load_models()


		self.window = tk.Tk()
		self.window.title(self.title)
		self.window.state('zoomed')

		self.frame1 = tk.Frame(self.window, bg = '#bbbbbb', width = 1000, height = 100)
		self.frame2 = tk.Frame(self.window, bg = '#999999', width =  200, height = 400)
		self.frame3 = tk.Frame(self.window, bg = '#777777', width =  600, height = 400)
		self.frame4 = tk.Frame(self.window, bg = '#555555', width =  200, height = 400)

		self.frame1.pack(side='top')
		self.frame2.pack(side='left', fill='both', expand=True)
		self.frame3.pack(side='left', fill='both', expand=True)
		self.frame4.pack(side='left', fill='both', expand=True)

		self.create_header(self.frame1, self.title)
		self.create_file_browser(self.frame2)
		self.create_visualization(self.frame3)
		self.create_predictions(self.frame4)


		self.window.mainloop()

	def load_models(self):
		self.models = []

		for model in self.args['model_overview']:
			model_to_load = os.path.join(os.path.dirname(self.dataset), self.args['models'][model]['saved_model'])

			print(f'[{timestamp()}] (load_models) Loading model "{model}" ("{model_to_load}")')

			loaded_model = tf.keras.models.load_model(model_to_load)
			loaded_model.compile(optimizer='adam', loss=tf.losses.get('SparseCategoricalCrossentropy'), metrics=['accuracy'])
			loaded_model.summary()

			sys.stdout.write('\n')

			self.models.append(loaded_model)

	def create_header(self, frame, title):
		tk.Label(frame, text = title, bg = self.label_bg, font = (self.font, 32)).pack()

	def create_file_browser(self, frame):
		tk.Label(frame, text = 'File Browser', bg = self.label_bg, font = (self.font, 24)).pack()

		self.file_browser = tk.Listbox(frame, width=66, height=33)

		for image in self.args['images']:
			self.file_browser.insert('end', image)

		self.file_browser.pack(side='left', fill='both')

		self.file_browser_scrollbar = tk.Scrollbar(frame, orient='vertical', command=self.file_browser.yview)
		self.file_browser_scrollbar.pack(side='left', fill='y')
		self.file_browser.config(yscrollcommand=self.file_browser_scrollbar.set)

		self.file_browser.bind('<<ListboxSelect>>', self.file_browser_selected_file_change)

	def file_browser_selected_file_change(self, event):
		selected_file = event.widget.get(event.widget.curselection())

		print(f'[{timestamp()}] (file_browser_selected_file_change) selected_file="{selected_file}"')

		self.visualization_selected_file_change(selected_file)

	def create_visualization(self, frame):
		tk.Label(frame, text = 'Visualization', bg = self.label_bg, font = (self.font, 24)).pack()

		self.visualization = tk.Label(frame)
		self.visualization.pack(fill='both', expand=True)

	def visualization_selected_file_change(self, selected_file):
		print(f'[{timestamp()}] (visualization_selected_file_change) selected_file="{selected_file}"')

		self.image         = Image.open(selected_file)
		self.image_resized = self.image.resize((500, 500), Image.NEAREST)
		self.photo         = ImageTk.PhotoImage(self.image_resized)
		self.visualization.configure(image=self.photo)

		self.predictions_selected_file_change(selected_file)

	def create_predictions(self, frame):
		tk.Label(frame, text = 'Predictions', bg = self.label_bg, font = (self.font, 24)).pack()

		self.predictions_log = tk.Text(frame, width=66, height=33)
		self.predictions_log.pack(side='right', fill='both')

		self.predictions_log_scrollbar = tk.Scrollbar(frame, orient='vertical', command=self.predictions_log.yview)
		self.predictions_log_scrollbar.pack(side='right', fill='y')
		self.predictions_log.config(yscrollcommand=self.predictions_log_scrollbar.set)

	def predictions_selected_file_change(self, selected_file):
		print(f'[{timestamp()}] (predictions_selected_file_change) selected_file="{selected_file}"')

		self.predictions_log.insert('end', f'Predictions for "{selected_file}"\n')

		for current_model_index, current_model in enumerate(self.args['model_overview']):
			self.predictions_log.insert('end', '----------\n')
			self.predictions_log.insert('end', f'{current_model_index + 1}. model - "{current_model}"\n')

			predictions = 100 * self.models[current_model_index].predict(np.expand_dims(np.asarray(self.image)[:, :, :3], axis=0))[0]

			print(f'[{timestamp()}] (predictions_selected_file_change) current_model="{current_model}": predictions = {predictions}')

			for current_class_index, current_class in enumerate(self.args['models'][current_model]['classes']):
				self.predictions_log.insert('end', f'\t{str(current_class_index + 1).rjust(2)}. class - "{current_class}": {str(predictions[current_class_index]).rjust(5)} % confidence\n')

			p  = predictions.argmax()
			pt = f'Prediction: {p + 1}. class - "{self.args["models"][current_model]["classes"][p]}" ({predictions[p]} % confidence)\n'

			self.predictions_log.insert('end', pt)

		self.predictions_log.insert('end', '----------\n----------\n')
		self.predictions_log.see('end')


################################################################################
# parse_args
################################################################################

def parse_args():
	parser = argparse.ArgumentParser(description = 'Model Prediction GUI')

	parser.add_argument('--dataset', default = dataset, help = 'dataset for prediction and visualization')

	return parser.parse_args()


################################################################################
# main
################################################################################

if __name__ == '__main__':
	args = parse_args()

	print(f'args={args}\n')

	with open(args.dataset) as file:
		data = json.load(file)

	Model_Prediction_GUI(args.dataset, data)

