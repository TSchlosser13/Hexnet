#!/usr/bin/env python3.7


'''****************************************************************************
 * astronomical_image_generation.py: Shower Area Generation
 ******************************************************************************
 * v0.1 - 01.09.2020
 *
 * Copyright (c) 2020 Tobias Schlosser (tobias@tobias-schlosser.net)
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

import os
import random

import astropy.units     as u
import matplotlib.pyplot as plt
import numpy             as np

from ctapipe.image         import toymodel
from ctapipe.instrument    import CameraGeometry
from ctapipe.visualization import CameraDisplay
from matplotlib.pyplot     import imsave
from tqdm                  import tqdm


################################################################################
# Parameters
################################################################################

camgeoms = (
	'HESS-I',
	'HESS-II',
	'VERITAS',
	'Whipple109',
	'Whipple151'
)

output_dir = 'astronomical_image_generation'

images_to_generate_per_class = 10

show_and_save_figs = False
increase_verbosity = False


################################################################################
# Initialization
################################################################################

camgeoms_len = len(camgeoms)


################################################################################
# Visualize telescope array image
################################################################################

def visualize(geom, image, title, show_and_save_figs=False, increase_verbosity=False):
	pix_x = geom.pix_x / u.m
	pix_y = geom.pix_y / u.m

	pix_x -= pix_x.min()
	pix_y -= pix_y.min()

	pix_y = pix_y.max() - pix_y

	image_normalized = image - image.min()

	if image_normalized.max():
		image_normalized = (255 * (image_normalized / image_normalized.max())).astype(np.uint8)
	else:
		image_normalized = image_normalized.astype(np.uint8)


	pix_x_unique_sorted = np.sort(np.unique(pix_x))
	pix_y_unique_sorted = np.sort(np.unique(pix_y))

	x_diff = pix_x_unique_sorted[2] - pix_x_unique_sorted[0]
	y_diff = pix_y_unique_sorted[1] - pix_y_unique_sorted[0]


	pixels_per_row = []

	for y in pix_y_unique_sorted:
		pixels_per_row.append(len(pix_x[np.where(pix_y == y)[0]]))

	rows = len(pixels_per_row)
	cols = max(pixels_per_row)

	if increase_verbosity:
		print(f'\t> rows x cols = {rows} x {cols}')


	rows_is_even = 1 - rows % 2
	cols_is_even = 1 - cols % 2

	center            = int(rows / 2)
	center_is_odd_row = center % 2


	image_out      = np.zeros(shape = (rows, cols + rows_is_even * cols_is_even))
	image_out_list = []

	for y, x, pixel_value in zip(pix_y, pix_x, image_normalized):
		y_index      = int(y / y_diff + 0.1)
		y_is_odd_row = y_index % 2

		if y_is_odd_row:
			x_index = int(x / x_diff + 0.1)
		else:
			if cols_is_even:
				x_index = int(x / x_diff + 0.1 + (1 - center_is_odd_row))
			else:
				x_index = int(x / x_diff + 0.1 + center_is_odd_row)

		image_out[y_index][x_index] = pixel_value
		image_out_list.append(pixel_value)

	if increase_verbosity:
		print(f'\t> image_out =\n{image_out}')


	if show_and_save_figs:
		disp = CameraDisplay(geom, image=image_normalized)
		plt.show(disp)

		disp = CameraDisplay(geom, image=np.array(image_out_list))

		plt.savefig(f'{title}.png')
		plt.savefig(f'{title}.pdf')

		plt.show(disp)

		plt.close()

	imsave(f'{title}_hex.png', image_out, cmap='gray')


################################################################################
# Generate telescope array images
################################################################################

def test_ctapipe_image_generation():
	for camgeom_index, camgeom in enumerate(camgeoms):
		print(f'> ({camgeom_index + 1:{len(str(camgeoms_len))}}/{camgeoms_len}) camgeom={camgeom}')

		output_dir_camgeom = os.path.join(output_dir, camgeom)
		os.makedirs(output_dir_camgeom, exist_ok=True)

		geom = CameraGeometry.from_name(camgeom)


		# Class 1: no shower areas

		print('\t> Class 1: no shower areas')

		camgeom_class = os.path.join(output_dir_camgeom, '1_no_shower_areas')
		os.makedirs(camgeom_class, exist_ok=True)

		for image_index in tqdm(range(images_to_generate_per_class)):
			image = np.random.poisson(lam=0.25, size=geom.n_pixels)
			title = os.path.join(camgeom_class, f'{camgeom}_image{str(image_index).zfill(len(str(images_to_generate_per_class)))}')

			visualize(geom, image, title, show_and_save_figs, increase_verbosity)


		# Class 2: single shower area

		print('\t> Class 2: single shower area')

		camgeom_class = os.path.join(output_dir_camgeom, '2_single_shower_area')
		os.makedirs(camgeom_class, exist_ok=True)

		for image_index in tqdm(range(images_to_generate_per_class)):
			title = os.path.join(camgeom_class, f'{camgeom}_image{str(image_index).zfill(len(str(images_to_generate_per_class)))}')

			x = np.random.uniform(0.8 * min(geom.pix_x.value), 0.8 * max(geom.pix_x.value))
			y = np.random.uniform(0.8 * min(geom.pix_y.value), 0.8 * max(geom.pix_y.value))

			model = toymodel.Gaussian(
				x      = y                                    * u.m,
				y      = y                                    * u.m,
				width  = np.random.uniform( 0.05, 0.075     ) * u.m,
				length = np.random.uniform( 0.1,  0.15      ) * u.m,
				psi    = np.random.uniform( 0,    2 * np.pi ) * u.rad)

			image, _, _ = model.generate_image(
				geom,
				intensity    = np.random.uniform(1000, 3000),
				nsb_level_pe = 5)

			visualize(geom, image, title, show_and_save_figs, increase_verbosity)


		# Class 3: multiple shower areas (2 to 9)

		print('\t> Class 3: multiple shower areas (2 to 9)')

		camgeom_class = os.path.join(output_dir_camgeom, '3_multiple_shower_areas')
		os.makedirs(camgeom_class, exist_ok=True)

		for image_index in tqdm(range(images_to_generate_per_class)):
			image = np.zeros(geom.n_pixels)
			title = os.path.join(camgeom_class, f'{camgeom}_image{str(image_index).zfill(len(str(images_to_generate_per_class)))}')

			areas_to_generate_per_image = random.randint(2, 9)

			for _ in range(areas_to_generate_per_image):
				model = toymodel.Gaussian(
					x      = np.random.uniform( 0.8 * min(geom.pix_x.value), 0.8 * max(geom.pix_x.value) ) * u.m,
					y      = np.random.uniform( 0.8 * min(geom.pix_y.value), 0.8 * max(geom.pix_y.value) ) * u.m,
					width  = np.random.uniform( 0.05,                        0.075                       ) * u.m,
					length = np.random.uniform( 0.1,                         0.15                        ) * u.m,
					psi    = np.random.uniform( 0,                           2 * np.pi                   ) * u.rad)

				new_image, _, _ = model.generate_image(
					geom,
					intensity    = np.random.uniform(1000, 3000),
					nsb_level_pe = 5)

				image += new_image

			visualize(geom, image, title, show_and_save_figs, increase_verbosity)


################################################################################
# main
################################################################################

if __name__ == '__main__':
	os.makedirs(output_dir, exist_ok=True)

	test_ctapipe_image_generation()


