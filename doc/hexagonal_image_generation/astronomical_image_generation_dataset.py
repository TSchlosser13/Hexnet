#!/usr/bin/env python3.7


'''****************************************************************************
 * astronomical_image_generation_dataset.py: Dataset Generation
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

import math
import os
import random

import astropy.units as u
import numpy         as np

from ctapipe.image      import dilate, tailcuts_clean, toymodel
from ctapipe.instrument import CameraGeometry
from tqdm               import tqdm

from astronomical_image_generation import visualize


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

models = {
	'Gaussian'       : toymodel.Gaussian,
	'SkewedGaussian' : toymodel.SkewedGaussian,
	'RingGaussian'   : toymodel.RingGaussian
}

output_dir = 'astronomical_image_generation_dataset'

images_to_generate_per_class = 1000
intensity_range              = (1000, 3000)
nsb_level_pe                 = 5

show_and_save_figs = False
increase_verbosity = False


################################################################################
# Initialization
################################################################################

camgeoms_len = len(camgeoms)
models_len   = len(models)


################################################################################
# Postprocess and visualize telescope array image
################################################################################

def postprocess_and_visualize(geom, image, title, show_and_save_figs=False, increase_verbosity=False):
	visualize(geom, image, title, show_and_save_figs, increase_verbosity)

	cleanmask                 = tailcuts_clean(geom, image, picture_thresh=10, boundary_thresh=5)
	image_cleaned             = image.copy()
	image_cleaned[~cleanmask] = 0.0

	visualize(geom, image_cleaned, f'{title}_cleaned', show_and_save_figs, increase_verbosity)

	cleanmask_dilated                         = dilate(geom, cleanmask)
	image_cleaned_dilated                     = image.copy()
	image_cleaned_dilated[~cleanmask_dilated] = 0.0

	visualize(geom, image_cleaned_dilated, f'{title}_cleaned_dilated', show_and_save_figs, increase_verbosity)

	cleanmasks_visualized = cleanmask.astype(int) + cleanmask_dilated.astype(int)

	visualize(geom, cleanmasks_visualized, f'{title}_cleanmasks', show_and_save_figs, increase_verbosity)


################################################################################
# Generate telescope array image dataset
################################################################################

if __name__ == '__main__':
	os.makedirs(output_dir, exist_ok=True)

	data_file = open(os.path.join(output_dir, f'{output_dir}.csv'), 'w')

	print(
		'class index,'
		'class label,'
		'filename,'
		'number of centroids,'
		'centroid in meters (x,y),'
		'width in meters,'
		'length in meters,'
		'orientation in radians,'
		'skewness,'
		'outer radius in meters,'
		'inner radius in meters,'
		'intensity,'
		'nsb_level_pe',
		file=data_file)

	for camgeom_index, camgeom in enumerate(camgeoms):
		class_index = 1

		print(f'> ({camgeom_index + 1:{len(str(camgeoms_len))}}/{camgeoms_len}) camgeom={camgeom}')

		output_dir_camgeom = os.path.join(output_dir, camgeom)
		os.makedirs(output_dir_camgeom, exist_ok=True)

		geom = CameraGeometry.from_name(camgeom)


		# Class 1: no shower areas

		print('\t\t> Class 1: no shower areas')

		class_label = f'{class_index}_no_shower_areas'

		camgeom_class = os.path.join(output_dir_camgeom, class_label)
		os.makedirs(camgeom_class, exist_ok=True)

		for image_index in tqdm(range(images_to_generate_per_class)):
			image        = np.random.poisson(lam=0.25, size=geom.n_pixels)
			image_string = str(image_index).zfill(len(str(images_to_generate_per_class)))
			filename     = f'{camgeom}_image{image_string}'
			title        = os.path.join(camgeom_class, filename)

			print(
				f'{class_index},'
				f'{class_label},'
				f'{filename},'
				f'0,'
				f'-,'
				f'-,'
				f'-,'
				f'-,'
				f'-,'
				f'-,'
				f'-,'
				f'-,'
				f'-',
				file=data_file)

			postprocess_and_visualize(geom, image, title, show_and_save_figs, increase_verbosity)

		class_index += 1


		for model_index, (model_name, model_function) in enumerate(models.items()):
			print(f'\t> ({model_index + 1:{len(str(models_len))}}/{models_len}) model_name={model_name}')

			if model_function is models['Gaussian'] or model_function is models['SkewedGaussian']:
				min_centroid_distance = 0.125
			else: # models['RingGaussian']
				min_centroid_distance = 0.3


			# Class 2: single shower area

			print('\t\t> Class 2: single shower area')

			class_label = f'{class_index}_single_shower_area_{model_name}'

			camgeom_class = os.path.join(output_dir_camgeom, class_label)
			os.makedirs(camgeom_class, exist_ok=True)

			for image_index in tqdm(range(images_to_generate_per_class)):
				image_string = str(image_index).zfill(len(str(images_to_generate_per_class)))
				intensity    = intensity_range[0] + image_index * (intensity_range[1] - intensity_range[0]) / images_to_generate_per_class
				filename     = f'{camgeom}_image{image_string}_intensity{intensity:.1f}'
				title        = os.path.join(camgeom_class, filename)

				x = np.random.uniform(0.8 * min(geom.pix_x.value), 0.8 * max(geom.pix_x.value))
				y = np.random.uniform(0.8 * min(geom.pix_y.value), 0.8 * max(geom.pix_y.value))

				if model_function is models['Gaussian']:
					width       = np.random.uniform( 0.05, 0.075     )
					length      = np.random.uniform( 0.1,  0.15      )
					orientation = np.random.uniform( 0,    2 * np.pi )

					print(
						f'{class_index},'
						f'{class_label},'
						f'{filename},'
						f'1,'
						f'({x:.8f},{y:.8f}),'
						f'{width:.8f},'
						f'{length:.8f},'
						f'{orientation:.8f},'
						f'-,'
						f'-,'
						f'-,'
						f'{intensity:.1f},'
						f'{nsb_level_pe}',
						file=data_file)

					model = model_function(
						x      = x           * u.m,
						y      = y           * u.m,
						width  = width       * u.m,
						length = length      * u.m,
						psi    = orientation * u.rad)
				elif model_function is models['SkewedGaussian']:
					width       = np.random.uniform( 0.05, 0.075     )
					length      = np.random.uniform( 0.1,  0.15      )
					orientation = np.random.uniform( 0,    2 * np.pi )
					skewness    =    random.uniform( 0.1,  0.9       )

					print(
						f'{class_index},'
						f'{class_label},'
						f'{filename},'
						f'1,'
						f'({x:.8f},{y:.8f}),'
						f'{width:.8f},'
						f'{length:.8f},'
						f'{orientation:.8f},'
						f'{skewness:.8f},'
						f'-,'
						f'-,'
						f'{intensity:.1f},'
						f'{nsb_level_pe}',
						file=data_file)

					model = model_function(
						x        = x           * u.m,
						y        = y           * u.m,
						width    = width       * u.m,
						length   = length      * u.m,
						psi      = orientation * u.rad,
						skewness = skewness)
				else: # models['RingGaussian']
					inner_radius = np.random.uniform( 0.1,  0.5  )
					outer_radius = np.random.uniform( 0.05, 0.25 )

					print(
						f'{class_index},'
						f'{class_label},'
						f'{filename},'
						f'1,'
						f'({x:.8f},{y:.8f}),'
						f'-,'
						f'-,'
						f'-,'
						f'-,'
						f'{outer_radius:.8f},'
						f'{inner_radius:.8f},'
						f'{intensity:.1f},'
						f'{nsb_level_pe}',
						file=data_file)

					model = model_function(
						x      = x            * u.m,
						y      = y            * u.m,
						radius = inner_radius * u.m,
						sigma  = outer_radius * u.m)

				image, _, _ = model.generate_image(
					geom,
					intensity,
					nsb_level_pe = nsb_level_pe)

				postprocess_and_visualize(geom, image, title, show_and_save_figs, increase_verbosity)

			class_index += 1


			# Class 3: multiple shower areas (2 to 9)

			print('\t\t> Class 3: multiple shower areas (2 to 9)')

			class_label = f'{class_index}_multiple_shower_areas_{model_name}'

			camgeom_class = os.path.join(output_dir_camgeom, class_label)
			os.makedirs(camgeom_class, exist_ok=True)

			for image_index in tqdm(range(images_to_generate_per_class)):
				image        = np.zeros(geom.n_pixels)
				image_string = str(image_index).zfill(len(str(images_to_generate_per_class)))
				intensity    = intensity_range[0] + image_index * (intensity_range[1] - intensity_range[0]) / images_to_generate_per_class

				areas_to_generate_per_image = random.randint(2, 9)
				areas_generated_per_image   = 0
				iteration_cnt               = 1

				centroid_list     = []
				width_list        = []
				length_list       = []
				orientation_list  = []
				skewness_list     = []
				outer_radius_list = []
				inner_radius_list = []

				while areas_generated_per_image < areas_to_generate_per_image:
					x = np.random.uniform(0.8 * min(geom.pix_x.value), 0.8 * max(geom.pix_x.value))
					y = np.random.uniform(0.8 * min(geom.pix_y.value), 0.8 * max(geom.pix_y.value))

					iteration_cnt += 1

					if any(math.sqrt((x - c[0])**2 + (y - c[1])**2) < min_centroid_distance for c in centroid_list) and iteration_cnt < 10:
						continue

					areas_generated_per_image += 1

					centroid_list.append((x, y))

					if model_function is models['Gaussian']:
						width       = np.random.uniform( 0.05, 0.075     )
						length      = np.random.uniform( 0.1,  0.15      )
						orientation = np.random.uniform( 0,    2 * np.pi )

						width_list.append(width)
						length_list.append(length)
						orientation_list.append(orientation)

						model = model_function(
							x        = x           * u.m,
							y        = y           * u.m,
							width    = width       * u.m,
							length   = length      * u.m,
							psi      = orientation * u.rad)
					elif model_function is models['SkewedGaussian']:
						width       = np.random.uniform( 0.05, 0.075     )
						length      = np.random.uniform( 0.1,  0.15      )
						orientation = np.random.uniform( 0,    2 * np.pi )
						skewness    =    random.uniform( 0.1,  0.9       )

						width_list.append(width)
						length_list.append(length)
						orientation_list.append(orientation)
						skewness_list.append(skewness)

						model = model_function(
							x        = x           * u.m,
							y        = y           * u.m,
							width    = width       * u.m,
							length   = length      * u.m,
							psi      = orientation * u.rad,
							skewness = skewness)
					else: # models['RingGaussian']
						inner_radius = np.random.uniform( 0.1,  0.5  )
						outer_radius = np.random.uniform( 0.05, 0.25 )

						outer_radius_list.append(outer_radius)
						inner_radius_list.append(inner_radius)

						model = model_function(
							x      = x            * u.m,
							y      = y            * u.m,
							radius = inner_radius * u.m,
							sigma  = outer_radius * u.m)

					new_image, _, _ = model.generate_image(
						geom,
						intensity,
						nsb_level_pe = nsb_level_pe)

					image += new_image

				centroid_list_len = len(centroid_list)

				filename = f'{camgeom}_image{image_string}_intensity{intensity:.1f}_centroids{centroid_list_len}'
				title    = os.path.join(camgeom_class, filename)

				if model_function is models['Gaussian']:
					print(
						f'{class_index},'
						f'{class_label},'
						f'{filename},'
						f'{centroid_list_len},'
						f'{[(format(x, ".8f"), format(y, ".8f")) for (x, y) in centroid_list]},'
						f'{[format(width,       ".8f") for width       in width_list]},'
						f'{[format(length,      ".8f") for length      in length_list]},'
						f'{[format(orientation, ".8f") for orientation in orientation_list]},'
						f'-,'
						f'-,'
						f'-,'
						f'{intensity:.1f},'
						f'{nsb_level_pe}',
						file=data_file)
				elif model_function is models['SkewedGaussian']:
					print(
						f'{class_index},'
						f'{class_label},'
						f'{filename},'
						f'{centroid_list_len},'
						f'{[(format(x, ".8f"), format(y, ".8f")) for (x, y) in centroid_list]},'
						f'{[format(width,       ".8f") for width       in width_list]},'
						f'{[format(length,      ".8f") for length      in length_list]},'
						f'{[format(orientation, ".8f") for orientation in orientation_list]},'
						f'{[format(skewness,    ".8f") for skewness    in skewness_list]},'
						f'-,'
						f'-,'
						f'{intensity:.1f},'
						f'{nsb_level_pe}',
						file=data_file)
				else: # models['RingGaussian']
					print(
						f'{class_index},'
						f'{class_label},'
						f'{filename},'
						f'{centroid_list_len},'
						f'{[(format(x, ".8f"), format(y, ".8f")) for (x, y) in centroid_list]},'
						f'-,'
						f'-,'
						f'-,'
						f'-,'
						f'{[format(outer_radius, ".8f") for outer_radius in outer_radius_list]},'
						f'{[format(inner_radius, ".8f") for inner_radius in inner_radius_list]},'
						f'{intensity:.1f},'
						f'{nsb_level_pe}',
						file=data_file)

				postprocess_and_visualize(geom, image, title, show_and_save_figs, increase_verbosity)

			class_index += 1

	data_file.close()


