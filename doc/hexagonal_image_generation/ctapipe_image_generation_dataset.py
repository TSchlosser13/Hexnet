#!/usr/bin/env python3.7


import math
import os
import random

import astropy.units as u
import numpy         as np

from ctapipe.image      import dilate, tailcuts_clean, toymodel
from ctapipe.instrument import CameraGeometry
from tqdm               import tqdm

from ctapipe_image_generation import visualize


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

output_dir = 'ctapipe_image_generation_dataset'

images_to_generate_per_class = 1000
intensity_range              = (1000, 3000)

show_and_save_figs = False
increase_verbosity = False


camgeoms_len = len(camgeoms)
models_len   = len(models)


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


if __name__ == '__main__':
	os.makedirs(output_dir, exist_ok=True)

	for camgeom_index, camgeom in enumerate(camgeoms):
		print(f'> ({camgeom_index + 1:{len(str(camgeoms_len))}}/{camgeoms_len}) camgeom={camgeom}')

		output_dir_camgeom = os.path.join(output_dir, camgeom)
		os.makedirs(output_dir_camgeom, exist_ok=True)

		geom = CameraGeometry.from_name(camgeom)


		# Class 0: no shower areas

		print('\t\t> Class 0: no shower areas')

		camgeom_class = os.path.join(output_dir_camgeom, '0_no_shower_areas')
		os.makedirs(camgeom_class, exist_ok=True)

		for image_index in tqdm(range(images_to_generate_per_class)):
			image        = np.random.poisson(lam=0.25, size=geom.n_pixels)
			image_string = str(image_index).zfill(len(str(images_to_generate_per_class)))
			title        = os.path.join(camgeom_class, f'{camgeom}_image{image_string}')

			postprocess_and_visualize(geom, image, title, show_and_save_figs, increase_verbosity)


		for model_index, (model_name, model_function) in enumerate(models.items()):
			print(f'\t> ({model_index + 1:{len(str(models_len))}}/{models_len}) model_name={model_name}')

			if model_function is models['Gaussian'] or models['SkewedGaussian']:
				min_centroid_distance = 0.125
			else: # models['RingGaussian']
				min_centroid_distance = 0.3


			# Class 1: single shower area

			print('\t\t> Class 1: single shower area')

			camgeom_class = os.path.join(output_dir_camgeom, f'1_single_shower_area_{model_name}')
			os.makedirs(camgeom_class, exist_ok=True)

			for image_index in tqdm(range(images_to_generate_per_class)):
				image_string = str(image_index).zfill(len(str(images_to_generate_per_class)))
				intensity    = intensity_range[0] + image_index * (intensity_range[1] - intensity_range[0]) / images_to_generate_per_class
				title        = os.path.join(camgeom_class, f'{camgeom}_image{image_string}_intensity{intensity:.1f}')

				x = np.random.uniform(0.8 * min(geom.pix_x.value), 0.8 * max(geom.pix_x.value))
				y = np.random.uniform(0.8 * min(geom.pix_y.value), 0.8 * max(geom.pix_y.value))

				if model_function is models['Gaussian']:
					model = model_function(
						x      = x                                    * u.m,
						y      = y                                    * u.m,
						width  = np.random.uniform( 0.05, 0.075     ) * u.m,
						length = np.random.uniform( 0.1,  0.15      ) * u.m,
						psi    = np.random.uniform( 0,    2 * np.pi ) * u.rad)
				elif model_function is models['SkewedGaussian']:
					model = model_function(
						x      = x                                      * u.m,
						y      = y                                      * u.m,
						width    = np.random.uniform( 0.05, 0.075     ) * u.m,
						length   = np.random.uniform( 0.1,  0.15      ) * u.m,
						psi      = np.random.uniform( 0,    2 * np.pi ) * u.rad,
						skewness =    random.uniform( 0.1,  0.9       ))
				else: # models['RingGaussian']
					model = model_function(
						x      = x                               * u.m,
						y      = y                               * u.m,
						radius = np.random.uniform( 0.1,  0.5  ) * u.m,
						sigma  = np.random.uniform( 0.05, 0.25 ) * u.m)

				image, _, _ = model.generate_image(
					geom,
					intensity,
					nsb_level_pe = 5)

				postprocess_and_visualize(geom, image, title, show_and_save_figs, increase_verbosity)


			# Class 2: multiple shower areas (2 to 9)

			print('\t\t> Class 2: multiple shower areas (2 to 9)')

			camgeom_class = os.path.join(output_dir_camgeom, f'2_multiple_shower_areas_(2_to_9)_{model_name}')
			os.makedirs(camgeom_class, exist_ok=True)

			for image_index in tqdm(range(images_to_generate_per_class)):
				image        = np.zeros(geom.n_pixels)
				image_string = str(image_index).zfill(len(str(images_to_generate_per_class)))
				intensity    = intensity_range[0] + image_index * (intensity_range[1] - intensity_range[0]) / images_to_generate_per_class

				areas_to_generate_per_image = random.randint(2, 9)

				centroids = []

				for _ in range(areas_to_generate_per_image):
					x = np.random.uniform(0.8 * min(geom.pix_x.value), 0.8 * max(geom.pix_x.value))
					y = np.random.uniform(0.8 * min(geom.pix_y.value), 0.8 * max(geom.pix_y.value))

					if any(math.sqrt((x - c[0])**2 + (y - c[1])**2) < min_centroid_distance for c in centroids):
						continue

					centroids.append((x, y))

					if model_function is models['Gaussian']:
						model = model_function(
							x      = x                                    * u.m,
							y      = y                                    * u.m,
							width  = np.random.uniform( 0.05, 0.075     ) * u.m,
							length = np.random.uniform( 0.1,  0.15      ) * u.m,
							psi    = np.random.uniform( 0,    2 * np.pi ) * u.rad)
					elif model_function is models['SkewedGaussian']:
						model = model_function(
							x        = x                                    * u.m,
							y        = y                                    * u.m,
							width    = np.random.uniform( 0.05, 0.075     ) * u.m,
							length   = np.random.uniform( 0.1,  0.15      ) * u.m,
							psi      = np.random.uniform( 0,    2 * np.pi ) * u.rad,
							skewness =    random.uniform( 0.1,  0.9       ))
					else: # models['RingGaussian']
						model = model_function(
							x      = x                               * u.m,
							y      = y                               * u.m,
							radius = np.random.uniform( 0.1,  0.5  ) * u.m,
							sigma  = np.random.uniform( 0.05, 0.25 ) * u.m)

					new_image, _, _ = model.generate_image(
						geom,
						intensity,
						nsb_level_pe = 5)

					image += new_image

				title = os.path.join(camgeom_class, f'{camgeom}_image{image_string}_intensity{intensity:.1f}_centroids{len(centroids)}')

				postprocess_and_visualize(geom, image, title, show_and_save_figs, increase_verbosity)

