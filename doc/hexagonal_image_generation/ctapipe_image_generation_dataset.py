#!/usr/bin/env python3.7


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

images_to_generate_per_class = 10000

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


		for model_index, (model_name, model_function) in enumerate(models.items()):
			print(f'\t> ({model_index + 1:{len(str(models_len))}}/{models_len}) model_name={model_name}')


			# Class 0: no areas of photons

			if not model_index:
				print('\t\t> Class 0: no areas of photons')

				camgeom_class = os.path.join(output_dir_camgeom, '0_areas')
				os.makedirs(camgeom_class, exist_ok=True)

				for image_index in tqdm(range(images_to_generate_per_class)):
					image = np.zeros(geom.n_pixels)
					title = os.path.join(camgeom_class, f'{camgeom}_image{str(image_index).zfill(len(str(images_to_generate_per_class)))}')

					postprocess_and_visualize(geom, image, title, show_and_save_figs, increase_verbosity)


			# Class 1: 1 area of photons

			print('\t\t> Class 1: 1 area of photons')

			camgeom_class = os.path.join(output_dir_camgeom, f'1_area_{model_name}')
			os.makedirs(camgeom_class, exist_ok=True)

			for image_index in tqdm(range(images_to_generate_per_class)):
				title = os.path.join(camgeom_class, f'{camgeom}_image{str(image_index).zfill(len(str(images_to_generate_per_class)))}')

				if model_function is models['Gaussian']:
					model = model_function(
						x      = np.random.uniform(-0.8,  0.8)       * u.m,
						y      = np.random.uniform(-0.8,  0.8)       * u.m,
						width  = np.random.uniform( 0.05, 0.075)     * u.m,
						length = np.random.uniform( 0.1,  0.15)      * u.m,
						psi    = np.random.uniform( 0,    2 * np.pi) * u.rad)
				elif model_function is models['SkewedGaussian']:
					model = model_function(
						x        = np.random.uniform(-0.8,  0.8)       * u.m,
						y        = np.random.uniform(-0.8,  0.8)       * u.m,
						width    = np.random.uniform( 0.05, 0.075)     * u.m,
						length   = np.random.uniform( 0.1,  0.15)      * u.m,
						psi      = np.random.uniform( 0,    2 * np.pi) * u.rad,
						skewness =    random.uniform( 0.1,  0.9))
				else: # models['RingGaussian']
					model = model_function(
						x      = np.random.uniform(-0.8,  0.8)  * u.m,
						y      = np.random.uniform(-0.8,  0.8)  * u.m,
						radius = np.random.uniform( 0.1,  0.5)  * u.m,
						sigma  = np.random.uniform( 0.05, 0.25) * u.m)

				image, _, _ = model.generate_image(
					geom,
					intensity    = np.random.uniform(1000, 3000),
					nsb_level_pe = 5)

				postprocess_and_visualize(geom, image, title, show_and_save_figs, increase_verbosity)


			# Class 2: 2 to 9 areas of photons

			print('\t\t> Class 2: 2 to 9 areas of photons')

			camgeom_class = os.path.join(output_dir_camgeom, f'2-9_areas_{model_name}')
			os.makedirs(camgeom_class, exist_ok=True)

			for image_index in tqdm(range(images_to_generate_per_class)):
				image = np.zeros(geom.n_pixels)
				title = os.path.join(camgeom_class, f'{camgeom}_image{str(image_index).zfill(len(str(images_to_generate_per_class)))}')

				areas_to_generate_per_image = random.randint(2, 9)

				for _ in range(areas_to_generate_per_image):
					if model_function is models['Gaussian']:
						model = model_function(
							x      = np.random.uniform(-0.8,  0.8)       * u.m,
							y      = np.random.uniform(-0.8,  0.8)       * u.m,
							width  = np.random.uniform( 0.05, 0.075)     * u.m,
							length = np.random.uniform( 0.1,  0.15)      * u.m,
							psi    = np.random.uniform( 0,    2 * np.pi) * u.rad)
					elif model_function is models['SkewedGaussian']:
						model = model_function(
							x        = np.random.uniform(-0.8,  0.8)       * u.m,
							y        = np.random.uniform(-0.8,  0.8)       * u.m,
							width    = np.random.uniform( 0.05, 0.075)     * u.m,
							length   = np.random.uniform( 0.1,  0.15)      * u.m,
							psi      = np.random.uniform( 0,    2 * np.pi) * u.rad,
							skewness =    random.uniform( 0.1,  0.9))
					else: # models['RingGaussian']
						model = model_function(
							x      = np.random.uniform(-0.8,  0.8)  * u.m,
							y      = np.random.uniform(-0.8,  0.8)  * u.m,
							radius = np.random.uniform( 0.1,  0.5)  * u.m,
							sigma  = np.random.uniform( 0.05, 0.25) * u.m)

					new_image, _, _ = model.generate_image(
						geom,
						intensity    = np.random.uniform(1000, 3000),
						nsb_level_pe = 5)

					image += new_image

				postprocess_and_visualize(geom, image, title, show_and_save_figs, increase_verbosity)

