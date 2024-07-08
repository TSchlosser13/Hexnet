#!/usr/bin/env python3.7


################################################################################
# Imports
################################################################################

import os

import astropy.units     as u
import matplotlib.pyplot as plt
import numpy             as np

from ctapipe.image         import toymodel
from ctapipe.instrument    import CameraGeometry
from ctapipe.visualization import CameraDisplay
from matplotlib.pyplot     import imsave


################################################################################
# Parameters
################################################################################

camgeoms = (
	'ASTRICam',
	'CHEC',
	'DigiCam',
	'FACT',
	'FlashCam',
	# 'hess',
	'HESS-I',
	'HESS-II',
	'LSTCam',
	'LSTCam-002',
	'LSTCam-003',
	'MAGICCam',
	'MAGICCamMars',
	'NectarCam',
	'NectarCam-003',
	'SCTCam',
	'VERITAS',
	'Whipple109',
	'Whipple151',
	'Whipple331',
	'Whipple490'
)

output_dir = 'ctapipe_image_generation_test'

plt.rcParams.update({'font.size': 20})


################################################################################
# Initialization
################################################################################

camgeoms_len = len(camgeoms)

os.makedirs(output_dir, exist_ok=True)


################################################################################
# Visualize hexagonal camera geometries with hexagonal image data storage
################################################################################

for camgeom_index, camgeom in enumerate(camgeoms):
	print(f'> ({camgeom_index + 1:{len(str(camgeoms_len))}}/{camgeoms_len}) camgeom={camgeom}')

	output_dir_camgeom = os.path.join(output_dir, camgeom)

	geom  = CameraGeometry.from_name(camgeom)
	model = toymodel.Gaussian(x = 0.2 * u.m, y = 0.0 * u.m, width = 0.05 * u.m, length = 0.15 * u.m, psi = '35d')

	image, sig, bg = model.generate_image(geom, intensity=1500, nsb_level_pe=5)


	pix_x = geom.pix_x / u.m
	pix_y = geom.pix_y / u.m

	pix_x -= pix_x.min()
	pix_y -= pix_y.min()

	pix_y = pix_y.max() - pix_y

	image_normalized = image - image.min()
	image_normalized = (255 * (image_normalized / image_normalized.max())).astype(np.uint8)


	pix_x_unique_sorted = np.sort(np.unique(pix_x))
	pix_y_unique_sorted = np.sort(np.unique(pix_y))

	x_diff = pix_x_unique_sorted[2] - pix_x_unique_sorted[0]
	y_diff = pix_y_unique_sorted[1] - pix_y_unique_sorted[0]


	pixels_per_row = []

	for y in pix_y_unique_sorted:
		pixels_per_row.append(len(pix_x[np.where(pix_y == y)[0]]))

	rows = len(pixels_per_row)
	cols = max(pixels_per_row)

	print(f'\t> rows x cols = {rows} x {cols}')


	rows_is_even = 1 - rows % 2
	cols_is_even = 1 - cols % 2

	center            = int(rows / 2)
	center_is_odd_row = center % 2


	try:
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

		print(f'\t> image_out =\n{image_out}')
	except:
		pass


	disp = CameraDisplay(geom, image=image_normalized)
	disp.show()

	plt.savefig(f'{output_dir_camgeom}.png', bbox_inches='tight')
	plt.savefig(f'{output_dir_camgeom}.pdf', bbox_inches='tight')

	try:
		disp = CameraDisplay(geom, image=np.array(image_out_list))
		disp.show()

		imsave(f'{output_dir_camgeom}_hex.png', image_out, cmap='gray')
	except:
		pass

	plt.close()


