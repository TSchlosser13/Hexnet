#!/usr/bin/env python3.7


import astropy.units     as u
import matplotlib.pyplot as plt
import numpy             as np

from ctapipe.image         import toymodel
from ctapipe.instrument    import CameraGeometry
from ctapipe.visualization import CameraDisplay
from matplotlib.pyplot     import imsave


camgeoms = (
	'HESS-I',
	'HESS-II',
	'VERITAS',
	'Whipple109',
	'Whipple151'
)

camgeoms_len = len(camgeoms)


for camgeom_index, camgeom in enumerate(camgeoms):
	print(f'> ({camgeom_index + 1:{len(str(camgeoms_len))}}/{camgeoms_len}) camgeom={camgeom}')

	geom  = CameraGeometry.from_name(camgeom)
	model = toymodel.Gaussian(x = 0.2 * u.m, y = 0.0 * u.m, width = 0.05 * u.m, length = 0.15 * u.m, psi = '35d')

	image, sig, bg = model.generate_image(geom, intensity=1500, nsb_level_pe=5)


	pix_x = geom.pix_x / u.m
	pix_y = geom.pix_y / u.m

	pix_x -= pix_x.min()
	pix_y -= pix_y.min()

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


	cols_is_odd = cols % 2

	center            = int(rows / 2)
	center_is_odd_row = center % 2


	image_out      = np.zeros(shape = (rows, cols))
	image_out_list = []

	for y, x, pixel_value in zip(pix_y, pix_x, image_normalized):
		y_index       = int(y / y_diff + 0.1)
		y_is_even_row = 1 - y_index % 2

		if y_is_even_row and cols_is_odd:
			x_index = int(x / x_diff + 0.1 + center_is_odd_row)
		else:
			x_index = int(x / x_diff + 0.1)

		image_out[y_index][x_index] = pixel_value
		image_out_list.append(pixel_value)

	print(f'\t> image_out =\n{image_out}')


	disp = CameraDisplay(geom, image=image_normalized)
	plt.show(disp)

	disp = CameraDisplay(geom, image=np.array(image_out_list))

	plt.savefig(f'{camgeom}.png')
	plt.savefig(f'{camgeom}.pdf')

	plt.show(disp)

	plt.close()

	imsave(f'{camgeom}_hex.png', image_out, cmap='gray')

