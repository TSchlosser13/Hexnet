#!/usr/bin/env python3.7


################################################################################
# Imports
################################################################################

import astropy.units     as u
import matplotlib.pyplot as plt

from ctapipe.image         import toymodel
from ctapipe.instrument    import CameraGeometry
from ctapipe.visualization import CameraDisplay


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


################################################################################
# Initialization
################################################################################

camgeoms_len = len(camgeoms)


################################################################################
# Visualize hexagonal camera geometries and debugging
################################################################################

for camgeom_index, camgeom in enumerate(camgeoms):
	print(f'> ({camgeom_index + 1:{len(str(camgeoms_len))}}/{camgeoms_len}) camgeom={camgeom}')

	geom  = CameraGeometry.from_name(camgeom)
	model = toymodel.Gaussian(x = 0.2 * u.m, y = 0.0 * u.m, width = 0.05 * u.m, length = 0.15 * u.m, psi = '35d')

	image, sig, bg = model.generate_image(geom, intensity=1500, nsb_level_pe=5)

	disp = CameraDisplay(geom, image=image)
	plt.show(disp)


	print(vars(geom).keys())
	print(vars(model).keys())
	print(image)
	print(vars(disp).keys())


