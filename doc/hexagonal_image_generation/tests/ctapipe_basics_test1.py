#!/usr/bin/env python3.7


import astropy.units     as u
import matplotlib.pyplot as plt

from ctapipe.image         import toymodel
from ctapipe.instrument    import CameraGeometry
from ctapipe.visualization import CameraDisplay

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

camgeoms_len = len(camgeoms)

for camgeom_index, camgeom in enumerate(camgeoms):
	print(f'> ({camgeom_index + 1:{len(str(camgeoms_len))}}/{camgeoms_len}) camgeom={camgeom}')

	geom  = CameraGeometry.from_name(camgeom)
	model = toymodel.Gaussian(x = 0.2 * u.m, y = 0.0 * u.m, width = 0.05 * u.m, length = 0.15 * u.m, psi = '35d')

	image, sig, bg = model.generate_image(geom, intensity=1500, nsb_level_pe=5)

	disp = CameraDisplay(geom, image=image)
	plt.show(disp)

