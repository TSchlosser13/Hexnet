Hexnet: Hexagonal Image Generation
==================================


![Hexnet logo](../logos/Hexnet_logo_large.png "Hexnet logo")

[![build](https://travis-ci.com/TSchlosser13/Hexnet.svg?branch=master)](https://travis-ci.com/TSchlosser13/Hexnet)
![os](https://img.shields.io/badge/os-linux%20%7C%20windows-blue)
![python](https://img.shields.io/badge/python-3.7-blue)
[![license](https://img.shields.io/github/license/TSchlosser13/Hexnet)](https://github.com/TSchlosser13/Hexnet/blob/master/LICENSE.txt)


---

Scripts and tutorials on hexagonal image generation.

For the documentation of this project see [../](https://github.com/TSchlosser13/Hexnet/tree/master/doc).

---




Generation of Basic Geometric Primitives
----------------------------------------

### Installation

Install SymPy version 1.6.2

```
conda create -n geometric_primitives python=3.7
conda activate geometric_primitives
pip install colorama==0.4.4 joblib==0.17.0 matplotlib sympy==1.6.2 tqdm==4.47.0
```


### Image Generation Script

Current classes for generation

- Class 0: lines
- Class 1: curves
- Class 2: circles
- Class 3: line-based grids
- Class 4: curve-based grids
- Class 5: lines, curves, and circles
- Class 6: miscellaneous

```
python geometric_primitives_image_generation_dataset.py
```

See also geometric_primitives_image_generation.py.




Generation of Cherenkov Telescope Array (CTA) Images Using ctapipe
------------------------------------------------------------------

### Installation

Install ctapipe version 0.8.0

```
mkdir ctapipe
cd ctapipe

CTAPIPE_VER=0.8.0
wget https://raw.githubusercontent.com/cta-observatory/ctapipe/v$CTAPIPE_VER/environment.yml
conda env create -n ctapipe -f environment.yml
conda activate ctapipe
conda install -c cta-observatory ctapipe=$CTAPIPE_VER
```


### Image Generation Script

Current classes for generation

- Class 0: no shower areas
- Class 1: single shower area
- Class 2: multiple shower areas (2 to 9)

```
python ctapipe_image_generation_dataset.py
```

See also ctapipe_image_generation.py.


### Test Scripts

```
cd ctapipe
```

Show all camera geometries

```
python ctapipe_basics_test1.py
```

Image postprocessing

```
python ctapipe_basics_test2.py
```

Generate hexagonal image data

```
python ctapipe_basics_test3.py
```

Inspect camera geometry, model, image, and display properties

```
python ctapipe_basics_test4.py
```

Generate and output hexagonal image data

```
python ctapipe_image_generation_test.py
```

See also notebooks \*.ipynb.


### Find Camera Geometries

```
git clone https://github.com/cta-observatory/ctapipe-extra

cd ctapipe-extra/ctapipe_resources
ls | grep ".camgeom.fits.gz"
```

```
ASTRICam.camgeom.fits.gz
CHEC.camgeom.fits.gz
DigiCam.camgeom.fits.gz
FACT.camgeom.fits.gz
FlashCam.camgeom.fits.gz
hess_camgeom.fits.gz
HESS-I.camgeom.fits.gz
HESS-II.camgeom.fits.gz
LSTCam.camgeom.fits.gz
LSTCam-002.camgeom.fits.gz
LSTCam-003.camgeom.fits.gz
MAGICCam.camgeom.fits.gz
MAGICCamMars.camgeom.fits.gz
NectarCam.camgeom.fits.gz
NectarCam-003.camgeom.fits.gz
SCTCam.camgeom.fits.gz
VERITAS.camgeom.fits.gz
Whipple109.camgeom.fits.gz
Whipple151.camgeom.fits.gz
Whipple331.camgeom.fits.gz
Whipple490.camgeom.fits.gz
```

- ASTRICam
- CHEC
- DigiCam
- FACT
- FlashCam
- HESS-I
- HESS-II
- LSTCam
- LSTCam-002
- LSTCam-003
- MAGICCam
- MAGICCamMars
- NectarCam
- NectarCam-003
- SCTCam
- VERITAS
- Whipple109
- Whipple151
- Whipple331
- Whipple490


### References

- https://cta-observatory.github.io/ctapipe/

- https://cta-observatory.github.io/ctapipe/api/ctapipe.instrument.CameraGeometry.html

- https://cta-observatory.github.io/ctapipe/examples/dilate_image.html
- https://cta-observatory.github.io/ctapipe/examples/InstrumentDescription.html

- https://cta-observatory.github.io/ctapipe/tutorials/lst_analysis_bootcamp_2018.html

