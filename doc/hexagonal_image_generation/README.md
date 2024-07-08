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




Generation of Geometric Primitives
----------------------------------

### Installation

Install SymPy version 1.6.2

```
conda create -n geometric_primitives python=3.7
conda activate geometric_primitives
pip install colorama==0.4.4 joblib==0.17.0 matplotlib sympy==1.6.2 tqdm==4.47.0
```


### Image Generation Scripts

Current classes for generation

- Class 1: lines
- Class 2: curves
- Class 3: circles
- Class 4: line-based grids
- Class 5: curve-based grids
- Class 6: lines, curves, and circles

To generate the dataset:

```
python geometric_primitives_dataset.py
```

See also geometric_primitives.py.


### Test Scripts

```
cd geometric_primitives
```

Visualize test functions with square / hexagonal image data storage:

```
python geometric_primitives_test.py
```

See also notebook geometric_primitives_test.ipynb.




Generation of Astronomical Images
---------------------------------

### Installation

Install ctapipe version 0.8.0

```
conda create -n astronomical_image_generation python=3.7
conda activate astronomical_image_generation
conda install -c cta-observatory ctapipe=0.8.0 matplotlib=3.1.0
```


### Image Generation Scripts

Current classes for generation

- Class 1: no shower areas
- Class 2: single shower area with Gaussian distribution
- Class 3: multiple shower areas with Gaussian distributions
- Class 4: single shower area with skewed Gaussian distribution
- Class 5: multiple shower areas with skewed Gaussian distributions
- Class 6: single shower area with ring Gaussian distribution
- Class 7: multiple shower areas with ring Gaussian distributions

To generate the dataset:

```
python astronomical_image_generation_dataset.py
```

See also astronomical_image_generation.py.


### Test Scripts

```
cd astronomical_image_generation
```

Show all camera geometries:

```
python ctapipe_basics_test1.py
```

Image postprocessing:

```
python ctapipe_basics_test2.py
```

Generate hexagonal image data:

```
python ctapipe_basics_test3.py
```

Inspect camera geometry, model, image, and display properties:

```
python ctapipe_basics_test4.py
```

Generate and output hexagonal image data:

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
(...)
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

- https://www.astropy.org/
- https://pypi.org/project/eventio/
- https://www.mpi-hd.mpg.de/hfm/~bernlohr/sim_telarray/

- https://cta-observatory.github.io/ctapipe/
- https://cta-observatory.github.io/ctapipe/api/ctapipe.instrument.CameraGeometry.html
- https://cta-observatory.github.io/ctapipe/examples/dilate_image.html
- https://cta-observatory.github.io/ctapipe/examples/InstrumentDescription.html
- https://cta-observatory.github.io/ctapipe/tutorials/lst_analysis_bootcamp_2018.html


