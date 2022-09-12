Hexnet: _The_ Hexagonal Machine Learning Module
===============================================


![Hexnet logo](../doc/logos/Hexnet_logo_large.png "Hexnet logo")

[![build](https://travis-ci.com/TSchlosser13/Hexnet.svg?branch=master)](https://travis-ci.com/TSchlosser13/Hexnet)
![os](https://img.shields.io/badge/os-linux%20%7C%20windows-blue)
![python](https://img.shields.io/badge/python-3.7-blue)
[![license](https://img.shields.io/github/license/TSchlosser13/Hexnet)](https://github.com/TSchlosser13/Hexnet/blob/master/_ML/LICENSE.txt)


---

This is **_The_ Hexagonal Machine Learning Module** of this project.

For the base system of this project for hexagonal transformation and visualization see [../](https://github.com/TSchlosser13/Hexnet).

---


![Hexnet screenshot](doc/Hexnet_screenshot.png "Hexnet screenshot")




Installation
------------

### GPU

```
conda create -n Hexnet_GPU python=3.7 -c conda-forge --file requirements_GPU_conda.txt
conda activate Hexnet_GPU
pip install -r requirements_GPU_pip.txt
```


"requirements_GPU_conda.txt" and "requirements_GPU_pip.txt" were generated via

```
conda list --export > requirements_GPU_conda.txt
```

and

```
pip list --format=freeze > requirements_GPU_pip.txt
```

with

```
cudatoolkit=10.1.243
cudnn=7.6.5
imgaug=0.3.0
```

and

```
natsort==7.0.1
pandas==1.0.5
protobuf==3.20.1
scikit-learn==0.23.1
seaborn==0.10.1
tensorflow-gpu==2.1.0
tqdm==4.47.0
```

(12/09/2022)


### CPU

```
conda create -n Hexnet_CPU python=3.7 -c conda-forge --file requirements_CPU_conda.txt
conda activate Hexnet_CPU
pip install -r requirements_CPU_pip.txt
```


"requirements_CPU_conda.txt" and "requirements_CPU_pip.txt" were generated via

```
conda list --export > requirements_CPU_conda.txt
```

and

```
pip list --format=freeze > requirements_CPU_pip.txt
```

with

```
imgaug=0.3.0
```

and

```
natsort==7.0.1
pandas==1.0.5
protobuf==3.20.1
scikit-learn==0.23.1
seaborn==0.10.1
tensorflow==2.1.0
tqdm==4.47.0
```

(12/09/2022)




Getting started: 30 seconds to Hexnet
-------------------------------------

Implement your own square and hexagonal lattice format based models in models/models.py ...

```
import tensorflow as tf

from tensorflow.keras        import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPool2D

from layers.layers import HConv2D, HMaxPool2D, SConv2D, SMaxPool2D


def model_SCNN_test(input_shape, classes, kernel_size, pool_size):
	model = Sequential()

	model.add(SConv2D(filters=32, kernel_size=kernel_size, padding='SAME', activation=tf.nn.relu, input_shape=input_shape))
	model.add(SMaxPool2D(pool_size=pool_size, padding='SAME'))
	model.add(Dropout(rate=0.25))
	model.add(SConv2D(filters=64, kernel_size=kernel_size, padding='SAME', activation=tf.nn.relu))
	model.add(SMaxPool2D(pool_size=pool_size, padding='SAME'))
	model.add(Dropout(rate=0.25))
	model.add(Flatten())
	model.add(Dense(units=128, activation=tf.nn.relu))
	model.add(Dropout(rate=0.5))
	model.add(Dense(units=classes, activation=tf.nn.softmax))

	return model


def model_HCNN_test(input_shape, classes, kernel_size, pool_size):
	model = Sequential()

	model.add(HConv2D(filters=32, kernel_size=kernel_size, padding='SAME', activation=tf.nn.relu, input_shape=input_shape))
	model.add(HMaxPool2D(pool_size=pool_size, padding='SAME'))
	model.add(Dropout(rate=0.25))
	model.add(HConv2D(filters=64, kernel_size=kernel_size, padding='SAME', activation=tf.nn.relu))
	model.add(HMaxPool2D(pool_size=pool_size, padding='SAME'))
	model.add(Dropout(rate=0.25))
	model.add(Flatten())
	model.add(Dense(units=128, activation=tf.nn.relu))
	model.add(Dropout(rate=0.5))
	model.add(Dense(units=classes, activation=tf.nn.softmax))

	return model
```


... or extend already implemented models ...

```
from models.resnets import model_SResNet_v2, model_HResNet_v2

def model_SResNet_v2_test(input_shape, classes, n=2):
	return model_SResNet_v2(input_shape, classes, n)

def model_HResNet_v2_test(input_shape, classes, n=2):
	return model_HResNet_v2(input_shape, classes, n)
```


These are then deployed using ...

```
python Hexnet.py --model HResNet_v2
```


For help, see ...

```
python Hexnet.py --help
```




License
-------

[MIT License](LICENSE.txt)


Funding
-------

The European Union and the European Social Fund for Germany partially funded this research.

![ESF logo](../doc/logos/ESF_logo.png "ESF logo")


