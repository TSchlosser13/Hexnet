Hexnet: _The_ Hexagonal Machine Learning Module
===============================================


![../doc/logo/Hexnet_logo_large.png](../doc/logo/Hexnet_logo_large.png "Hexnet logo")


---

This is **_The_ Hexagonal Machine Learning Module** of this project.

For the base system of this project for hexagonal transformation and visualization see [../](https://github.com/TSchlosser13/Hexnet).

---




Installation
------------

### CPU

```
conda create -n Hexnet_CPU python=3.7 -c conda-forge --file requirements_CPU_conda.txt
conda activate Hexnet_CPU
pip install -r requirements_CPU_pip.txt
```


### GPU

```
conda create -n Hexnet_GPU python=3.7 -c conda-forge --file requirements_GPU_conda.txt
conda activate Hexnet_GPU
pip install -r requirements_GPU_pip.txt
```




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


