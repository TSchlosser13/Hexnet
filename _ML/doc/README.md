Hexnet: Hexagonal Machine Learning Examples
===========================================


![Hexnet logo](../../doc/logos/Hexnet_logo_large.png "Hexnet logo")

[![build](https://travis-ci.com/TSchlosser13/Hexnet.svg?branch=master)](https://travis-ci.com/TSchlosser13/Hexnet)
![os](https://img.shields.io/badge/os-linux%20%7C%20windows-blue)
![python](https://img.shields.io/badge/python-3.7-blue)
[![license](https://img.shields.io/github/license/TSchlosser13/Hexnet)](https://github.com/TSchlosser13/Hexnet/blob/master/_ML/LICENSE.txt)


---

Examples of hexagonal machine learning.

For **_The_ Hexagonal Machine Learning Module** of this project see [../](https://github.com/TSchlosser13/Hexnet/tree/master/_ML).

---




Create a Classification Dataset
-------------------------------

```
python Hexnet.py --model --dataset dataset --create-dataset {'train':0.9,'test':0.1}
```


Visualize the Dataset
---------------------

```
python Hexnet.py --model --dataset dataset --visualize-dataset
```


Multi-Label Classification
--------------------------

```
python Hexnet.py --disable-output --dataset dataset --model CNN_multilabel_test --loss keras_CategoricalCrossentropy
```


Model Comparison Test Script
----------------------------

```
cd tests
python Model_Comparison.py --dataset dataset1 dataset2 --model CNN SCNN
```


