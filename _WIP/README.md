Work in Progress
================

Setup
-----

### CPU

```
conda create -n Hexnet_CPU python==3.7 --file requirements_CPU_conda.txt
conda activate Hexnet_CPU
conda remove --force tensorflow
conda install -c conda-forge imgaug
pip install -r requirements_CPU_pip.txt
```


### GPU

```
conda create -n Hexnet_GPU python==3.7 --file requirements_GPU_conda.txt
conda activate Hexnet_GPU
conda remove --force tensorflow-gpu
conda install -c conda-forge imgaug
pip install -r requirements_GPU_pip.txt
```




Usage
-----

```
./Hexnet.py [options]
```

