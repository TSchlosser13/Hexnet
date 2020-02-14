Hexnet - _The_ Hexagonal Image Processing Framework
===================================================


![doc/logo/Hexnet_logo_large.png](doc/logo/Hexnet_logo_large.png "Hexnet logo")


---

This is the base system of this project for hexagonal transformation and visualization.

For **_The_ Hexagonal Machine Learning Module** of this project see [_ML/](_ML/).

---




Installation
------------

### Linux

```
sudo apt-get install libepoxy-dev libgtk-3-dev libxml2-utils
```

Install ImageMagick

```
sudo apt-get install libjpeg-dev libpng-dev libtiff-dev
# https://imagemagick.org/script/install-source.php
wget https://imagemagick.org/download/ImageMagick.tar.gz
mkdir ImageMagick
tar xvzf ImageMagick.tar.gz --directory ImageMagick --strip-components 1
cd ImageMagick
./configure
make
sudo make install
sudo ldconfig /usr/local/lib
cd ..
```


### Windows

Install MSYS2: https://www.msys2.org/

```
pacman -S base-devel git mingw-w64-x86_64-gtk3 mingw-w64-x86_64-libepoxy mingw-w64-x86_64-toolchain
```

Install ImageMagick

```
# https://imagemagick.org/script/install-source.php
wget https://imagemagick.org/download/ImageMagick.tar.gz
mkdir ImageMagick
tar xvzf ImageMagick.tar.gz --directory ImageMagick --strip-components 1
cd ImageMagick
./configure
make
make install
cd ..
```


### Hexnet

Standalone

```
make Hexnet
```

Shared object for **_The_ Hexagonal Machine Learning Module** of this project

```
make Hexnet.so
```


Usage
-----

```
./Hexnet --help
```


Test
----

```
make test
```


License
-------

[MIT License](LICENSE.txt)

