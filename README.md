Hexnet - _The_ Hexagonal Image Processing Framework
===================================================

![doc/logo/Hexnet_logo_large.png](doc/logo/Hexnet_logo_large.png "Hexnet logo")

Setup
-----

```
sudo apt-get install libepoxy-dev libgtk-3-dev libxml2-utils
```

### ImageMagick

https://imagemagick.org/script/install-source.php

```
sudo apt-get install libjpeg-dev libpng-dev libtiff-dev
wget https://imagemagick.org/download/ImageMagick.tar.gz
mkdir ImageMagick
tar xvzf ImageMagick.tar.gz -C ImageMagick --strip-components 1
cd ImageMagick
./configure
make
sudo make install
sudo ldconfig /usr/local/lib
cd ..
```


### Hexnet

```
make
```




Usage
-----

```
./Hexnet [options]
```


