Hexnet with Windows
===================

Setup
-----

Install MSYS2: https://www.msys2.org/

```
pacman -S base-devel git mingw-w64-x86_64-gtk3 mingw-w64-x86_64-libepoxy mingw-w64-x86_64-toolchain
```

Run `C:/msys64/mingw64.exe`

### ImageMagick

https://imagemagick.org/script/install-source.php

```
wget https://imagemagick.org/download/ImageMagick.tar.gz
mkdir ImageMagick
tar xvzf ImageMagick.tar.gz -C ImageMagick --strip-components 1
cd ImageMagick
./configure
make
make install
cd ..
```


### Hexnet

Continue with [../README.md](../README.md)


