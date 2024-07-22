Extended Hexagonal Image Processing Framework HIP in C (CHIP)
=============================================================


---

This is the **Extended Hexagonal Image Processing Framework HIP in C (CHIP)** of this project.

It optimizes and extends the *Hexagonal Image Processing Framework HIP* by *Lee Middleton and Jayanthi Sivaswamy* \[1\] with:

- Implementations for the hexagonal discrete cosine transform (H-DCT) \[2\],
- hexagonal image quantization,
- hexagonal filters (i.e., low-/high-pass, un-/blurring, and Lanczos filters) \[3\], and
- hexagonal scaling, as well as

- Lookup tables (LUT) for a more efficient processing.

```
[1] Middleton, L., & Sivaswamy, J. (2005). Hexagonal image processing: A practical approach. Springer Science & Business Media.

[2] Azam, M., Anjum, M. A., & Javed, M. Y. (2010, February). Discrete cosine transform (DCT) based face recognition in hexagonal images. In 2010 The 2nd International Conference on Computer and Automation Engineering (ICCAE) (Vol. 2, pp. 474-479). IEEE.

[3] James D. Allen (2003). Filter Banks for Images on Hexagonal Grid. Signal Solutions, Unltd. https://fabpedigree.com/james/hexim1.htm
```

---




Installation
------------

### Dependencies

```
sudo apt-get install libepoxy-dev libgtk-3-dev libxml2-utils libjpeg-dev libpng-dev libtiff-dev

wget https://github.com/ImageMagick/ImageMagick/archive/refs/tags/7.1.1-0.tar.gz && \
	tar xzf 7.1.1-0.tar.gz && \
	cd ImageMagick-7.1.1-0 && \
	./configure && \
	make

sudo make install && sudo ldconfig /usr/local/lib && cd ..
```


### CHIP

Standalone

```
make CHIP
```




Usage
-----

```
./CHIP
```

```
Usage: ./CHIP \
           <input image filename> \
           <title of the output> \
           <Hexarray size in order (number of pixels = 7^order) (1-7)> \
           <image transformation (0-3): bilinear / bicubic / Lanczos / B-spline interpolation> \
           <hexagonal-to-square image transformation (h2s) (>0.0): interpolation radius> \
           <hexagonal-to-square image transformation (h2s) (>0): number of pthreads> \
           <square-to-hexagonal image transformation (s2h) (>0.0): scaling factor> \
           <hexagonal-to-square image transformation (h2s) (>0.0): scaling factor> \
           <hexagonal DCT (0|2|5): disabled / with Hexarray order 1 / with Hexarray order 2> \
           <hexagonal DCT mode (0-5): H-DCT / DCT-H (only N = 5) / Log Space 1D-DCT-H | 1D-DCT-H (3)> \
           <hexagonal quantization (1-99): quality factor> \
           <hexagonal fitlers (0-6): blurring / unblurring / low-pass / high-pass filters (4)> \
           <hexagonal scaling (0-4): downscaling based on Hexarray order (HIP) (2) / own up- / own downscaling>
```


Test
----

```
make test
```


License
-------

[MIT License](LICENSE.txt)

