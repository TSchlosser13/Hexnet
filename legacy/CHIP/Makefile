###############################################################################
 # Makefile: Makefile for the CHIP Framework
 ##############################################################################
 # v0.1 - 01.04.2016
 #
 # Copyright (c) 2016 Tobias Schlosser (tobias@tobias-schlosser.net)
 #
 # Permission is hereby granted, free of charge, to any person obtaining a
 # copy of this software and associated documentation files (the "Software"),
 # to deal in the Software without restriction, including without limitation
 # the rights to use, copy, modify, merge, publish, distribute, sublicense,
 # and/or sell copies of the Software, and to permit persons to whom the
 # Software is furnished to do so, subject to the following conditions:
 #
 # The above copyright notice and this permission notice shall be included in
 # all copies or substantial portions of the Software.
 #
 # THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 # IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 # FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 # AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 # LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 # FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 # DEALINGS IN THE SOFTWARE.
 ###############################################################################


CC        = gcc
CFLAGS    = -march=native -O3 -pipe -std=c99 -Wall
CFLAGS   += -fcommon -Wno-sizeof-pointer-div
includes  = core/Hexarray.c core/Hexint.c core/Hexsamp.c
includes += DCT/HDCT.c filters/filters.c quantization/quantization.c
includes += misc/Array.c misc/precalculations.c
LDFLS     = -lm -lpthread `pkg-config MagickWand --cflags --libs`


.PHONY: clean test

CHIP: CHIP.c
	$(CC) $(CFLAGS) CHIP.c -o CHIP $(includes) $(LDFLS)

clean:
	find . -maxdepth 1 ! -name "CHIP.c" ! -name "LICENSE.txt" ! -name "Makefile" ! -name "README.md" -type f -delete

test:
	./CHIP ../../tests/testset/USC/4.2.03.tiff Baboon 5 1 1.0 1 1.0 3.0 5 3 90 0 0

