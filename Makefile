###############################################################################
 # Makefile: Makefile for The Hexagonal Image Processing Framework Hexnet
 ##############################################################################
 # v0.1 - 01.09.2018
 #
 # Copyright (c) 2018 Tobias Schlosser (tobias@tobias-schlosser.net)
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


################################################################################
# Variables
################################################################################

CC       = gcc
CFLAGS   = -march=native -Ofast -pipe -std=c99 -Wall
CFLAGS  += `pkg-config --cflags epoxy gtk+-3.0 MagickWand`
LDFLAGS  = `pkg-config --libs   epoxy gtk+-3.0 MagickWand` -lm

src = \
	Hexnet.c \
	$(wildcard */*.c)

built_src = gui/gui.gresource.c

obj = $(built_src:.c=.o) $(src:.c=.o)


################################################################################
# Targets
################################################################################

.PHONY: clean install uninstall test

Hexnet: $(obj)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

Hexnet.so: CFLAGS  += -fpic
Hexnet.so: LDFLAGS += -shared
Hexnet.so: $(obj)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

gui/gui.gresource.c: gui/gui.gresource.xml
	glib-compile-resources --target=$@ --generate-source $<

clean:
	rm -fv $(built_src) $(obj) Hexnet Hexnet.so *.png

install:
	cp -v Hexnet    /usr/bin
	cp -v Hexnet.so /usr/lib

uninstall:
	rm -fv /usr/bin/Hexnet
	rm -fv /usr/lib/Hexnet.so

test:
	./Hexnet -i tests/testset/USC/4.1.01.tiff -o 4.1.01_out.png --s2h-rad 1.0 --h2s-len 1.0 -d -v


