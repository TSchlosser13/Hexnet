# Makefile


CC       = gcc
CFLAGS   = -march=native -Ofast -pipe -std=c99 -Wall
CFLAGS  += `pkg-config --cflags epoxy gtk+-3.0 MagickWand`
LDFLAGS  = `pkg-config --libs   epoxy gtk+-3.0 MagickWand` -lm

src = \
	Hexnet.c \
	$(wildcard */*.c)

built_src = gui/gui.gresource.c

obj = $(built_src:.c=.o) $(src:.c=.o)


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


