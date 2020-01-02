# Makefile


CC       = gcc
CFLAGS   = -march=native -Ofast -pipe -std=c99 -Wall
CFLAGS  += `pkg-config --cflags epoxy gtk+-3.0 MagickWand`
LDFLAGS  = `pkg-config --libs   epoxy gtk+-3.0 MagickWand` -lm

src = \
	Hexnet.c          \
	$(wildcard */*.c)

built_src = gui/gui.gresource.c

obj = $(built_src:.c=.o) $(src:.c=.o)


.PHONY: clean test

Hexnet: $(obj)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

gui/gui.gresource.c: gui/gui.gresource.xml
	glib-compile-resources --target=$@ --generate-source $<

clean:
	rm -f $(built_src) $(obj) Hexnet

test:
	./Hexnet -i tests/testset/USC/4.1.01.tiff -o 4.1.01_out.jpg --s2h-rad 1.0 --h2s-len 1.0 -d -v

