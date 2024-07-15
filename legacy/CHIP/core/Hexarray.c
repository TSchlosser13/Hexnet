/******************************************************************************
 * Hexarray.c: Funktionen auf Hexarrays
 ******************************************************************************
 * v0.1 - 01.04.2016
 *
 * Copyright (c) 2016 Tobias Schlosser (tobias@tobias-schlosser.net)
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 ******************************************************************************/


#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include <math.h>

#include "../misc/defines.h"
#include "../misc/precalculations.h"
#include "../misc/types.h"

#include "Hexint.h"

#include "Hexarray.h"

#include "MagickWand/MagickWand.h"


// Modi: RGB_Hexarray (int) / fRGB_Hexarray (float)

void Hexarray_init(void* hexarray, unsigned int order, bool type) {
	if(!type) {
		((RGB_Hexarray*)hexarray)->size = pow(7, order);

		((RGB_Hexarray*)hexarray)->p = (int**)malloc(((RGB_Hexarray*)hexarray)->size * sizeof(int*));
		for(unsigned int i = 0; i < ((RGB_Hexarray*)hexarray)->size; i++) {
			((RGB_Hexarray*)hexarray)->p[i] = (int*)calloc(3, sizeof(int));
		}
	} else {
		((fRGB_Hexarray*)hexarray)->size = pow(7, order);

		((fRGB_Hexarray*)hexarray)->p = (float**)malloc(((fRGB_Hexarray*)hexarray)->size * sizeof(float*));
		for(unsigned int i = 0; i < ((fRGB_Hexarray*)hexarray)->size; i++) {
			((fRGB_Hexarray*)hexarray)->p[i] = (float*)calloc(3, sizeof(float));
		}
	}
}

void Hexarray_free(void* hexarray, bool type) {
	if(!type) {
		for(unsigned int i = 0; i < ((RGB_Hexarray*)hexarray)->size; i++) {
			free(((RGB_Hexarray*)hexarray)->p[i]);
		}
		free(((RGB_Hexarray*)hexarray)->p);
	} else {
		for(unsigned int i = 0; i < ((fRGB_Hexarray*)hexarray)->size; i++) {
			free(((fRGB_Hexarray*)hexarray)->p[i]);
		}
		free(((fRGB_Hexarray*)hexarray)->p);
	}
}

void Hexarray2file(void* hexarray, char* filename, bool type) {
	FILE* file = fopen(filename, "w");

	fputc('<', file);
	if(!type) {
		for(unsigned int i = 0; i < ((RGB_Hexarray*)hexarray)->size; i++) {
			if(i < ((RGB_Hexarray*)hexarray)->size - 1) {
				fprintf(file, "( %u %u %u ),", ((RGB_Hexarray*)hexarray)->p[i][0], ((RGB_Hexarray*)hexarray)->p[i][1], ((RGB_Hexarray*)hexarray)->p[i][2]);
			} else {
				fprintf(file, "( %u %u %u )>", ((RGB_Hexarray*)hexarray)->p[i][0], ((RGB_Hexarray*)hexarray)->p[i][1], ((RGB_Hexarray*)hexarray)->p[i][2]);
			}
		}
	} else {
		for(unsigned int i = 0; i < ((fRGB_Hexarray*)hexarray)->size; i++) {
			if(i < ((fRGB_Hexarray*)hexarray)->size - 1) {
				fprintf(file, "( %f %f %f ),", ((fRGB_Hexarray*)hexarray)->p[i][0], ((fRGB_Hexarray*)hexarray)->p[i][1], ((fRGB_Hexarray*)hexarray)->p[i][2]);
			} else {
				fprintf(file, "( %f %f %f )>", ((fRGB_Hexarray*)hexarray)->p[i][0], ((fRGB_Hexarray*)hexarray)->p[i][1], ((fRGB_Hexarray*)hexarray)->p[i][2]);
			}
		}
	}

	fclose(file);
}


// Weitere Ausgaben als PNG

// Eindimensional (auf einer Zeile)
void Hexarray2PNG_1D(RGB_Hexarray hexarray, char* filename) {
	MagickWand* mw = NewMagickWand();
	PixelWand*  pw = NewPixelWand();

	MagickNewImage(mw, hexarray.size, 1, pw);

	uint8_t* img = (uint8_t*)malloc(hexarray.size * 3 * sizeof(uint8_t));

	for(unsigned int i = 0; i < hexarray.size; i++) {
		img[3 * i]     = hexarray.p[i][0]; // R
		img[3 * i + 1] = hexarray.p[i][1]; // G
		img[3 * i + 2] = hexarray.p[i][2]; // B
	}

	MagickImportImagePixels(mw, 0, 0, hexarray.size, 1, "RGB", CharPixel, img);
	MagickWriteImage(mw, filename);

	DestroyPixelWand(pw);
	DestroyMagickWand(mw);

	free(img);
}

// Zweidimensional mittels schiefer Achse
void Hexarray2PNG_2D(RGB_Hexarray hexarray, char* filename) {
	const unsigned int width  = pc_smx.x - pc_smn.x + 1;
	const unsigned int height = pc_smx.y - pc_smn.y + 1;

	MagickWand* mw = NewMagickWand();
	PixelWand*  pw = NewPixelWand();

	MagickNewImage(mw, width, height, pw);

	uint8_t* img = (uint8_t*)calloc(width * height * 3, sizeof(uint8_t));

	for(unsigned int i = 0; i < hexarray.size; i++) {
		fPoint2d ps = getSpatial(Hexint_init(i, 0)); // schiefe Achse

		ps.x -= pc_smn.x;
		ps.y  = pc_smx.y - ps.y;

		const unsigned int pos = (unsigned int)(ps.y * width * 3 + 3 * ps.x);

		img[pos]     = hexarray.p[i][0]; // R
		img[pos + 1] = hexarray.p[i][1]; // G
		img[pos + 2] = hexarray.p[i][2]; // B
	}

	MagickImportImagePixels(mw, 0, 0, width, height, "RGB", CharPixel, img);
	MagickWriteImage(mw, filename);

	DestroyPixelWand(pw);
	DestroyMagickWand(mw);

	free(img);
}

// Zweidimensional mittels kartesischem KOS
void Hexarray2PNG_2D_directed(RGB_Hexarray hexarray, char* filename) {
	const unsigned int width  = pc_smx.x - pc_smn.x + 1;
	const unsigned int height = pc_smx.y - pc_smn.y + 1;

	MagickWand* mw = NewMagickWand();
	PixelWand*  pw = NewPixelWand();

	MagickNewImage(mw, width, height, pw);

	uint8_t* img = (uint8_t*)calloc(width * height * 3, sizeof(uint8_t));

	for(unsigned int i = 0; i < hexarray.size; i++) {
		fPoint2d ps = getSpatial(Hexint_init(i, 0)); // schiefe Achse

		// Direktion (Umwandlung kartesisches KOS)
		if(ps.y > 1.0f) {
			ps.x -= roundf((ps.y - 1) / 2);
		} else if(ps.y < 0.0f) {
			ps.x -= roundf(ps.y / 2);
		}

		ps.x -= pc_smn.x;
		ps.y  = pc_smx.y - ps.y;

		const unsigned int pos = (unsigned int)(ps.y * width * 3 + 3 * ps.x);

		img[pos]     = hexarray.p[i][0]; // R
		img[pos + 1] = hexarray.p[i][1]; // G
		img[pos + 2] = hexarray.p[i][2]; // B
	}

	MagickImportImagePixels(mw, 0, 0, width, height, "RGB", CharPixel, img);
	MagickWriteImage(mw, filename);

	DestroyPixelWand(pw);
	DestroyMagickWand(mw);

	free(img);
}


// Skalierung: eigens entwickelte / nach HIP

// Modi: herunter / hoch (times x /7 / *7)
void Hexarray_scale(void* hexarray, bool mode, unsigned int times, bool type) {
	const unsigned int sub_size = pow(7, times); // Skalierungsfaktor

	// Herunterskalierung
	if(!mode) {
		unsigned int rgb_new[3] = { 0, 0, 0 };

		for(unsigned int i = 0; i < ((RGB_Hexarray*)hexarray)->size; i++) {
			rgb_new[0] += ((RGB_Hexarray*)hexarray)->p[i][0];
			rgb_new[1] += ((RGB_Hexarray*)hexarray)->p[i][1];
			rgb_new[2] += ((RGB_Hexarray*)hexarray)->p[i][2];

			// Berechnung via Durchschnittsbildung
			if(!((i + 1) % sub_size)) {
				const unsigned pos = i / sub_size;

				((RGB_Hexarray*)hexarray)->p[pos][0] = rgb_new[0] / sub_size;
				((RGB_Hexarray*)hexarray)->p[pos][1] = rgb_new[1] / sub_size;
				((RGB_Hexarray*)hexarray)->p[pos][2] = rgb_new[2] / sub_size;

				rgb_new[0] = rgb_new[1] = rgb_new[2] = 0;
			}
		}

		((RGB_Hexarray*)hexarray)->size /= sub_size;
	// Hochskalierung
	} else {
		unsigned int order = 1 + times;
		RGB_Hexarray tempx;

		for(unsigned int i = ((RGB_Hexarray*)hexarray)->size; i > 7; i /= 7) {
			order++;
		}
		Hexarray_init(&tempx, order, 0);


		// Berechnung
		for(unsigned int i = 0; i < ((RGB_Hexarray*)hexarray)->size; i++) {
			for(unsigned int ext = 0; ext < sub_size; ext++) {
				const unsigned int pos = i * sub_size + ext;

				tempx.p[pos][0] = ((RGB_Hexarray*)hexarray)->p[i][0];
				tempx.p[pos][1] = ((RGB_Hexarray*)hexarray)->p[i][1];
				tempx.p[pos][2] = ((RGB_Hexarray*)hexarray)->p[i][2];
			}
		}


		// Neue Größe setzen
		Hexarray_free(hexarray, 0);
		Hexarray_init(hexarray, order, 0);

		// Werte rausschreiben
		for(unsigned int i = 0; i < tempx.size; i++) {
			((RGB_Hexarray*)hexarray)->p[i][0] = tempx.p[i][0];
			((RGB_Hexarray*)hexarray)->p[i][1] = tempx.p[i][1];
			((RGB_Hexarray*)hexarray)->p[i][2] = tempx.p[i][2];
		}


		Hexarray_free(&tempx, 0);
	}
}

// Modi: 1 / 7 Versionen (nur Herunterskalierung, mod = Mul.-Modifikator)
void Hexarray_scale_HIP(void* hexarray, bool mode, unsigned int mod) {
	unsigned int order = 1;
	RGB_Hexarray tempx;

	for(unsigned int i = ((RGB_Hexarray*)hexarray)->size; i > 7; i /= 7) {
		order++;
	}
	if(!mode) {
		Hexarray_init(&tempx, --order, 1);
	} else {
		Hexarray_init(&tempx, order, 1);
	}


	// Berechnung via Multiplikation von Hexints
	for(unsigned int i = 0; i < tempx.size; i++) {
		unsigned int new = getInt(mul_Hexint(Hexint_init(i, 0), Hexint_init(mod, 1)));

		if(new > ((RGB_Hexarray*)hexarray)->size - 1) {
			new %= tempx.size;
		}

		tempx.p[i][0] = ((RGB_Hexarray*)hexarray)->p[new][0];
		tempx.p[i][1] = ((RGB_Hexarray*)hexarray)->p[new][1];
		tempx.p[i][2] = ((RGB_Hexarray*)hexarray)->p[new][2];
	}


	if(!mode) {
		// Neue Größe setzen
		Hexarray_free(hexarray, 0);
		Hexarray_init(hexarray, order, 0);
	}

	// Werte rausschreiben
	for(unsigned int i = 0; i < tempx.size; i++) {
		((RGB_Hexarray*)hexarray)->p[i][0] = tempx.p[i][0];
		((RGB_Hexarray*)hexarray)->p[i][1] = tempx.p[i][1];
		((RGB_Hexarray*)hexarray)->p[i][2] = tempx.p[i][2];
	}


	Hexarray_free(&tempx, 0);
}

