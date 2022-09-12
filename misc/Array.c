/******************************************************************************
 * Array.c: Operations on Square Arrays
 ******************************************************************************
 * v0.1 - 01.09.2018
 *
 * Copyright (c) 2018 Tobias Schlosser (tobias@tobias-schlosser.net)
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


#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "MagickWand/MagickWand.h"

#include "Array.h"
#include "console.h"
#include "types.h"


void Array_init(Array* array, u32 width, u32 height, u32 depth, float len) {
	array->width     = width;
	array->height    = height;
	array->depth     = depth;
	array->pixels    = array->width * array->height;
	array->size      = array->depth * array->pixels;
	array->len       = len;
	array->width_sq  = array->width  * array->len;
	array->height_sq = array->height * array->len;

	array->p = (u8*)malloc(array->size * sizeof(u8));
}

void Array_init_from_Hexarray(Array* array, Hexarray hexarray, float len) {
	u32 width;
	u32 height;

	if(hexarray.height_hex - hexarray.rad_o > len) {
		width  = ceilf((hexarray.width_hex  - hexarray.dia_i) / len);
		height = ceilf((hexarray.height_hex - hexarray.rad_o) / len);
	} else {
		width  = ceilf(hexarray.width_hex / len);
		height = 1;
	}

	Array_init(array, width, height, hexarray.depth, len);
}

void Array_free(Array* array) {
	array->width     = 0;
	array->height    = 0;
	array->depth     = 0;
	array->pixels    = 0;
	array->size      = 0;
	array->len       = 0.0f;
	array->width_sq  = 0.0f;
	array->height_sq = 0.0f;

	free(array->p);
}


void Array_print_info(Array array, char* title) {
	const u32 p  = 6;
	const u32 wu = log10f(array.size) + 1;
	const u32 wf = wu + 1 + p;

	printf(
		SGM_FC_GREEN
			"[Array_print_info: \"%s\"]\n"
		SGM_TA_RESET

		"width     = %*u\n"
		"height    = %*u\n"
		"depth     = %*u\n"
		"pixels    = %*u\n"
		"size      = %*u\n"
		"len       = %*.*f\n"
		"width_sq  = %*.*f\n"
		"height_sq = %*.*f\n",
		 title,
		 wu,    array.width,
		 wu,    array.height,
		 wu,    array.depth,
		 wu,    array.pixels,
		 wu,    array.size,
		 wf, p, array.len,
		 wf, p, array.width_sq,
		 wf, p, array.height_sq);
}


void file_to_Array(char* filename, Array* array) {
	MagickWand* mw = NewMagickWand();

	MagickReadImage(mw, filename);
	const size_t mw_width  = MagickGetImageWidth(mw);
	const size_t mw_height = MagickGetImageHeight(mw);
	Array_init(array, mw_width, mw_height, 3, 1.0f);
	MagickExportImagePixels(mw, 0, 0, mw_width, mw_height, "RGB", CharPixel, array->p);

	DestroyMagickWand(mw);
}

void Array_to_file(Array array, char* filename) {
	MagickWand* mw = NewMagickWand();
	PixelWand*  pw = NewPixelWand();

	MagickNewImage(mw, array.width, array.height, pw);
	MagickImportImagePixels(mw, 0, 0, array.width, array.height, "RGB", CharPixel, array.p);
	MagickWriteImage(mw, filename);

	DestroyPixelWand(pw);
	DestroyMagickWand(mw);
}

