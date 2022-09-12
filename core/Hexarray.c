/******************************************************************************
 * Hexarray.c: Operations on Hexagonal Arrays
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


/******************************************************************************
 * Includes
 ******************************************************************************/

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "MagickWand/MagickWand.h"

#include "Hexarray.h"

#include "../misc/console.h"
#include "../misc/defines.h"
#include "../misc/types.h"


/******************************************************************************
 * Initialize Hexarray
 ******************************************************************************/

void Hexarray_init(Hexarray* hexarray, u32 width, u32 height, u32 depth, float rad_o) {
	hexarray->width  = width;
	hexarray->height = height;
	hexarray->depth  = depth;
	hexarray->pixels = hexarray->width * hexarray->height;
	hexarray->size   = hexarray->depth * hexarray->pixels;
	hexarray->rad_o  = rad_o;
	hexarray->rad_i  = M_SQRT3_2 * hexarray->rad_o;
	hexarray->dia_o  = 2 * hexarray->rad_o;
	hexarray->dia_i  = 2 * hexarray->rad_i;
	hexarray->dist_w = hexarray->dia_i;
	hexarray->dist_h = 1.5f * hexarray->rad_o;

	if(hexarray->height > 1) {
		hexarray->width_hex  = hexarray->width * hexarray->dia_i + hexarray->rad_i;
		hexarray->height_hex = hexarray->dia_o + (hexarray->height - 1) * hexarray->dist_h;
	} else {
		hexarray->width_hex  = hexarray->width * hexarray->dia_i;
		hexarray->height_hex = hexarray->dia_o;
	}


	hexarray->p = (u8*)malloc(hexarray->size * sizeof(u8));
}


/******************************************************************************
 * Initialize Hexarray from Array
 ******************************************************************************/

void Hexarray_init_from_Array(Hexarray* hexarray, Array array, float rad_o) {
	u32 width;
	u32 height;

	if(array.height > rad_o) {
		width  = ceilf(array.width  / (M_SQRT3 * rad_o));
		height = ceilf(array.height / (1.5f * rad_o));
	} else {
		width  = ceilf(array.width / (M_SQRT3 * rad_o));
		height = 1;
	}

	Hexarray_init(hexarray, width, height, array.depth, rad_o);
}


/******************************************************************************
 * Initialize Hexarray from Hexarray
 ******************************************************************************/

void Hexarray_init_from_Hexarray(Hexarray* h1, Hexarray h2, float rad_o) {
	u32 width;
	u32 height;

	if(h2.height_hex - h2.rad_o > rad_o) {
		width  = ceilf((h2.width_hex  - h2.dia_i) / (M_SQRT3 * rad_o));
		height = ceilf((h2.height_hex - h2.rad_o) / (1.5f * rad_o));
	} else {
		width  = ceilf(h2.width_hex / (M_SQRT3 * rad_o));
		height = 1;
	}

	Hexarray_init(h1, width, height, h2.depth, rad_o);
}


/******************************************************************************
 * Free Hexarray
 ******************************************************************************/

void Hexarray_free(Hexarray* hexarray) {
	hexarray->width      = 0;
	hexarray->height     = 0;
	hexarray->depth      = 0;
	hexarray->pixels     = 0;
	hexarray->size       = 0;
	hexarray->rad_o      = 0.0f;
	hexarray->rad_i      = 0.0f;
	hexarray->dia_o      = 0.0f;
	hexarray->dia_i      = 0.0f;
	hexarray->dist_w     = 0.0f;
	hexarray->dist_h     = 0.0f;
	hexarray->width_hex  = 0.0f;
	hexarray->height_hex = 0.0f;

	free(hexarray->p);
}


/******************************************************************************
 * Hexarray debugging
 ******************************************************************************/

void Hexarray_print_info(Hexarray hexarray, char* title) {
	const u32 p  = 6;
	const u32 wu = log10f(hexarray.size) + 1;
	const u32 wf = wu + 1 + p;

	printf(
		SGM_FC_GREEN
			"[Hexarray_print_info: \"%s\"]\n"
		SGM_TA_RESET

		"width      = %*u\n"
		"height     = %*u\n"
		"depth      = %*u\n"
		"pixels     = %*u\n"
		"size       = %*u\n"
		"rad_o      = %*.*f\n"
		"rad_i      = %*.*f\n"
		"dia_o      = %*.*f\n"
		"dia_i      = %*.*f\n"
		"dist_w     = %*.*f\n"
		"dist_h     = %*.*f\n"
		"width_hex  = %*.*f\n"
		"height_hex = %*.*f\n",
		 title,
		 wu,    hexarray.width,
		 wu,    hexarray.height,
		 wu,    hexarray.depth,
		 wu,    hexarray.pixels,
		 wu,    hexarray.size,
		 wf, p, hexarray.rad_o,
		 wf, p, hexarray.rad_i,
		 wf, p, hexarray.dia_o,
		 wf, p, hexarray.dia_i,
		 wf, p, hexarray.dist_w,
		 wf, p, hexarray.dist_h,
		 wf, p, hexarray.width_hex,
		 wf, p, hexarray.height_hex);
}


/******************************************************************************
 * Load Hexarray from file
 ******************************************************************************/

void file_to_Hexarray(char* filename, Hexarray* hexarray) {
	MagickWand* mw = NewMagickWand();

	MagickReadImage(mw, filename);
	const size_t mw_width  = MagickGetImageWidth(mw);
	const size_t mw_height = MagickGetImageHeight(mw);
	Hexarray_init(hexarray, mw_width, mw_height, 3, 1.0f);
	MagickExportImagePixels(mw, 0, 0, mw_width, mw_height, "RGB", CharPixel, hexarray->p);

	DestroyMagickWand(mw);
}


/******************************************************************************
 * Save Hexarray to file
 ******************************************************************************/

void Hexarray_to_file(Hexarray hexarray, char* filename) {
	MagickWand* mw = NewMagickWand();
	PixelWand*  pw = NewPixelWand();

	MagickNewImage(mw, hexarray.width, hexarray.height, pw);
	MagickImportImagePixels(mw, 0, 0, hexarray.width, hexarray.height, "RGB", CharPixel, hexarray.p);
	MagickWriteImage(mw, filename);

	DestroyPixelWand(pw);
	DestroyMagickWand(mw);
}

