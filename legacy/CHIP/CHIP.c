/******************************************************************************
 * CHIP.c: Extended Hexagonal Image Processing Framework HIP in C
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


/******************************************************************************
 * Defines
 ******************************************************************************/

#define _POSIX_C_SOURCE 199309L


/******************************************************************************
 * Includes
 ******************************************************************************/

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "MagickWand/MagickWand.h"

#include "core/Hexarray.h"
#include "core/Hexsamp.h"

#include "DCT/HDCT.h"

#include "filters/filters.h"

#include "quantization/quantization.h"

#include "misc/Array.h"
#include "misc/precalculations.h"
#include "misc/types.h"




#define CLOCK_DIFF(i) ( \
	runtimes[i] = end.tv_sec  - begin.tv_sec + \
	             (end.tv_nsec - begin.tv_nsec) / 1000000000.0 )




/******************************************************************************
 * main
 ******************************************************************************/

int main(int argc, char** argv) {

	/**************************************************************************
	 * Variables
	 **************************************************************************/

	char*        ifname;      // input image filename
	char         ofname[256]; // output image filename

	// Parameters
	char*        title;     // title of the output
	unsigned int order;     // Hexarray size in order (number of pixels = 7^order) (1-7)
	unsigned int mode;      // image transformation (0-3): bilinear / bicubic / Lanczos / B-spline interpolation
	float        radius;    // hexagonal-to-square image transformation (h2s) (>0.0): interpolation radius
	unsigned int threads;   // hexagonal-to-square image transformation (h2s) (>0): number of pthreads
	float        s2h_scale; // square-to-hexagonal image transformation (s2h) (>0.0): scaling factor
	float        h2s_scale; // hexagonal-to-square image transformation (h2s) (>0.0): scaling factor
	unsigned int DCT_N;     // hexagonal DCT (0|2|5): disabled / with Hexarray order 1 / with Hexarray order 2
	unsigned int DCT_mode;  // hexagonal DCT mode (0-5): H-DCT / DCT-H (only N = 5) / Log Space 1D-DCT-H | 1D-DCT-H (3)
	unsigned int quant_qf;  // hexagonal quantization (1-99): quality factor
	unsigned int filter;    // hexagonal fitlers (0-6): blurring / unblurring / low-pass / high-pass filters (4)
	unsigned int scaling;   // hexagonal scaling (0-4): downscaling based on Hexarray order (HIP) (2) / own up- / own downscaling


	/**************************************************************************
	 * Parameters
	 **************************************************************************/

	setbuf(stdout, NULL);
	setbuf(stderr, NULL);

	if(argc < 14) {
		puts(
			"Usage: ./CHIP \\\n"
			"           <input image filename> \\\n"
			"           <title of the output> \\\n"
			"           <Hexarray size in order (number of pixels = 7^order) (1-7)> \\\n"
			"           <image transformation (0-3): bilinear / bicubic / Lanczos / B-spline interpolation> \\\n"
			"           <hexagonal-to-square image transformation (h2s) (>0.0): interpolation radius> \\\n"
			"           <hexagonal-to-square image transformation (h2s) (>0): number of pthreads> \\\n"
			"           <square-to-hexagonal image transformation (s2h) (>0.0): scaling factor> \\\n"
			"           <hexagonal-to-square image transformation (h2s) (>0.0): scaling factor> \\\n"
			"           <hexagonal DCT (0|2|5): disabled / with Hexarray order 1 / with Hexarray order 2> \\\n"
			"           <hexagonal DCT mode (0-5): H-DCT / DCT-H (only N = 5) / Log Space 1D-DCT-H | 1D-DCT-H (3)> \\\n"
			"           <hexagonal quantization (1-99): quality factor> \\\n"
			"           <hexagonal fitlers (0-6): blurring / unblurring / low-pass / high-pass filters (4)> \\\n"
			"           <hexagonal scaling (0-4): downscaling based on Hexarray order (HIP) (2) / own up- / own downscaling>\n");

		ifname    = "../../tests/testset/USC/4.2.03.tiff";
		title     = "Baboon";
		order     =  5;
		mode      =  1;
		radius    =  1.0f;
		threads   =  1;
		s2h_scale =  1.0f;
		h2s_scale =  3.0f;
		DCT_N     =  5;
		DCT_mode  =  3;
		quant_qf  = 90;
		filter    =  0;
		scaling   =  0;
	} else {
		ifname    =                    argv[1];
		title     =                    argv[2];
		order     = (unsigned int)atoi(argv[3]);
		mode      = (unsigned int)atoi(argv[4]);
		radius    =        (float)atof(argv[5]);
		threads   = (unsigned int)atoi(argv[6]);
		s2h_scale =        (float)atof(argv[7]);
		h2s_scale =        (float)atof(argv[8]);
		DCT_N     = (unsigned int)atoi(argv[9]);
		DCT_mode  = (unsigned int)atoi(argv[10]);
		quant_qf  = (unsigned int)atoi(argv[11]);
		filter    = (unsigned int)atoi(argv[12]);
		scaling   = (unsigned int)atoi(argv[13]);
	}




	/**************************************************************************
	 * Initialization
	 **************************************************************************/


	MagickWandGenesis();


	MagickWand* mw = NewMagickWand();

	MagickReadImage(mw, ifname);
	const size_t width  = MagickGetImageWidth(mw);
	const size_t height = MagickGetImageHeight(mw);
	uint8_t*     img    = (uint8_t*)malloc(width * height * 3 * sizeof(uint8_t));
	MagickExportImagePixels(mw, 0, 0, width, height, "RGB", CharPixel, img);

	DestroyMagickWand(mw);

	RGB_Array    array;
	RGB_Hexarray hexarray;

	double runtimes[7] = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
	struct timespec begin, end;


	printf(
		"\n\"%s\" (%zu x %zu): \"%s\"\n"
		"    - order = %u, mode = %u, radius = %.2f, threads = %u\n"
		"    - s2h_scale = %.2f, h2s_scale = %.2f\n"
		"    - DCT_N = %u, DCT_mode = %u, quant_qf = %u\n"
		"    - filter = %u, scaling = %u\n",
		 ifname, width, height, title,
		 order, mode, radius, threads,
		 s2h_scale, h2s_scale,
		 DCT_N, DCT_mode, quant_qf,
		 filter, scaling);


	/**************************************************************************
	 * Load data from file
	 **************************************************************************/

	pArray2d_init(&array, width, height);

	for(unsigned int h = 0; h < height; h++) {
		for(unsigned int w = 0; w < width; w++) {
			const unsigned int p = h * width * 3 + 3 * w;

			array.p[w][h][0] = (unsigned char)img[p];     // R
			array.p[w][h][1] = (unsigned char)img[p + 1]; // G
			array.p[w][h][2] = (unsigned char)img[p + 2]; // B
		}
	}

	free(img);


	// Initialize precalculations

	clock_gettime(CLOCK_MONOTONIC, &begin);

	if(scaling < 3) {
		precalcs_init(order,     s2h_scale, radius);
	} else {
		precalcs_init(order + 1, s2h_scale, radius);
	}

	clock_gettime(CLOCK_MONOTONIC, &end);
	CLOCK_DIFF(0);




	// Square -> hex

	clock_gettime(CLOCK_MONOTONIC, &begin);

	hipsampleColour(array, &hexarray, order, h2s_scale, mode);

	clock_gettime(CLOCK_MONOTONIC, &end);
	CLOCK_DIFF(1);

	pArray2d_free(&array);
	strcpy(ofname, title);
	strcat(ofname, "_s2h.txt");
	Hexarray2file(&hexarray, ofname, 0);
	strcpy(ofname, title);
	strcat(ofname, "_s2h_1D.png");
	Hexarray2PNG_1D(hexarray, ofname);
	strcpy(ofname, title);
	strcat(ofname, "_s2h_2D_skewed.png");
	Hexarray2PNG_2D(hexarray, ofname);
	strcpy(ofname, title);
	strcat(ofname, "_s2h_2D_Cartesian.png");
	Hexarray2PNG_2D_directed(hexarray, ofname);




	/*
	 * H-DCT
	 */

	if(DCT_N == 5) {
		float**       psiCos_table;
		RGB_Array      rgb_HDCTArray;
		fRGB_Hexarray frgb_HDCTHexarray;


		if(!DCT_mode) {
			HDCT_init(  &psiCos_table, 5 );
		} else if(DCT_mode == 1) {
			DCTH_init(  &psiCos_table, 5 );
		} else {
			DCTH2_init( &psiCos_table, 5 );
		}


		clock_gettime(CLOCK_MONOTONIC, &begin);

		if(!DCT_mode) {
			HDCT_N5(hexarray, &rgb_HDCTArray, psiCos_table);
		} else if(DCT_mode == 1) {
			DCTH(hexarray, &rgb_HDCTArray, psiCos_table, 5);
		} else if(DCT_mode == 2) {
			DCTH2(hexarray, &frgb_HDCTHexarray, psiCos_table, 1, 5);
		} else {
			DCTH2(hexarray, &frgb_HDCTHexarray, psiCos_table, 0, 5);
		}

		clock_gettime(CLOCK_MONOTONIC, &end);
		CLOCK_DIFF(2);

		if(DCT_mode > 1) {
			strcpy(ofname, title);
			strcat(ofname, "_s2h_H-DCT.txt");
			Hexarray2file(&frgb_HDCTHexarray, ofname, 1);
		}


		/*
		 * Quantization
		 */

		if(quant_qf > 0 && quant_qf < 100) {
			clock_gettime(CLOCK_MONOTONIC, &begin);

			if(!DCT_mode) {
				Quant_custom(&rgb_HDCTArray, 0, 5, quant_qf);
			} else if(DCT_mode == 1) {
				Quant_custom(&rgb_HDCTArray, 1, 5, quant_qf);
			} else if(DCT_mode == 2 || DCT_mode == 3) {
				Quant_custom(&frgb_HDCTHexarray, 2, 5, quant_qf);
			} else {
				Quant_custom(&frgb_HDCTHexarray, DCT_mode - 1, 5, quant_qf);
			}

			clock_gettime(CLOCK_MONOTONIC, &end);
			CLOCK_DIFF(3);
		}


		clock_gettime(CLOCK_MONOTONIC, &begin);

		if(!DCT_mode) {
			IHDCT_N5(rgb_HDCTArray, &hexarray, psiCos_table);
		} else if(DCT_mode == 1) {
			IDCTH(rgb_HDCTArray, &hexarray, psiCos_table, 5);
		} else if(DCT_mode == 2) {
			IDCTH2(frgb_HDCTHexarray, &hexarray, psiCos_table, 1, 5);
		} else {
			IDCTH2(frgb_HDCTHexarray, &hexarray, psiCos_table, 0, 5);
		}

		clock_gettime(CLOCK_MONOTONIC, &end);
		CLOCK_DIFF(4);

		strcpy(ofname, title);
		strcat(ofname, "_s2h_H-IDCT.txt");
		Hexarray2file(&hexarray, ofname, 0);


		if(!DCT_mode) {
			HDCT_free(&psiCos_table, 5);
		} else if(DCT_mode == 1) {
			DCTH_free(&psiCos_table, 0, 5);
		} else {
			DCTH_free(&psiCos_table, 1, 5);
		}
	} else if(DCT_N == 2) {
		float**       psiCos_table;
		fRGB_Hexarray frgb_HDCTHexarray;


		if(DCT_mode < 2) {
			HDCT_init(  &psiCos_table, 2 );
		} else {
			DCTH2_init( &psiCos_table, 2 );
		}


		clock_gettime(CLOCK_MONOTONIC, &begin);

		if(DCT_mode < 2) {
			HDCT_N2(hexarray, &frgb_HDCTHexarray, psiCos_table);
		} else if(DCT_mode == 2) {
			DCTH2(hexarray, &frgb_HDCTHexarray, psiCos_table, 1, 2);
		} else {
			DCTH2(hexarray, &frgb_HDCTHexarray, psiCos_table, 0, 2);
		}

		clock_gettime(CLOCK_MONOTONIC, &end);
		CLOCK_DIFF(2);

		strcpy(ofname, title);
		strcat(ofname, "_s2h_H-DCT.txt");
		Hexarray2file(&frgb_HDCTHexarray, ofname, 1);


		/*
		 * Quantization
		 */

		if(quant_qf > 0 && quant_qf < 100) {
			clock_gettime(CLOCK_MONOTONIC, &begin);

			if(DCT_mode < 4) {
				Quant_custom(&frgb_HDCTHexarray, 2, 2, quant_qf);
			} else {
				Quant_custom(&frgb_HDCTHexarray, DCT_mode - 1, 2, quant_qf);
			}

			clock_gettime(CLOCK_MONOTONIC, &end);
			CLOCK_DIFF(3);
		}


		clock_gettime(CLOCK_MONOTONIC, &begin);

		if(DCT_mode < 2) {
			IHDCT_N2(frgb_HDCTHexarray, &hexarray, psiCos_table);
		} else if(DCT_mode == 2) {
			IDCTH2(frgb_HDCTHexarray, &hexarray, psiCos_table, 1, 2);
		} else {
			IDCTH2(frgb_HDCTHexarray, &hexarray, psiCos_table, 0, 2);
		}

		clock_gettime(CLOCK_MONOTONIC, &end);
		CLOCK_DIFF(4);

		strcpy(ofname, title);
		strcat(ofname, "_s2h_H-IDCT.txt");
		Hexarray2file(&hexarray, ofname, 0);


		if(DCT_mode < 2) {
			HDCT_free(&psiCos_table, 2);
		} else {
			DCTH_free(&psiCos_table, 1, 2);
		}
	}




	/*
	 * Hexagonal filters
	 */

	if(filter == 1 || filter == 2) {
		filter_unblurring( &hexarray, filter - 1 );
	} else if(filter) {
		filter_4x16(       &hexarray, filter - 3 );
	}


	/*
	 * Hexagonal scaling
	 */

	if(scaling == 1) {
		Hexarray_scale_HIP(&hexarray, 0, 10);
	} else if(scaling == 2) {
		Hexarray_scale_HIP(&hexarray, 1, 10);
	} else if(scaling == 3) {
		Hexarray_scale(&hexarray, 0, 1, 0);
	} else if(scaling > 3) {
		Hexarray_scale(&hexarray, 1, 1, 0);
	}


	// Hex -> square

	clock_gettime(CLOCK_MONOTONIC, &begin);

	sqsampleColour(hexarray, &array, radius, s2h_scale, mode, threads);

	clock_gettime(CLOCK_MONOTONIC, &end);
	CLOCK_DIFF(5);

	Hexarray_free(&hexarray, 0);


	// array -> img

	mw = NewMagickWand();
	PixelWand* pw = NewPixelWand();

	MagickNewImage(mw, array.x, array.y, pw);
	img = (uint8_t*)calloc(array.x * array.y * 3, sizeof(uint8_t));

	for(unsigned int h = 0; h < array.y; h++) {
		for(unsigned int w = 0; w < array.x; w++) {
			const unsigned int p = h * array.x * 3 + 3 * w;

			img[p]     = array.p[w][h][0]; // R
			img[p + 1] = array.p[w][h][1]; // G
			img[p + 2] = array.p[w][h][2]; // B
		}
	}

	strcpy(ofname, title);
	strcat(ofname, "_h2s.png");

	MagickImportImagePixels(mw, 0, 0, array.x, array.y, "RGB", CharPixel, img);
	MagickWriteImage(mw, ofname);

	DestroyPixelWand(pw);
	DestroyMagickWand(mw);

	free(img);

	strcpy(ofname, title);
	strcat(ofname, "_h2s.ppm");
	pArray2d2PPM(array, ofname);
	pArray2d_free(&array);


	// Free precalculations

	clock_gettime(CLOCK_MONOTONIC, &begin);

	precalcs_free();

	clock_gettime(CLOCK_MONOTONIC, &end);
	CLOCK_DIFF(6);


	printf(
		"\nLookup tables (LUT): init                            : %.6fs\n"
		  "Square-to-hexagonal image transformation (s2h)       : %.6fs\n"
		  "Hexagonal discrete cosine transform (H-DCT)          : %.6fs\n"
		  "Hexagonal quantization                               : %.6fs\n"
		  "Hexagonal inverse discrete cosine transform (H-IDCT) : %.6fs\n"
		  "Hexagonal-to-square image transformation (h2s)       : %.6fs\n"
		  "Lookup tables (LUT): free                            : %.6fs\n",
		 runtimes[0],
		 runtimes[1],
		 runtimes[2],
		 runtimes[3],
		 runtimes[4],
		 runtimes[5],
		 runtimes[6]);




	/**************************************************************************
	 * Termination
	 **************************************************************************/

	MagickWandTerminus();


	/**************************************************************************
	 * Exit CHIP
	 **************************************************************************/

	return 0;
}

