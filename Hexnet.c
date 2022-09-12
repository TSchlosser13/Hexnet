/******************************************************************************
 * Hexnet.c: The Hexagonal Image Processing Framework
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
 * Defines
 ******************************************************************************/

#define _POSIX_C_SOURCE 199309L


/******************************************************************************
 * Includes
 ******************************************************************************/

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "MagickWand/MagickWand.h"

#include "Hexnet.h"

#include "core/Hexarray.h"
#include "core/Hexsamp.h"

#include "gui/gui.h"

#include "misc/Array.h"
#include "misc/defines.h"
#include "misc/print.h"
#include "misc/strings.h"
#include "misc/types.h"

#include "tools/compare.h"
#include "tools/Sqsamp.h"


/******************************************************************************
 * main
 ******************************************************************************/

int main(int argc, char** argv) {

	/**************************************************************************
	 * Variables
	 **************************************************************************/

	bool enable_compare_s2s = false;
	bool enable_compare_s2h = false;
	u32  compare_metric     = COMPARE_PSNR;
	bool display_results    = false;
	bool increase_verbosity = false;

	char* filename_in    = NULL;
	char* filename_in_s2 = NULL;
	char* filename_out   = NULL;
	char  filename_out_h2s[256];
	char  filename_out_h2h[256];
	char  filename_out_s2s[256];

	Array array;
	Array array_s2;
	Array array_h2s;
	float array_h2s_len    = 0;
	Array array_s2s;
	u32   array_s2s_width  = 0;
	u32   array_s2s_height = 0;

	Hexarray hexarray;
	float    hexarray_rad_o     = 0;
	Hexarray hexarray_h2h;
	float    hexarray_h2h_rad_o = 0;

	double clock_diff;
	struct timespec ts1, ts2;
	int    status = 0;


	/**************************************************************************
	 * Parameters
	 **************************************************************************/

	setbuf(stdout, NULL);
	setbuf(stderr, NULL);

	print_info();

	for(u32 i = 1; i < argc; i++) {
		if(!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) {
			print_usage();
			putchar('\n');
			print_help_full();
			return 0;


		} else if((!strcmp(argv[i], "-i") || !strcmp(argv[i], "--input"))  && i + 1 < argc) {
			filename_in  = argv[++i];
		} else if((!strcmp(argv[i], "-o") || !strcmp(argv[i], "--output")) && i + 1 < argc) {
			filename_out = argv[++i];

		} else if(!strcmp(argv[i], "--s2h-rad") && i + 1 < argc) {
			hexarray_rad_o     = (float)atof(argv[++i]);
		} else if(!strcmp(argv[i], "--h2s-len") && i + 1 < argc) {
			array_h2s_len      = (float)atof(argv[++i]);
		} else if(!strcmp(argv[i], "--h2h-rad") && i + 1 < argc) {
			hexarray_h2h_rad_o = (float)atof(argv[++i]);
		} else if(!strcmp(argv[i], "--s2s-res") && i + 1 < argc) {
			array_s2s_width    = (u32)atoi(argv[++i]);

			if(i + 1 < argc) {
				if(!isdigit(argv[i + 1][0])) {
					array_s2s_height = array_s2s_width;
				} else {
					array_s2s_height = (u32)atoi(argv[++i]);
				}
			} else {
				array_s2s_height = array_s2s_width;
			}
		} else if(!strcmp(argv[i], "--compare-s2s")    && i + 1 < argc) {
			enable_compare_s2s = true;
			filename_in_s2     = argv[++i];
		} else if(!strcmp(argv[i], "--compare-s2h")) {
			enable_compare_s2h = true;
		} else if(!strcmp(argv[i], "--compare-metric") && i + 1 < argc) {
			compare_metric     = get_compare_method(argv[++i]);

		} else if(!strcmp(argv[i], "-d") || !strcmp(argv[i], "--display")) {
			display_results    = true;
		} else if(!strcmp(argv[i], "-v") || !strcmp(argv[i], "--verbose")) {
			increase_verbosity = true;


		} else {
			print_usage();
			print_error("Unrecognized option \"%s\".\n", argv[i]);
			print_help();
			return 1;
		}
	}

	if(!filename_in) {
		print_usage();
		print_help();
		return 0;
	}

	puts("\n");


	/**************************************************************************
	 * Initialization
	 **************************************************************************/

	MagickWandGenesis();




	/**************************************************************************
	 * Load data from file: file_to_Array()
	 **************************************************************************/

	if(hexarray_rad_o || enable_compare_s2s || array_s2s_width) {
		file_to_Array(filename_in, &array);

		if(enable_compare_s2s)
			file_to_Array(filename_in_s2, &array_s2);

		if(increase_verbosity) {
			Array_print_info(array, stringify(array));
			putchar('\n');

			if(enable_compare_s2s) {
				Array_print_info(array_s2, stringify(array_s2));
				putchar('\n');
			}
		}
	}


	/**************************************************************************
	 * Load data from file: Hexarray_init_from_Array() / file_to_Hexarray()
	 **************************************************************************/

	clock_gettime(CLOCK_MONOTONIC, &ts1);

	if(hexarray_rad_o) {
		Hexarray_init_from_Array(&hexarray, array, hexarray_rad_o);
	} else {
		file_to_Hexarray(filename_in, &hexarray);
	}

	clock_gettime(CLOCK_MONOTONIC, &ts2);
	clock_diff = CLOCK_DIFF(ts1, ts2);

	if(increase_verbosity)
		Hexarray_print_info(hexarray, stringify(hexarray));


	/**************************************************************************
	 * Load data from file: Array_init_from_Hexarray()
	 **************************************************************************/

	if(array_h2s_len) {
		Array_init_from_Hexarray(&array_h2s, hexarray, array_h2s_len);

		if(increase_verbosity) {
			putchar('\n');
			Array_print_info(array_h2s, stringify(array_h2s));
		}
	}


	/**************************************************************************
	 * Load data from file: Hexarray_init_from_Hexarray()
	 **************************************************************************/

	if(hexarray_h2h_rad_o) {
		Hexarray_init_from_Hexarray(&hexarray_h2h, hexarray, hexarray_h2h_rad_o);

		if(increase_verbosity) {
			putchar('\n');
			Hexarray_print_info(hexarray_h2h, stringify(hexarray_h2h));
		}
	}


	/**************************************************************************
	 * Load data from file: Array_init()
	 **************************************************************************/

	if(array_s2s_width) {
		Array_init(&array_s2s, array_s2s_width, array_s2s_height, 3, 1.0f);

		if(increase_verbosity) {
			putchar('\n');
			Array_print_info(array_s2s, stringify(array_s2s));
		}
	}


	/**************************************************************************
	 * Runtime: Hexarray_init_from_Array() / file_to_Hexarray()
	 **************************************************************************/

	if(increase_verbosity)
		puts("\n");

	printf("Hexarray_init    : %f\n", clock_diff);


	/**************************************************************************
	 * Transformation and runtime: Hexsamp_s2h()
	 **************************************************************************/

	if(hexarray_rad_o) {
		clock_gettime(CLOCK_MONOTONIC, &ts1);

		Hexsamp_s2h(array, &hexarray, 0);

		clock_gettime(CLOCK_MONOTONIC, &ts2);
		clock_diff = CLOCK_DIFF(ts1, ts2);
		printf("Hexsamp_s2h      : %f\n", clock_diff);
	}




	/**************************************************************************
	 * Array compare methods
	 **************************************************************************/

	if(enable_compare_s2s || enable_compare_s2h) {
		if(increase_verbosity)
			putchar('\n');

		if(enable_compare_s2s)
			compare_s2s(array, array_s2, compare_metric);

		if(enable_compare_s2h)
			compare_s2h(array, hexarray, compare_metric);

		if(increase_verbosity)
			putchar('\n');
	}




	/**************************************************************************
	 * Save data to file and runtime: Hexarray_to_file() (hexarray)
	 **************************************************************************/

	if(filename_out) {
		clock_gettime(CLOCK_MONOTONIC, &ts1);

		Hexarray_to_file(hexarray, filename_out);

		clock_gettime(CLOCK_MONOTONIC, &ts2);
		clock_diff = CLOCK_DIFF(ts1, ts2);
		printf("Hexarray_to_file : %f\n", clock_diff);
	}


	/**************************************************************************
	 * Transformation and runtime: Hexsamp_h2s()
	 **************************************************************************/

	if(array_h2s_len) {
		clock_gettime(CLOCK_MONOTONIC, &ts1);

		Hexsamp_h2s(hexarray, &array_h2s, 0);

		clock_gettime(CLOCK_MONOTONIC, &ts2);
		clock_diff = CLOCK_DIFF(ts1, ts2);
		printf("Hexsamp_h2s      : %f\n", clock_diff);
	}


	/**************************************************************************
	 * Transformation and runtime: Hexsamp_h2h()
	 **************************************************************************/

	if(hexarray_h2h_rad_o) {
		clock_gettime(CLOCK_MONOTONIC, &ts1);

		Hexsamp_h2h(hexarray, &hexarray_h2h, 0);

		clock_gettime(CLOCK_MONOTONIC, &ts2);
		clock_diff = CLOCK_DIFF(ts1, ts2);
		printf("Hexsamp_h2h      : %f\n", clock_diff);
	}


	/**************************************************************************
	 * Transformation and runtime: Sqsamp_s2s()
	 **************************************************************************/

	if(array_s2s_width) {
		clock_gettime(CLOCK_MONOTONIC, &ts1);

		Sqsamp_s2s(array, &array_s2s, 0);

		clock_gettime(CLOCK_MONOTONIC, &ts2);
		clock_diff = CLOCK_DIFF(ts1, ts2);
		printf("Sqsamp_s2s       : %f\n", clock_diff);
	}




	/**************************************************************************
	 * Start the Hexnet GUI
	 **************************************************************************/

	if(display_results) {
		char gui_title[256];

		sprintf(gui_title, "Hexnet " HEXNET_VERSION ", " HEXNET_YEAR_S " - %s", filename_in);

		status = gui_run(gui_title, 512, 512, hexarray);
	}




	/**************************************************************************
	 * Free memory and runtime: Hexarray_free() (hexarray)
	 **************************************************************************/

	clock_gettime(CLOCK_MONOTONIC, &ts1);

	Hexarray_free(&hexarray);

	clock_gettime(CLOCK_MONOTONIC, &ts2);
	clock_diff = CLOCK_DIFF(ts1, ts2);
	printf("Hexarray_free    : %f\n", clock_diff);


	/**************************************************************************
	 * Save data to file:
	 *     Array_to_file() and Hexarray_to_file()
	 *     (array_h2s, hexarray_h2h, and array_s2s)
	 **************************************************************************/

	if(filename_out && array_h2s_len) {
		strcpy(filename_out_h2s, filename_out);
		insert_string(filename_out_h2s, "_h2s", strrchr(filename_out, '.') - filename_out);
		Array_to_file(array_h2s, filename_out_h2s);
	}

	if(filename_out && hexarray_h2h_rad_o) {
		strcpy(filename_out_h2h, filename_out);
		insert_string(filename_out_h2h, "_h2h", strrchr(filename_out, '.') - filename_out);
		Hexarray_to_file(hexarray_h2h, filename_out_h2h);
	}

	if(filename_out && array_s2s_width) {
		strcpy(filename_out_s2s, filename_out);
		insert_string(filename_out_s2s, "_s2s", strrchr(filename_out, '.') - filename_out);
		Array_to_file(array_s2s, filename_out_s2s);
	}


	/**************************************************************************
	 * Free memory:
	 *     Array_free() and Hexarray_free()
	 *     (array, array_s2, array_h2s, hexarray_h2h, and array_s2s)
	 **************************************************************************/

	if(hexarray_rad_o || enable_compare_s2s || array_s2s_width)
		Array_free(&array);

	if(enable_compare_s2s)
		Array_free(&array_s2);

	if(array_h2s_len)
		Array_free(&array_h2s);

	if(hexarray_h2h_rad_o)
		Hexarray_free(&hexarray_h2h);

	if(array_s2s_width)
		Array_free(&array_s2s);




	/**************************************************************************
	 * Termination
	 **************************************************************************/

	MagickWandTerminus();


	/**************************************************************************
	 * Exit Hexnet
	 **************************************************************************/

	return status;
}


