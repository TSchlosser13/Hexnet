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


#define _POSIX_C_SOURCE 199309L


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


int main(int argc, char** argv) {
	bool enable_compare_s2s = false;
	bool enable_compare_s2h = false;
	u32  compare_metric     = COMPARE_PSNR;
	bool display_results    = false;
	bool increase_verbosity = false;

	char* filename_in    = NULL;
	char* filename_in_s2 = NULL;
	char* filename_out   = NULL;
	char  filename_out_h2s[256];
	char  filename_out_s2s[256];

	Array array_in;
	Array array_in_s2;
	Array array_out;
	float array_out_len        = 0;
	Array array_out_s2s;
	u32   array_out_s2s_width  = 0;
	u32   array_out_s2s_height = 0;

	Hexarray hexarray;
	float    hexarray_rad_o = 0;

	double clock_diff;
	struct timespec ts1, ts2;
	int    status = 0;


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

		} else if(!strcmp(argv[i], "--s2h-rad")        && i + 1 < argc) {
			hexarray_rad_o     = (float)atof(argv[++i]);
		} else if(!strcmp(argv[i], "--h2s-len")        && i + 1 < argc) {
			array_out_len      = (float)atof(argv[++i]);
		} else if(!strcmp(argv[i], "--s2s-res")        && i + 1 < argc) {
			array_out_s2s_width = (u32)atoi(argv[++i]);

			if(i + 1 < argc) {
				if(!isdigit(argv[i + 1][0])) {
					array_out_s2s_height = array_out_s2s_width;
				} else {
					array_out_s2s_height = (u32)atoi(argv[++i]);
				}
			} else {
				array_out_s2s_height = array_out_s2s_width;
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

	putchar('\n');


	MagickWandGenesis();
	file_to_Array(filename_in, &array_in);

	if(enable_compare_s2s)
		file_to_Array(filename_in_s2, &array_in_s2);

	printf("%s: %u x %u\n", filename_in, array_in.width, array_in.height);

	if(increase_verbosity) {
		puts("\n");
		Array_print_info(array_in, stringify(array_in));
		putchar('\n');

		if(enable_compare_s2s) {
			Array_print_info(array_in_s2, stringify(array_in_s2));
			putchar('\n');
		}
	}


	clock_gettime(CLOCK_MONOTONIC, &ts1);

	if(hexarray_rad_o) {
		Hexarray_init_from_Array(&hexarray, array_in, hexarray_rad_o);
	} else {
		file_to_Hexarray(filename_in, &hexarray);
	}

	clock_gettime(CLOCK_MONOTONIC, &ts2);
	clock_diff = CLOCK_DIFF(ts1, ts2);


	if(increase_verbosity)
		Hexarray_print_info(hexarray, stringify(hexarray));


	if(array_out_len) {
		Array_init_from_Hexarray(&array_out, hexarray, array_out_len);

		if(increase_verbosity) {
			putchar('\n');
			Array_print_info(array_out, stringify(array_out));
		}
	}

	if(array_out_s2s_width) {
		Array_init(&array_out_s2s, array_out_s2s_width, array_out_s2s_height, 3, 1.0f);

		if(increase_verbosity) {
			putchar('\n');
			Array_print_info(array_out_s2s, stringify(array_out_s2s));
		}
	}


	if(increase_verbosity)
		puts("\n");

	printf("Hexarray_init    : %f\n", clock_diff);


	if(hexarray_rad_o) {
		clock_gettime(CLOCK_MONOTONIC, &ts1);

		Hexsamp_s2h(array_in, &hexarray, 0);

		clock_gettime(CLOCK_MONOTONIC, &ts2);
		clock_diff = CLOCK_DIFF(ts1, ts2);
		printf("Hexsamp_s2h      : %f\n", clock_diff);
	}


	if(enable_compare_s2s || enable_compare_s2h) {
		if(increase_verbosity)
			putchar('\n');

		if(enable_compare_s2s)
			compare_s2s(array_in, array_in_s2, compare_metric);

		if(enable_compare_s2h)
			compare_s2h(array_in, hexarray, compare_metric);

		if(increase_verbosity)
			putchar('\n');
	}


	if(filename_out) {
		clock_gettime(CLOCK_MONOTONIC, &ts1);

		Hexarray_to_file(hexarray, filename_out);

		clock_gettime(CLOCK_MONOTONIC, &ts2);
		clock_diff = CLOCK_DIFF(ts1, ts2);
		printf("Hexarray_to_file : %f\n", clock_diff);
	}


	if(array_out_len) {
		clock_gettime(CLOCK_MONOTONIC, &ts1);

		Hexsamp_h2s(hexarray, &array_out, 0);

		clock_gettime(CLOCK_MONOTONIC, &ts2);
		clock_diff = CLOCK_DIFF(ts1, ts2);
		printf("Hexsamp_h2s      : %f\n", clock_diff);
	}


	if(array_out_s2s_width) {
		clock_gettime(CLOCK_MONOTONIC, &ts1);

		Sqsamp_s2s(array_in, &array_out_s2s, 0);

		clock_gettime(CLOCK_MONOTONIC, &ts2);
		clock_diff = CLOCK_DIFF(ts1, ts2);
		printf("Sqsamp_s2s       : %f\n", clock_diff);
	}


	if(display_results) {
		char gui_title[256];
		sprintf(gui_title, "Hexnet " HEXNET_VERSION ", " HEXNET_YEAR_S " - %s", filename_in);
		status = gui_run(gui_title, 512, 512, hexarray);
	}


	clock_gettime(CLOCK_MONOTONIC, &ts1);

	Hexarray_free(&hexarray);

	clock_gettime(CLOCK_MONOTONIC, &ts2);
	clock_diff = CLOCK_DIFF(ts1, ts2);
	printf("Hexarray_free    : %f\n", clock_diff);


	if(filename_out && array_out_len) {
		strcpy(filename_out_h2s, filename_out);
		insert_string(filename_out_h2s, "_h2s", strrchr(filename_out, '.') - filename_out);
		Array_to_file(array_out, filename_out_h2s);
	}

	if(filename_out && array_out_s2s_width) {
		strcpy(filename_out_s2s, filename_out);
		insert_string(filename_out_s2s, "_s2s", strrchr(filename_out, '.') - filename_out);
		Array_to_file(array_out_s2s, filename_out_s2s);
	}

	Array_free(&array_in);

	if(enable_compare_s2s)
		Array_free(&array_in_s2);

	if(array_out_len)
		Array_free(&array_out);

	if(array_out_s2s_width)
		Array_free(&array_out_s2s);

	MagickWandTerminus();


	return status;
}

