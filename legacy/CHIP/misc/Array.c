/******************************************************************************
 * Array.c
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


#include <stdio.h>
#include <stdlib.h>

#include "types.h"

#include "Array.h"


void pArray2d_init(RGB_Array* rgb_array, unsigned int x, unsigned int y) {
	rgb_array->x = x;
	rgb_array->y = y;


	rgb_array->p = (int***)malloc(rgb_array->x * sizeof(int**));

	for(x = 0; x < rgb_array->x; x++) {
		rgb_array->p[x] = (int**)malloc(rgb_array->y * sizeof(int*));

		for(y = 0; y < rgb_array->y; y++)
			rgb_array->p[x][y] = (int*)calloc(3, sizeof(int));
	}
}

void pArray2d_free(RGB_Array* rgb_array) {
	for(unsigned int x = 0; x < rgb_array->x; x++) {
		for(unsigned int y = 0; y < rgb_array->y; y++)
			free(rgb_array->p[x][y]);

		free(rgb_array->p[x]);
	}

	free(rgb_array->p);
}


// pArray2d to Portable PixMap
void pArray2d2PPM(RGB_Array rgb_array, char* filename) {
	FILE* PPM = fopen(filename, "w");

	fprintf(PPM, "P3\n# %s\n%u %u\n255\n", filename, rgb_array.x, rgb_array.y);

	for(unsigned int h = 0; h < rgb_array.y; h++) {
		for(unsigned int w = 0; w < rgb_array.x; w++) {
			for(unsigned int c = 0; c < 3; c++) {
				if(c) {
					if(rgb_array.p[w][h][c] < 10) {
						fprintf(PPM, "   %u", rgb_array.p[w][h][c]);
					} else if(rgb_array.p[w][h][c] < 100) {
						fprintf(PPM, "  %u",  rgb_array.p[w][h][c]);
					} else {
						fprintf(PPM, " %u",   rgb_array.p[w][h][c]);
					}
				} else {
					if(rgb_array.p[w][h][c] < 10) {
						fprintf(PPM, "  %u", rgb_array.p[w][h][c]);
					} else if(rgb_array.p[w][h][c] < 100) {
						fprintf(PPM, " %u",  rgb_array.p[w][h][c]);
					} else {
						fprintf(PPM, "%u",   rgb_array.p[w][h][c]);
					}
				}
			}

			if(w < rgb_array.x - 1) {
				fputs("  ", PPM);
			} else {
				fputc('\n', PPM);
			}
		}
	}

	fclose(PPM);
}

