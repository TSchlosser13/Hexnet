/******************************************************************************
 * Hexsamp.c: Hexagonale Bildtransformationen
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

#include <math.h>
#include <pthread.h>

#include "../misc/Array.h"
#include "../misc/defines.h"
#include "../misc/precalculations.h"
#include "../misc/types.h"

#include "Hexint.h"
#include "Hexarray.h"

#include "Hexsamp.h"


// pthreads structure for hexagonal-to-square image transformation
struct pthreads_args {
	RGB_Hexarray rgb_hexarray;
	RGB_Array*   rgb_array;
	float        radius;
	float        scale;
	iPoint2d     cart_min;
	unsigned int technique;
	unsigned int threads;
	unsigned int thread_id;
};


// Sinc-Funktion
float sinc(float x) {
	return x ? sin(M_PI * x) / (M_PI * x) : 1.0f;
}

// Kernel (Modi: BL / BC / Lanczos / B-Splines (B_3))
float kernel(float x, float y, unsigned int technique) {
	fPoint2d f;

	// BL
	if(!technique) {
		const fPoint2d abs = { .x = fabs(x), .y = fabs(y) };

		if(abs.x >= 0 && abs.x < 1) {
			f.x = 1 - abs.x;
		} else {
			f.x = 0.0f;
		}
		if(abs.y >= 0 && abs.y < 1) {
			f.y = 1 - abs.y;
		} else {
			f.y = 0.0f;
		}
	// BC
	} else if(technique == 1) {
		const fPoint2d abs   = { .x = fabs(x),       .y = fabs(y) };
		const fPoint2d abs_2 = { .x = abs.x * abs.x, .y = abs.y * abs.y };

		if(abs.x >= 0 && abs.x < 1) {
			f.x = 2 * abs_2.x * abs.x - 3 * abs_2.x + 1;
		} else {
			f.x = 0.0f;
		}
		if(abs.y >= 0 && abs.y < 1) {
			f.y = 2 * abs_2.y * abs.y - 3 * abs_2.y + 1;
		} else {
			f.y = 0.0f;
		}
	// Lanczos
	} else if(technique == 2) {
		const fPoint2d abs = { .x = fabs(x), .y = fabs(y) };

		if(abs.x >= 0 && abs.x < 1) {
			f.x = sinc(abs.x) * sinc(abs.x / 2);
		} else {
			f.x = 0.0f;
		}
		if(abs.y >= 0 && abs.y < 1) {
			f.y = sinc(abs.y) * sinc(abs.y / 2);
		} else {
			f.y = 0.0f;
		}
	// B-Splines (B_3)
	} else {
		const fPoint2d abs   = { .x = 2 * fabs(x),   .y = 2 * fabs(y) };
		const fPoint2d abs_2 = { .x = abs.x * abs.x, .y = abs.y * abs.y };

		if(abs.x < 1) {
			f.x = (3 * abs_2.x * abs.x - 6 * abs_2.x + 4) / 6;
		} else if(abs.x < 2) {
			f.x = (-abs_2.x * abs.x + 6 * abs_2.x - 12 * abs.x + 8) / 6;
		} else {
			f.x = 0.0f;
		}
		if(abs.y < 1) {
			f.y = (3 * abs_2.y * abs.y - 6 * abs_2.y + 4) / 6;
		} else if(abs.y < 2) {
			f.y = (-abs_2.y * abs.y + 6 * abs_2.y - 12 * abs.y + 8) / 6;
		} else {
			f.y = 0.0f;
		}
	}

	return f.x * f.y;
}


// Sq -> HIP
void hipsampleColour(RGB_Array rgb_array, RGB_Hexarray* rgb_hexarray,
 unsigned int order, float scale, unsigned int technique) {
	const fPoint2d cart_a = { .x = rgb_array.x / 2.0f, .y = rgb_array.y / 2.0f };

	Hexarray_init(rgb_hexarray, order, 0);

	for(unsigned int i = 0; i < rgb_hexarray->size; i++) {
		const float row_hex = cart_a.y - scale * pc_reals[i][1];
		const float col_hex = cart_a.x + scale * pc_reals[i][0];
		const int   row     = roundf(row_hex);
		const int   col     = roundf(col_hex);
		      float out[3]  = { 0.0f, 0.0f, 0.0f };
		      float out_n   =   0.0f;

		// Patch Transformation: Reichweite (7x7 -> 3x3 = 49 -> 9)
		// for(int x = col - 3; x <= col + 3; x++) {
		for(int x = col - 1; x <= col + 1; x++) {
			// for(int y = row - 3; y <= row + 3; y++) {
			for(int y = row - 1; y <= row + 1; y++) {
				if(x >= 0 && x < rgb_array.x && y >= 0 && y < rgb_array.y) {
					const float xh = fabs(col_hex - x);
					const float yh = fabs(row_hex - y);
					      float k  = kernel(xh, yh, technique);


					// Patch Transformation: Flächeninhalt
					if((xh > 0.5f && xh < 1.0f) || (yh > 0.5f && yh < 1.0f)) {
						float factor = 1.0f;

						if(xh > 0.5f)
							factor *= (1.5f - xh);

						if(yh > 0.5f)
							factor *= (1.5f - yh);

						k *= factor;
					}


					out[0] += k * rgb_array.p[x][y][0];
					out[1] += k * rgb_array.p[x][y][1];
					out[2] += k * rgb_array.p[x][y][2];
					out_n  += k;
				}
			}
		}

		// Patch Transformation: Normalisierung
		if(out_n > 0.0f) {
			rgb_hexarray->p[i][0] = (int)roundf(out[0] / out_n);
			rgb_hexarray->p[i][1] = (int)roundf(out[1] / out_n);
			rgb_hexarray->p[i][2] = (int)roundf(out[2] / out_n);
		} else {
			rgb_hexarray->p[i][0] = (int)roundf(out[0]);
			rgb_hexarray->p[i][1] = (int)roundf(out[1]);
			rgb_hexarray->p[i][2] = (int)roundf(out[2]);
		}
	}
}


// HIP -> sq

// Pthreads
void* resample(void* args_p) {
	struct pthreads_args* args = args_p;

	const RGB_Hexarray rgb_hexarray = args->rgb_hexarray;
	const RGB_Array*   rgb_array    = args->rgb_array;
	const float        radius       = args->radius;
	const float        scale        = args->scale;
	const iPoint2d     cart_min     = args->cart_min;
	const unsigned int technique    = args->technique;
	const unsigned int threads      = args->threads;
	const unsigned int thread_id    = args->thread_id;


	fPoint2d cart_a = { .x = cart_min.x, .y = cart_min.y + thread_id * scale };
	const unsigned int i_max = radius > 1.0f ? 49 : 7; // TODO?

	for(unsigned int y = thread_id; y < rgb_array->y; y += threads) {
		for(unsigned int x = 0; x < rgb_array->x; x++) {
			float out[3] = { 0.0f, 0.0f, 0.0f };
			float out_n  =   0.0f;


			for(unsigned int i = 0; i < i_max; i++) {
				const unsigned int hi = pc_adds[pc_nearest[x][y]][i];
				// const unsigned int hi = pc_adds[x][y][i]; // schlechtere Lokalität

				if(hi < rgb_hexarray.size) {
					const fPoint2d cart_ha = { .x = pc_reals[hi][0], .y = pc_reals[hi][1] };

					if(fabs(cart_a.x - cart_ha.x) <= radius && fabs(cart_a.y - cart_ha.y) <= radius) {
						const float k = kernel(cart_a.x - cart_ha.x, cart_a.y - cart_ha.y, technique);

						out[0] += k * rgb_hexarray.p[hi][0];
						out[1] += k * rgb_hexarray.p[hi][1];
						out[2] += k * rgb_hexarray.p[hi][2];
						out_n  += k;
					}
				}
			}


			// Patch Transformation: Normalisierung
			if(out_n > 0.0f) {
				out[0] = roundf(out[0] / out_n);
				out[1] = roundf(out[1] / out_n);
				out[2] = roundf(out[2] / out_n);
			} else {
				out[0] = roundf(out[0]);
				out[1] = roundf(out[1]);
				out[2] = roundf(out[2]);
			}

			if(out[0] < 255) {
				rgb_array->p[x][rgb_array->y - y - 1][0] = (int)out[0];
			} else {
				rgb_array->p[x][rgb_array->y - y - 1][0] = 255;
			}
			if(out[1] < 255) {
				rgb_array->p[x][rgb_array->y - y - 1][1] = (int)out[1];
			} else {
				rgb_array->p[x][rgb_array->y - y - 1][1] = 255;
			}
			if(out[2] < 255) {
				rgb_array->p[x][rgb_array->y - y - 1][2] = (int)out[2];
			} else {
				rgb_array->p[x][rgb_array->y - y - 1][2] = 255;
			}


			cart_a.x += scale;
		}

		cart_a.x  = cart_min.x;
		cart_a.y += threads * scale;
	}


	free(args);

	return NULL;
}

void sqsampleColour(RGB_Hexarray rgb_hexarray, RGB_Array* rgb_array,
 float radius, float scale, unsigned int technique, unsigned int threads) {
	rgb_array->x = (unsigned int)roundf((pc_rmx.x - pc_rmn.x) / scale) + 1;
	rgb_array->y = (unsigned int)roundf((pc_rmx.y - pc_rmn.y) / scale) + 1;
	pArray2d_init(rgb_array, rgb_array->x, rgb_array->y);


	// Pthreads

	struct pthreads_args** threads_args = (struct pthreads_args**)malloc(threads * sizeof(struct pthreads_args*));

	for(unsigned int i = 0; i < threads; i++) {
		threads_args[i] = (struct pthreads_args*)malloc(sizeof(struct pthreads_args));

		threads_args[i]->rgb_hexarray = rgb_hexarray;
		threads_args[i]->rgb_array    = rgb_array;
		threads_args[i]->radius       = radius;
		threads_args[i]->scale        = scale;
		threads_args[i]->cart_min     = pc_rmn;
		threads_args[i]->technique    = technique;
		threads_args[i]->threads      = threads;
		threads_args[i]->thread_id    = i;
	}

	if(threads > 1) {
		pthread_t* threads_ids = (pthread_t*)malloc(threads * sizeof(pthread_t));

		for(unsigned int i = 0; i < threads; i++)
			pthread_create(&threads_ids[i], NULL, &resample, threads_args[i]);

		for(unsigned int i = 0; i < threads; i++)
			pthread_join(threads_ids[i], NULL);

		free(threads_ids);
	} else {
		resample(threads_args[0]);
	}

	free(threads_args);
}

