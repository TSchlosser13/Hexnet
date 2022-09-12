/******************************************************************************
 * Hexsamp.c: Hexagonal Image Transformation
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

#include "Hexsamp.h"

#include "../misc/defines.h"
#include "../misc/types.h"


/******************************************************************************
 * Interpolation kernel function
 ******************************************************************************/

float Hexsamp_kernel(float x, float y, u32 method) {
	float k;

	if(!method) {
		if(x < 1 && y < 1) {
			k = (1 - x) * (1 - y);
		} else {
			k = 0.0f;
		}
	} else {
		k = 1.0f;
	}

	return k;
}


/******************************************************************************
 * Array to Hexarray transformation
 ******************************************************************************/

void Hexsamp_s2h(Array array, Hexarray* hexarray, u32 method) {
	const float r  = MAX(1.0f, hexarray->dia_i);
	const i32   ri = (i32)ceilf(r);
	const float wb = ((array.width  - 1) - (hexarray->width_hex  - hexarray->dia_i)) / 2;
	const float hb = ((array.height - 1) - (hexarray->height_hex - hexarray->dia_o)) / 2;

	for(u32 h = 0; h < hexarray->height; h++) {
		const float hh  = hb + h * hexarray->dist_h;
		const i32   hhi = roundf(hh);

		for(u32 w = 0; w < hexarray->width; w++) {
			const float wh   = !(h % 2) ? wb + w * hexarray->dia_i : wb + hexarray->rad_i + w * hexarray->dia_i;
			const i32   whi  = roundf(wh);
			      float o[3] = { 0.0f, 0.0f, 0.0f };
			      float on   =   0.0f;


			for(i32 y = hhi - ri; y <= hhi + ri; y++) {
				for(i32 x = whi - ri; x <= whi + ri; x++) {
					const float rx = fabs(wh - x);
					const float ry = fabs(hh - y);

					if(x >= 0 && x < array.width && y >= 0 && y < array.height && rx < r && ry < r) {
						const float k = Hexsamp_kernel(rx / r, ry / r, method);
						const u32   p = 3 * (y * array.width + x);

						o[0] += k * array.p[p];
						o[1] += k * array.p[p + 1];
						o[2] += k * array.p[p + 2];
						on   += k;
					}
				}
			}

			if(on) {
				o[0] /= on;
				o[1] /= on;
				o[2] /= on;
			}


			const u32 p = 3 * (h * hexarray->width + w);

			if(o[0] < 255) {
				hexarray->p[p]     = (u32)roundf(o[0]);
			} else {
				hexarray->p[p]     = 255;
			}

			if(o[1] < 255) {
				hexarray->p[p + 1] = (u32)roundf(o[1]);
			} else {
				hexarray->p[p + 1] = 255;
			}

			if(o[2] < 255) {
				hexarray->p[p + 2] = (u32)roundf(o[2]);
			} else {
				hexarray->p[p + 2] = 255;
			}
		}
	}
}


/******************************************************************************
 * Hexarray to Array transformation
 ******************************************************************************/

void Hexsamp_h2s(Hexarray hexarray, Array* array, u32 method) {
	const float r  = MAX(hexarray.dia_i, array->len);
	const i32   ri = (i32)ceilf(r);
	const float wb = ((hexarray.width_hex  - hexarray.dia_i) - (array->width_sq  - array->len)) / 2;
	const float hb = ((hexarray.height_hex - hexarray.dia_o) - (array->height_sq - array->len)) / 2;

	for(u32 h = 0; h < array->height; h++) {
		const float hs   = hb + h * array->len;
		const float hsh  = hs / hexarray.dist_h;
		const i32   hshi = roundf(hsh);

		for(u32 w = 0; w < array->width; w++) {
			const float ws   = wb + w * array->len;
			const float wsh  = !(hshi % 2) ? ws / hexarray.dia_i : -hexarray.rad_i + ws / hexarray.dia_i;
			const i32   wshi = roundf(wsh);
			      float o[3] = { 0.0f, 0.0f, 0.0f };
			      float on   =   0.0f;


			for(i32 y = hshi - ri; y <= hshi + ri; y++) {
				const float hh = y * hexarray.dist_h;

				for(i32 x = wshi - ri; x <= wshi + ri; x++) {
					const float wh = !(y % 2) ? x * hexarray.dia_i : hexarray.rad_i + x * hexarray.dia_i;
					const float rx = fabs(ws - wh);
					const float ry = fabs(hs - hh);

					if(x >= 0 && x < hexarray.width && y >= 0 && y < hexarray.height && rx < r && ry < r) {
						const float k = Hexsamp_kernel(rx / r, ry / r, method);
						const u32   p = 3 * (y * hexarray.width + x);

						o[0] += k * hexarray.p[p];
						o[1] += k * hexarray.p[p + 1];
						o[2] += k * hexarray.p[p + 2];
						on   += k;
					}
				}
			}

			if(on) {
				o[0] /= on;
				o[1] /= on;
				o[2] /= on;
			}


			const u32 p = 3 * (h * array->width + w);

			if(o[0] < 255) {
				array->p[p]     = (u32)roundf(o[0]);
			} else {
				array->p[p]     = 255;
			}

			if(o[1] < 255) {
				array->p[p + 1] = (u32)roundf(o[1]);
			} else {
				array->p[p + 1] = 255;
			}

			if(o[2] < 255) {
				array->p[p + 2] = (u32)roundf(o[2]);
			} else {
				array->p[p + 2] = 255;
			}
		}
	}
}


/******************************************************************************
 * Hexarray to Hexarray transformation
 ******************************************************************************/

void Hexsamp_h2h(Hexarray h1, Hexarray* h2, u32 method) {
	const float r  = MAX(h1.dia_i, h2->dia_i);
	const i32   ri = (i32)ceilf(r);
	const float wb = ((h1.width_hex  - h1.dia_i) - (h2->width_hex  - h2->dia_i)) / 2;
	const float hb = ((h1.height_hex - h1.dia_o) - (h2->height_hex - h2->dia_o)) / 2;

	for(u32 h = 0; h < h2->height; h++) {
		const float ht   = hb + h * h2->dist_h;
		const float hth  = ht / h1.dist_h;
		const i32   hthi = roundf(hth);

		for(u32 w = 0; w < h2->width; w++) {
			const float wt   = !(h    % 2) ? wb + w * h2->dia_i : wb + h2->rad_i + w * h2->dia_i;
			const float wth  = !(hthi % 2) ? wt / h1.dia_i      : -h1.rad_i + wt / h1.dia_i;
			const i32   wthi = roundf(wth);
			      float o[3] = { 0.0f, 0.0f, 0.0f };
			      float on   =   0.0f;


			for(i32 y = hthi - ri; y <= hthi + ri; y++) {
				const float hh = y * h1.dist_h;

				for(i32 x = wthi - ri; x <= wthi + ri; x++) {
					const float wh = !(y % 2) ? x * h1.dia_i : h1.rad_i + x * h1.dia_i;
					const float rx = fabs(wt - wh);
					const float ry = fabs(ht - hh);

					if(x >= 0 && x < h1.width && y >= 0 && y < h1.height && rx < r && ry < r) {
						const float k = Hexsamp_kernel(rx / r, ry / r, method);
						const u32   p = 3 * (y * h1.width + x);

						o[0] += k * h1.p[p];
						o[1] += k * h1.p[p + 1];
						o[2] += k * h1.p[p + 2];
						on   += k;
					}
				}
			}

			if(on) {
				o[0] /= on;
				o[1] /= on;
				o[2] /= on;
			}


			const u32 p = 3 * (h * h2->width + w);

			if(o[0] < 255) {
				h2->p[p]     = (u32)roundf(o[0]);
			} else {
				h2->p[p]     = 255;
			}

			if(o[1] < 255) {
				h2->p[p + 1] = (u32)roundf(o[1]);
			} else {
				h2->p[p + 1] = 255;
			}

			if(o[2] < 255) {
				h2->p[p + 2] = (u32)roundf(o[2]);
			} else {
				h2->p[p + 2] = 255;
			}
		}
	}
}


