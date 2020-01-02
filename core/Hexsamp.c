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


#include <math.h>

#include "Hexsamp.h"

#include "../misc/defines.h"
#include "../misc/types.h"


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

