/******************************************************************************
 * Sqsamp.c: Square Image Transformation
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

#include "Sqsamp.h"

#include "../misc/defines.h"
#include "../misc/types.h"


/******************************************************************************
 * Interpolation kernel function
 ******************************************************************************/

float Sqsamp_kernel(float x, float y, u32 method) {
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
 * Array to Array transformation
 ******************************************************************************/

void Sqsamp_s2s(Array s1, Array* s2, u32 method) {
	const float ws = (float)s1.width  / s2->width;
	const float hs = (float)s1.height / s2->height;
	const float r  = MAX(1.0f, MAX(ws, hs));
	const i32   ri = (i32)ceilf(r);

	for(u32 h = 0; h < s2->height; h++) {
		const float h_s1   = h * hs;
		const i32   h_s1_i = roundf(h_s1);

		for(u32 w = 0; w < s2->width; w++) {
			const float w_s1   = w * ws;
			const i32   w_s1_i = roundf(w_s1);
			      float o[3]   = { 0.0f, 0.0f, 0.0f };
			      float on     =   0.0f;


			for(i32 y = h_s1_i - ri; y <= h_s1_i + ri; y++) {
				for(i32 x = w_s1_i - ri; x <= w_s1_i + ri; x++) {
					const float rx = fabs(w_s1 - x);
					const float ry = fabs(h_s1 - y);

					if(x >= 0 && x < s1.width && y >= 0 && y < s1.height && rx < r && ry < r) {
						const float k = Sqsamp_kernel(rx / r, ry / r, method);
						const u32   p = 3 * (y * s1.width + x);

						o[0] += k * s1.p[p];
						o[1] += k * s1.p[p + 1];
						o[2] += k * s1.p[p + 2];
						on   += k;
					}
				}
			}

			if(on) {
				o[0] /= on;
				o[1] /= on;
				o[2] /= on;
			}


			const u32 p = 3 * (h * s2->width + w);

			if(o[0] < 255) {
				s2->p[p]     = (u32)roundf(o[0]);
			} else {
				s2->p[p]     = 255;
			}

			if(o[1] < 255) {
				s2->p[p + 1] = (u32)roundf(o[1]);
			} else {
				s2->p[p + 1] = 255;
			}

			if(o[2] < 255) {
				s2->p[p + 2] = (u32)roundf(o[2]);
			} else {
				s2->p[p + 2] = 255;
			}
		}
	}
}

