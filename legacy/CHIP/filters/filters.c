/******************************************************************************
 * filters.c: Hexagonal Filter Banks
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


#include <math.h>

#include "../misc/defines.h"
#include "../misc/precalculations.h"
#include "../misc/types.h"

#include "../core/Hexint.h"
#include "../core/Hexarray.h"
#include "../core/Hexsamp.h" // sinc

#include "filters.h"


// Modi: Low- (1) / High-pass filters (4: 0°, 120°, 240°, (0° + 120° + 240°) / 3)
void filter_4x16(RGB_Hexarray* rgb_hexarray, unsigned int mode) {
	const unsigned int order = (unsigned int)(logf(rgb_hexarray->size) / logf(7));


	if(mode == 4) {
		RGB_Hexarray deg0;
		RGB_Hexarray deg120;
		RGB_Hexarray deg240;

		Hexarray_init(&deg0,   order, 0);
		Hexarray_init(&deg120, order, 0);
		Hexarray_init(&deg240, order, 0);

		for(unsigned int i = 0; i < rgb_hexarray->size; i++) {
			deg0.p[i][0] = deg120.p[i][0] = deg240.p[i][0] = rgb_hexarray->p[i][0];
			deg0.p[i][1] = deg120.p[i][1] = deg240.p[i][1] = rgb_hexarray->p[i][1];
			deg0.p[i][2] = deg120.p[i][2] = deg240.p[i][2] = rgb_hexarray->p[i][2];
		}

		filter_4x16(&deg0,   1);
		filter_4x16(&deg120, 2);
		filter_4x16(&deg240, 3);

		for(unsigned int i = 0; i < rgb_hexarray->size; i++) {
			rgb_hexarray->p[i][0] = (int)roundf((deg0.p[i][0] + deg120.p[i][0] + deg240.p[i][0]) / 3);
			rgb_hexarray->p[i][1] = (int)roundf((deg0.p[i][1] + deg120.p[i][1] + deg240.p[i][1]) / 3);
			rgb_hexarray->p[i][2] = (int)roundf((deg0.p[i][2] + deg120.p[i][2] + deg240.p[i][2]) / 3);
		}

		return;
	}


	// Low-pass filter
	/*const int low_pass_filter[7][5] =
		{ {    0,    0,    0,  -14,    0 },
		  {    0,    0,    0,  -84,  -14 },
		  {  -14,    0,  276,  259,    0 },
		  {  -84,  259,  759,  276,    0 },
		  {  -14,    0,  276,  259,    0 },
		  {    0,    0,    0,  -84,  -14 },
		  {    0,    0,    0,  -14,    0 } };*/
	const int low_pass_filter[16] = {
		-14, -84, -14, -14, 276, 259, -84, 259, 759, 276, -14, 276, 259, -84, -14, -14 };

	// High-pass filter 0°
	/*const int high_pass_filter_deg0[7][5] =
		{ {    0,    0,    0,   50,    0 },
		  {    0,    0,    0,  300,   50 },
		  {   -2,    0,   84, -925,    0 },
		  {  -12,   37,  231,   84,    0 },
		  {   -2,    0,   84,   37,    0 },
		  {    0,    0,    0,  -12,   -2 },
		  {    0,    0,    0,   -2,    0 } };*/
	const int high_pass_filter_deg0[16] = {
		50, 300, 50, -2, 84, -925, -12, 37, 231, 84, -2, 84, 37, -12, -2, -2 };

	// High-pass filter 120°
	/*const int high_pass_filter_deg120[7][5] =
		{ {    0,    0,    0,   -2,    0 },
		  {    0,    0,    0,  -12,   -2 },
		  {   -2,    0,   84,   37,    0 },
		  {  -12,   37,  231,   84,    0 },
		  {   -2,    0,   84, -925,    0 },
		  {    0,    0,    0,  300,   50 },
		  {    0,    0,    0,   50,    0 } };*/
	const int high_pass_filter_deg120[16] = {
		-2, -12, -2, -2, 84, 37, -12, 37, 231, 84, -2, 84, -925, 300, 50, 50 };

	// High-pass filter 240°
	/*const int high_pass_filter_deg240[7][5] =
		{ {    0,    0,    0,   -2,    0 },
		  {    0,    0,    0,  -12,   -2 },
		  {   50,    0,   84,   37,    0 },
		  {  300, -925,  231,   84,    0 },
		  {   50,    0,   84,   37,    0 },
		  {    0,    0,    0,  -12,   -2 },
		  {    0,    0,    0,   -2,    0 } };*/
	const int high_pass_filter_deg240[16] = {
		-2, -12, -2, 50, 84, 37, 300, -925, 231, 84, 50, 84, 37, -12, -2, -2 };


	RGB_Hexarray out;

	Hexarray_init(&out, order, 0);

	for(unsigned int p = 0; p < rgb_hexarray->size; p++) {
		if(!(p % 49)) {
			bool skip = true;

			for(unsigned int q = p; q < p + 49; q++) {
				if(rgb_hexarray->p[q][0] || rgb_hexarray->p[q][1] || rgb_hexarray->p[q][2]) {
					skip = false;
					break;
				}
			}

			if(skip) {
				p += 48;
				continue;
			}
		}


		fPoint2d ps_bak = getSpatial(Hexint_init(p, 0));

		for(unsigned int q = 0; q < 16; q++) {
			fPoint2d ps = { .x = ps_bak.x, .y = ps_bak.y };
			int      hp;

			// Offsets as provided by the original filters
			switch(q) {
				case  0 : ps.x += 2; ps.y += 3; break;
				case  1 : ps.x += 2; ps.y += 2; break;
				case  2 : ps.x += 3; ps.y += 2; break;
				case  3 : ps.x -= 2; ps.y += 1; break;
				case  4 :            ps.y += 1; break;
				case  5 : ps.x += 1; ps.y += 1; break;
				case  6 : ps.x -= 2;            break;
				case  7 : ps.x -= 1;            break;
				case  8 :                       break;
				case  9 : ps.x += 1;            break;
				case 10 : ps.x -= 3; ps.y -= 1; break;
				case 11 : ps.x -= 1; ps.y -= 1; break;
				case 12 :            ps.y -= 1; break;
				case 13 :            ps.y -= 2; break;
				case 14 : ps.x += 1; ps.y -= 2; break;
				case 15 : ps.x -= 1; ps.y -= 3; break;
				default :                       break;
			}

			if(ps.x >= pc_smn.x && ps.y >= pc_smn.y && ps.x <= pc_smx.x && ps.y <= pc_smx.y) {
				ps.x -= pc_smn.x;
				ps.y -= pc_smn.y;

				if(pc_spatials[(int)(ps.x)][(int)(ps.y)] < rgb_hexarray->size) {
					hp = pc_spatials[(int)(ps.x)][(int)(ps.y)];
				} else {
					hp = -1;
				}
			} else {
				hp = -1;
			}

			if(hp > -1) {
				if(!mode) {
					out.p[p][0] += rgb_hexarray->p[hp][0] * low_pass_filter[q];
					out.p[p][1] += rgb_hexarray->p[hp][1] * low_pass_filter[q];
					out.p[p][2] += rgb_hexarray->p[hp][2] * low_pass_filter[q];
				} else if(mode == 1) {
					out.p[p][0] += rgb_hexarray->p[hp][0] * high_pass_filter_deg0[q];
					out.p[p][1] += rgb_hexarray->p[hp][1] * high_pass_filter_deg0[q];
					out.p[p][2] += rgb_hexarray->p[hp][2] * high_pass_filter_deg0[q];
				} else if(mode == 2) {
					out.p[p][0] += rgb_hexarray->p[hp][0] * high_pass_filter_deg120[q];
					out.p[p][1] += rgb_hexarray->p[hp][1] * high_pass_filter_deg120[q];
					out.p[p][2] += rgb_hexarray->p[hp][2] * high_pass_filter_deg120[q];
				} else {
					out.p[p][0] += rgb_hexarray->p[hp][0] * high_pass_filter_deg240[q];
					out.p[p][1] += rgb_hexarray->p[hp][1] * high_pass_filter_deg240[q];
					out.p[p][2] += rgb_hexarray->p[hp][2] * high_pass_filter_deg240[q];
				}
			}
		}
	}

	for(unsigned int i = 0; i < out.size; i++) {
		out.p[i][0] = (unsigned int)roundf(out.p[i][0] / 1014);
		out.p[i][1] = (unsigned int)roundf(out.p[i][1] / 1014);
		out.p[i][2] = (unsigned int)roundf(out.p[i][2] / 1014);

		if(out.p[i][0] < 256) {
			rgb_hexarray->p[i][0] = out.p[i][0];
		} else {
			rgb_hexarray->p[i][0] = 255;
		}
		if(out.p[i][1] < 256) {
			rgb_hexarray->p[i][1] = out.p[i][1];
		} else {
			rgb_hexarray->p[i][1] = 255;
		}
		if(out.p[i][2] < 256) {
			rgb_hexarray->p[i][2] = out.p[i][2];
		} else {
			rgb_hexarray->p[i][2] = 255;
		}
	}

	Hexarray_free(&out, 0);
}

// Modi: Blurring- / Unblurring-filter
void filter_unblurring(RGB_Hexarray* rgb_hexarray, bool mode) {
	// Blurring filter
	/*const int blurring_filter[6][6] =
		{ {    0,    0,    1,    0,    0,    0 },
		  {    0,    1,    9,   10,    0,    0 },
		  {    0,    0,   90,   82,   18,    1 },
		  {    0,   82,  730,   90,    9,    1 },
		  {    0,    0,   81,   90,   18,    0 },
		  {    0,    9,    0,    0,    0,    0 } };*/
	const int blurring_filter[17] = {
		1, 1, 9, 10, 90, 82, 18, 1, 82, 730, 90, 9, 1, 81, 90, 18, 9 };

	// Unblurring filter
	/*const int unblurring_filter[6][6] =
		{ {    0,    0,   -1,    0,    0,    0 },
		  {    0,   -1,   10,    9,    0,    0 },
		  {    0,    0,  -90, -101,   20,   -1 },
		  {    0, -101,  999,  -90,   10,   -1 },
		  {    0,    0, -100,  -90,   20,    0 },
		  {    0,   10,    0,    0,    0,    0 } };*/
	const int unblurring_filter[17] = {
		-1, -1, 10, 9, -90, -101, 20, -1, -101, 999, -90, 10, -1, -100, -90, 20, 10 };


	RGB_Hexarray out;

	Hexarray_init(&out, (unsigned int)(logf(rgb_hexarray->size) / logf(7)), 0);

	for(unsigned int p = 0; p < rgb_hexarray->size; p++) {
		if(!(p % 49)) {
			bool skip = true;

			for(unsigned int q = p; q < p + 49; q++) {
				if(rgb_hexarray->p[q][0] || rgb_hexarray->p[q][1] || rgb_hexarray->p[q][2]) {
					skip = false;
					break;
				}
			}

			if(skip) {
				p += 48;
				continue;
			}
		}


		fPoint2d ps_bak = getSpatial(Hexint_init(p, 0));

		for(unsigned int q = 0; q < 17; q++) {
			fPoint2d ps = { .x = ps_bak.x, .y = ps_bak.y };
			int      hp;

			// Offsets as provided by the original filters
			switch(q) {
				case  0 : ps.x += 1; ps.y += 3; break;
				case  1 :            ps.y += 2; break;
				case  2 : ps.x += 1; ps.y += 2; break;
				case  3 : ps.x += 2; ps.y += 2; break;
				case  4 :            ps.y += 1; break;
				case  5 : ps.x += 1; ps.y += 1; break;
				case  6 : ps.x += 2; ps.y += 1; break;
				case  7 : ps.x += 3; ps.y += 1; break;
				case  8 : ps.x -= 1;            break;
				case  9 :                       break;
				case 10 : ps.x += 1;            break;
				case 11 : ps.x += 2;            break;
				case 12 : ps.x += 3;            break;
				case 13 : ps.x -= 1; ps.y -= 1; break;
				case 14 :            ps.y -= 1; break;
				case 15 : ps.x += 1; ps.y -= 1; break;
				case 16 : ps.x -= 2; ps.y -= 2; break;
				default :                       break;
			}

			if(ps.x >= pc_smn.x && ps.y >= pc_smn.y && ps.x <= pc_smx.x && ps.y <= pc_smx.y) {
				ps.x -= pc_smn.x;
				ps.y -= pc_smn.y;

				if(pc_spatials[(int)(ps.x)][(int)(ps.y)] < rgb_hexarray->size) {
					hp = pc_spatials[(int)(ps.x)][(int)(ps.y)];
				} else {
					hp = -1;
				}
			} else {
				hp = -1;
			}

			if(hp > -1) {
				if(!mode) {
					out.p[p][0] += rgb_hexarray->p[hp][0] * blurring_filter[q];
					out.p[p][1] += rgb_hexarray->p[hp][1] * blurring_filter[q];
					out.p[p][2] += rgb_hexarray->p[hp][2] * blurring_filter[q];
				} else {
					out.p[p][0] += rgb_hexarray->p[hp][0] * unblurring_filter[q];
					out.p[p][1] += rgb_hexarray->p[hp][1] * unblurring_filter[q];
					out.p[p][2] += rgb_hexarray->p[hp][2] * unblurring_filter[q];
				}
			}
		}
	}

	for(unsigned int i = 0; i < out.size; i++) {
		if(!mode) {		
			out.p[i][0] = (unsigned int)roundf(out.p[i][0] / 1322);
			out.p[i][1] = (unsigned int)roundf(out.p[i][1] / 1322);
			out.p[i][2] = (unsigned int)roundf(out.p[i][2] / 1322);
		} else {
			out.p[i][0] = (unsigned int)roundf(out.p[i][0] /  500);
			out.p[i][1] = (unsigned int)roundf(out.p[i][1] /  500);
			out.p[i][2] = (unsigned int)roundf(out.p[i][2] /  500);
		}

		if(out.p[i][0] < 256) {
			rgb_hexarray->p[i][0] = out.p[i][0];
		} else {
			rgb_hexarray->p[i][0] = 255;
		}
		if(out.p[i][1] < 256) {
			rgb_hexarray->p[i][1] = out.p[i][1];
		} else {
			rgb_hexarray->p[i][1] = 255;
		}
		if(out.p[i][2] < 256) {
			rgb_hexarray->p[i][2] = out.p[i][2];
		} else {
			rgb_hexarray->p[i][2] = 255;
		}
	}

	Hexarray_free(&out, 0);
}


// Lanczos-Filter (a = Größe des Trägers)

float L(float x, float a) {
	return fabs(x) <= a ? sinc(x) * sinc(x / a) : 0.0f;
}

void filter_lanczos(RGB_Hexarray* rgb_hexarray, float a, float intensity) {
	fRGB_Hexarray out;
	fPoint2d      ps, ps_bak;

	Hexarray_init(&out, (unsigned int)(logf(rgb_hexarray->size) / logf(7)), 1);

	for(unsigned int p = 0; p < rgb_hexarray->size; p++) {
		float out_n = 0.0f;

		ps_bak = getSpatial(Hexint_init(p, 0));

		for(unsigned int q = 0; q < 7; q++) {
			float        k;
			unsigned int hp;

			ps = ps_bak;

			switch(q) {
				case  0 :                             break;
				case  1 : ps.x += 1.0f;               break;
				case  2 : ps.x += 1.0f; ps.y += 1.0f; break;
				case  3 :               ps.y += 1.0f; break;
				case  4 : ps.x -= 1.0f;               break;
				case  5 : ps.x -= 1.0f; ps.y -= 1.0f; break;
				case  6 :               ps.y -= 1.0f; break;
				default :                             break;
			}

			if(ps.x >= pc_smn.x && ps.y >= pc_smn.y && ps.x <= pc_smx.x && ps.y <= pc_smx.y) {
				ps.x -= pc_smn.x;
				ps.y -= pc_smn.y;

				if(pc_spatials[(int)(ps.x)][(int)(ps.y)] || !p) {
					k  = q ? L(1.0f / intensity, a) : 1.0f;
					hp = pc_spatials[(int)(ps.x)][(int)(ps.y)];
				} else {
					continue;
				}
			} else {
				continue;
			}

			out.p[p][0] += k * rgb_hexarray->p[hp][0];
			out.p[p][1] += k * rgb_hexarray->p[hp][1];
			out.p[p][2] += k * rgb_hexarray->p[hp][2];
			out_n       += k;
		}

		out.p[p][0] /= out_n;
		out.p[p][1] /= out_n;
		out.p[p][2] /= out_n;
	}

	for(unsigned int i = 0; i < out.size; i++) {
		rgb_hexarray->p[i][0] = (unsigned int)roundf(out.p[i][0]);
		rgb_hexarray->p[i][1] = (unsigned int)roundf(out.p[i][1]);
		rgb_hexarray->p[i][2] = (unsigned int)roundf(out.p[i][2]);
	}

	Hexarray_free(&out, 1);
}

