/******************************************************************************
 * compare.c: Array Compare Methods
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
#include <stdio.h>

#include "compare.h"

#include "../misc/defines.h"
#include "../misc/types.h"


i32 get_compare_method(char* method) {
	if(!strcmp(method, "AE")) {
		return COMPARE_AE;
	} else if(!strcmp(method, "SE")) {
		return COMPARE_SE;
	} else if(!strcmp(method, "MAE")) {
		return COMPARE_MAE;
	} else if(!strcmp(method, "MSE")) {
		return COMPARE_MSE;
	} else if(!strcmp(method, "RMSE")) {
		return COMPARE_RMSE;
	} else if(!strcmp(method, "PSNR")) {
		return COMPARE_PSNR;
	} else {
		return COMPARE_ERROR;
	}
}


u32 ae(u8* p1, u8* p2, u32 size) {
	u32 ae = 0;

	for(u32 i = 0; i < size; i++)
		ae += abs(p1[i] - p2[i]);

	return ae;
}

u32 se(u8* p1, u8* p2, u32 size) {
	u32 se = 0;

	for(u32 i = 0; i < size; i++) {
		const i32 diff = p1[i] - p2[i];

		se += diff * diff;
	}

	return se;
}

float mae(u8* p1, u8* p2, u32 size) {
	return (float)ae(p1, p2, size) / size;
}

float mse(u8* p1, u8* p2, u32 size) {
	return (float)se(p1, p2, size) / size;
}

float rmse(u8* p1, u8* p2, u32 size) {
	return sqrtf(mse(p1, p2, size));
}

float psnr(u8* p1, u8* p2, u32 size) {
	return 10 * logf(65025 / mse(p1, p2, size));
}


#define col_changed(c1, c2)             ( (u32)(c1) != (u32)(c2) && (c2) != (u32)(c2) )
#define row_changed(r1, r2)             ( (u32)(r1) != (u32)(r2) && (r2) != (u32)(r2) )
#define col_row_changed(c1, c2, r1, r2) ( col_changed(c1, c2) && row_changed(r1, r2) )

bool pixels_differ(u8* p1, u8* p2, u32 size) {
	for(u32 i = 0; i < size; i++) {
		if(p1[i] != p2[i])
			return true;
	}

	return false;
}

i32 pixels_diff(u8* p1, u8* p2, u32 size, u32 method) {
	if(method == COMPARE_AE || method == COMPARE_MAE) {
		return ae(p1, p2, size);
	} else if( method == COMPARE_SE   ||
	           method == COMPARE_MSE  ||
	           method == COMPARE_RMSE ||
	           method == COMPARE_PSNR ) {
		return se(p1, p2, size);
	} else {
		return COMPARE_ERROR;
	}
}


double _compare_s2s(Array s1, Array s2, u32 method) {
	const u32 depth = s1.depth;
	const u32 size  = s1.size;

	u32 width_min;
	u32 width_max;
	u32 height_min;
	u32 height_max;
	u32 p1;
	u32 p2;
	u32 p1_c;
	u32 p2_c;
	u32 p1_r;
	u32 p2_r;
	u32 p1_cr;
	u32 p2_cr;

	float  area;
	i32    diff;
	double result = 0.0;


	if(s1.width > s2.width) {
		width_min = s2.width;
		width_max = s1.width;
		p1_c      = 0;
		p2_c      = depth;
	} else {
		width_min = s1.width;
		width_max = s2.width;
		p1_c      = depth;
		p2_c      = 0;
	}

	p1_cr = p1_c;
	p2_cr = p2_c;

	if(s1.height > s2.height) {
		height_min = s2.height;
		height_max = s1.height;
		p1_r       = 0;
		p2_r       = depth * s2.width;
	} else {
		height_min = s1.height;
		height_max = s2.height;
		p1_r       = depth * s1.width;
		p2_r       = 0;
	}

	p1_cr += p1_r;
	p2_cr += p2_r;


	const float ws_min = (float)width_min  / width_max;
	const float ws_max = (float)width_max  / width_min;
	const float hs_min = (float)height_min / height_max;
	const float hs_max = (float)height_max / height_min;


	for(u32 h_max = 0; h_max < height_max; h_max++) {
		const float h_min     = h_max * hs_min;
		const float h_min_max = (u32)(h_min + 1) * hs_max;

		for(u32 w_max = 0; w_max < width_max; w_max++) {
			const float w_min     = w_max * ws_min;
			const float w_min_max = (u32)(w_min + 1) * ws_max;


			if(s1.width > s2.width) {
				p1 = w_max;
				p2 = (u32)w_min;
			} else {
				p1 = (u32)w_min;
				p2 = w_max;
			}

			if(s1.height > s2.height) {
				p1 += h_max * s1.width;
				p2 += (u32)h_min * s2.width;
			} else {
				p1 += (u32)h_min * s1.width;
				p2 += h_max * s2.width;
			}

			p1 *= depth;
			p2 *= depth;


			area    = (MIN(w_max + 1, w_min_max) - w_max) * (MIN(h_max + 1, h_min_max) - h_max);
			diff    = pixels_diff(s1.p + p1, s2.p + p2, depth, method);
			result += area * diff;

			if(col_changed(w_min, w_min + ws_min)) {
				area    = (w_max + 1 - w_min_max) * (MIN(h_max + 1, h_min_max) - h_max);
				diff    = pixels_diff(s1.p + p1 + p1_c, s2.p + p2 + p2_c, depth, method);
				result += area * diff;
			}

			if(row_changed(h_min, h_min + hs_min)) {
				area    = (MIN(w_max + 1, w_min_max) - w_max) * (h_max + 1 - h_min_max);
				diff    = pixels_diff(s1.p + p1 + p1_r, s2.p + p2 + p2_r, depth, method);
				result += area * diff;
			}

			if(col_row_changed(w_min, w_min + ws_min, h_min, h_min + hs_min)) {
				area    = (w_max + 1 - w_min_max) * (h_max + 1 - h_min_max);
				diff    = pixels_diff(s1.p + p1 + p1_cr, s2.p + p2 + p2_cr, depth, method);
				result += area * diff;
			}
		}
	}


	result *= MIN(1, (float)s1.width / s2.width) * MIN(1, (float)s1.height / s2.height);

	if( method == COMPARE_MAE  ||
	    method == COMPARE_MSE  ||
	    method == COMPARE_RMSE ||
	    method == COMPARE_PSNR ) {
		result /= size;
	}

	if(method == COMPARE_RMSE) {
		result = sqrtf(result);
	} else if(method == COMPARE_PSNR) {
		result = 10 * logf(65025 / result);
	}


	return result;
}

void compare_s2s(Array s1, Array s2, u32 method) {
	printf("[compare_s2s] result = %f\n", _compare_s2s(s1, s2, method));
}

double _compare_s2h(Array s, Hexarray h, u32 method) {
	const u32 depth = s.depth;
	const u32 size  = s.size;

	double area;
	i32    diff;
	double result = 0.0;


	const double wb = (h.width_hex  - s.width_sq)  / 2;
	const double hb = (h.height_hex - s.height_sq) / 2;

	const double ws_s = s.len;
	const double ws_h = h.rad_i;
	const double hs_s = s.len;
	const double hs_h = h.rad_o / 2;
	const double sr_h = hs_h / ws_h;


	double wi = 0.0;
	double hi = 0.0;


	while(hi < s.height_sq) {
		const double hh   = hb + hi;
		const double hhm  = fmod(hh, hs_h);
		const double hhmr = hs_h - hhm;

		const double hm      = fmod(hi, hs_s);
		const double hmr     = hs_s - hm;
		const double hmr_min = MIN(hmr, hhmr);

		const double hd = fmod(hh, h.dist_h);
		const double hn = MAX(hd + hmr_min, nextafter(hd, hd + 1));

		const u32 hu  = (u32)(hi / hs_s);
		const u32 hhu = (u32)(hh / h.dist_h);


		while(wi < s.width_sq) {
			const double wh   = !(hhu % 2) ? wb + wi : wb - h.rad_i + wi;
			const double whm  = fmod(wh, ws_h);
			const double whmr = ws_h - whm;

			const double wm      = fmod(wi, ws_s);
			const double wmr     = ws_s - wm;
			const double wmr_min = MIN(wmr, whmr);

			const double wd = fmod(wh, h.dist_w);
			const double wn = MAX(wd + wmr_min, nextafter(wd, wd + 1));

			const u32 wu  = (u32)(wi / ws_s);
			const u32 whu = (u32)(wh / h.dist_w);


			if(hd < hs_h - sr_h * wd) {
				const double w_min = MAX(wd, (hs_h - hn) / sr_h);
				const double w_max = MIN((hs_h - hd) / sr_h, wn);
				const double h_min = hs_h - sr_h * w_max;
				const double h_max = hs_h - sr_h * w_min;

				const u32 whuc = hhu % 2 ? whu : whu - 1;
				const u32 hhuc = hhu - 1;
				const u32 ps   = depth * (hu   * s.width + wu);
				const u32 ph   = depth * (hhu  * h.width + whu);
				const i32 phc  = depth * (hhuc * h.width + whuc);


				if(h_min < hn) {
					area = wmr_min * (h_min - hd) + ((w_max - w_min) * (h_max - h_min)) / 2;

					if(phc >= 0) {
						diff    = pixels_diff(s.p + ps, h.p + phc, depth, method);
						result += area * diff;
					}

					area    = wmr_min * hmr_min - area;
					diff    = pixels_diff(s.p + ps, h.p + ph, depth, method);
					result += area * diff;
				} else {
					if(phc >= 0) {
						area    = wmr_min * hmr_min;
						diff    = pixels_diff(s.p + ps, h.p + phc, depth, method);
						result += area * diff;
					}
				}
			} else if(hd < hs_h - sr_h * (h.dist_w - wn)) {
				const double w_min = MAX(wd, (hd + hs_h) / sr_h);
				const double w_max = MIN((hn + hs_h) / sr_h, wn);
				const double h_min = hs_h - sr_h * (h.dist_w - w_min);
				const double h_max = hs_h - sr_h * (h.dist_w - w_max);

				const u32 whuc = hhu % 2 ? whu + 1 : whu;
				const u32 hhuc = hhu - 1;
				const u32 ps   = depth * (hu   * s.width + wu);
				const u32 ph   = depth * (hhu  * h.width + whu);
				const i32 phc  = depth * (hhuc * h.width + whuc);


				if(h_min < hn) {
					area = wmr_min * (h_min - hd) + ((w_max - w_min) * (h_max - h_min)) / 2;

					if(phc >= 0 && phc < h.size) {
						diff    = pixels_diff(s.p + ps, h.p + phc, depth, method);
						result += area * diff;
					}

					area    = wmr_min * hmr_min - area;
					diff    = pixels_diff(s.p + ps, h.p + ph, depth, method);
					result += area * diff;
				} else {
					if(phc >= 0 && phc < h.size) {
						area    = wmr_min * hmr_min;
						diff    = pixels_diff(s.p + ps, h.p + phc, depth, method);
						result += area * diff;
					}
				}
			} else {
				const u32 ps = depth * (hu  * s.width + wu);
				const u32 ph = depth * (hhu * h.width + whu);

				area    = wmr_min * hmr_min;
				diff    = pixels_diff(s.p + ps, h.p + ph, depth, method);
				result += area * diff;
			}


			wi = MAX(wi + wmr_min, nextafter(wi + wmr_min, wi + wmr_min + 1));
		}


		wi = 0.0;
		hi = MAX(hi + hmr_min, nextafter(hi + hmr_min, hi + hmr_min + 1));
	}


	if( method == COMPARE_MAE  ||
	    method == COMPARE_MSE  ||
	    method == COMPARE_RMSE ||
	    method == COMPARE_PSNR ) {
		result /= size;
	}

	if(method == COMPARE_RMSE) {
		result = sqrtf(result);
	} else if(method == COMPARE_PSNR) {
		result = 10 * logf(65025 / result);
	}


	return result;
}

void compare_s2h(Array s, Hexarray h, u32 method) {
	printf("[compare_s2h] result = %f\n", _compare_s2h(s, h, method));
}

