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


/******************************************************************************
 * Includes
 ******************************************************************************/

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "compare.h"

#include "../misc/defines.h"
#include "../misc/types.h"




/******************************************************************************
 * Compare methods
 ******************************************************************************/


/******************************************************************************
 * Determine the compare method
 ******************************************************************************/

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
	} else if(!strcmp(method, "SSIM")) {
		return COMPARE_SSIM;
	} else if(!strcmp(method, "DSSIM")) {
		return COMPARE_DSSIM;
	} else {
		return COMPARE_ERROR;
	}
}


/******************************************************************************
 * sum, mean, variance, and covariance
 ******************************************************************************/

u32 sum(u8* p, u32 size) {
	u32 sum = 0;

	for(u32 i = 0; i < size; i++)
		sum += p[i];

	return sum;
}

float mean(u8* p, u32 size) {
	return (float)sum(p, size) / size;
}

float variance(u8* p, u32 size) {
	float variance = 0.0f;

	const float mean_p = mean(p, size);

	for(u32 i = 0; i < size; i++) {
		const float diff = p[i] - mean_p;

		variance += diff * diff;
	}

	return variance / size;
}

float covariance(u8* p1, u8* p2, u32 size) {
	float covariance = 0.0f;

	const float mean_p1 = mean(p1, size);
	const float mean_p2 = mean(p2, size);

	for(u32 i = 0; i < size; i++) {
		const float diff_p1 = p1[i] - mean_p1;
		const float diff_p2 = p2[i] - mean_p2;

		covariance += diff_p1 * diff_p2;
	}

	return covariance / size;
}


/******************************************************************************
 * ssim and dssim helper functions
 ******************************************************************************/

float _ssim(float mean_x, float mean_y, float variance_x, float variance_y, float covariance, float c1, float c2) {
	return ((2 * mean_x * mean_y + c1) * (2 * covariance + c2)) / ((mean_x * mean_x + mean_y * mean_y + c1) * (variance_x + variance_y + c2));
}

float _dssim(float mean_x, float mean_y, float variance_x, float variance_y, float covariance, float c1, float c2) {
	return (1 - _ssim(mean_x, mean_y, variance_x, variance_y, covariance, c1, c2)) / 2;
}


/******************************************************************************
 * ae, se, mae, mse, rmse, psnr, ssim, and dssim
 ******************************************************************************/

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

float ssim(u8* p1, u8* p2, u32 size) {
	const float c1 = SSIM_C1;
	const float c2 = SSIM_C2;

	float mean_p1        = 0.0f;
	float mean_p2        = 0.0f;
	float variance_p1    = 0.0f;
	float variance_p2    = 0.0f;
	float covariance_p12 = 0.0f;

	for(u32 i = 0; i < size; i++) {
		mean_p1 += p1[i];
		mean_p2 += p2[i];
	}

	mean_p1 /= size;
	mean_p2 /= size;

	for(u32 i = 0; i < size; i++) {
		const float diff_p1 = p1[i] - mean_p1;
		const float diff_p2 = p2[i] - mean_p2;

		variance_p1    += diff_p1 * diff_p1;
		variance_p2    += diff_p2 * diff_p2;
		covariance_p12 += diff_p1 * diff_p2;
	}

	variance_p1    /= size;
	variance_p2    /= size;
	covariance_p12 /= size;

	return _ssim(mean_p1, mean_p2, variance_p1, variance_p2, covariance_p12, c1, c2);
}

float dssim(u8* p1, u8* p2, u32 size) {
	return (1 - ssim(p1, p2, size)) / 2;
}




/******************************************************************************
 * Pixels differ and pixel difference
 ******************************************************************************/

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


/******************************************************************************
 * Compare helper functions
 ******************************************************************************/

#define col_changed(c1, c2)             ( (u32)(c1) != (u32)(c2) && (c2) != (u32)(c2) )
#define row_changed(r1, r2)             ( (u32)(r1) != (u32)(r2) && (r2) != (u32)(r2) )
#define col_row_changed(c1, c2, r1, r2) ( col_changed(c1, c2) && row_changed(r1, r2) )

void set_ps_areas(u8** p1s, u8* p1, u8** p2s, u8* p2, u32* ps_end, u32 depth, double** areas, double area, u32* areas_end) {
	for(u32 i = 0; i < depth; i++) {
		(*p1s)[*ps_end] = p1[i];
		(*p2s)[*ps_end] = p2[i];
		(*ps_end)++;
	}

	(*areas)[*areas_end] = area;
	(*areas_end)++;
}

void realloc_ps_areas(u8** p1s, u8** p2s, u32 ps_end, u32* ps_size, double** areas, u32 areas_end, u32* areas_size) {
	#define REALLOC_THRESHOLD 0.9f
	#define REALLOC_STEP_SIZE 1000000


	if(ps_end > REALLOC_THRESHOLD * *ps_size) {
		*ps_size += REALLOC_STEP_SIZE;
		*p1s      = (u8*)realloc(*p1s, *ps_size * sizeof(u8));
		*p2s      = (u8*)realloc(*p2s, *ps_size * sizeof(u8));
	}

	if(areas_end > REALLOC_THRESHOLD * *areas_size) {
		*areas_size += REALLOC_STEP_SIZE;
		*areas       = (double*)realloc(*areas, *areas_size * sizeof(double));
	}
}


/******************************************************************************
 * Compare Array to Array and Array to Hexarray
 ******************************************************************************/

double _compare_s2s(Array s1, Array s2, u32 method) {
	#define MALLOC_INIT_SIZE 1000000


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

	double area;
	i32    diff;
	double result = 0.0;

	// Collect pixel values and areas
	u32     ps_end     = 0;
	u32     areas_end  = 0;
	u32     ps_size    = MALLOC_INIT_SIZE;
	u32     areas_size = MALLOC_INIT_SIZE;
	u8*     p1s        =     (u8*)malloc(ps_size    * sizeof(u8));
	u8*     p2s        =     (u8*)malloc(ps_size    * sizeof(u8));
	double* areas      = (double*)malloc(areas_size * sizeof(double));


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


	const double ws_min = (double)width_min  / width_max;
	const double ws_max = (double)width_max  / width_min;
	const double hs_min = (double)height_min / height_max;
	const double hs_max = (double)height_max / height_min;


	for(u32 h_max = 0; h_max < height_max; h_max++) {
		const double h_min     = h_max * hs_min;
		const double h_min_max = (u32)(h_min + 1) * hs_max;

		for(u32 w_max = 0; w_max < width_max; w_max++) {
			const double w_min     = w_max * ws_min;
			const double w_min_max = (u32)(w_min + 1) * ws_max;


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


			if(method == COMPARE_SSIM || method == COMPARE_DSSIM)
				realloc_ps_areas(&p1s, &p2s, ps_end, &ps_size, &areas, areas_end, &areas_size);


			area    = (MIN(w_max + 1, w_min_max) - w_max) * (MIN(h_max + 1, h_min_max) - h_max);
			diff    = pixels_diff(s1.p + p1, s2.p + p2, depth, method);
			result += area * diff;

			if(method == COMPARE_SSIM || method == COMPARE_DSSIM)
				set_ps_areas(&p1s, s1.p + p1, &p2s, s2.p + p2, &ps_end, depth, &areas, area, &areas_end);

			if(col_changed(w_min, w_min + ws_min)) {
				area    = (w_max + 1 - w_min_max) * (MIN(h_max + 1, h_min_max) - h_max);
				diff    = pixels_diff(s1.p + p1 + p1_c, s2.p + p2 + p2_c, depth, method);
				result += area * diff;

				if(method == COMPARE_SSIM || method == COMPARE_DSSIM)
					set_ps_areas(&p1s, s1.p + p1 + p1_c, &p2s, s2.p + p2 + p2_c, &ps_end, depth, &areas, area, &areas_end);
			}

			if(row_changed(h_min, h_min + hs_min)) {
				area    = (MIN(w_max + 1, w_min_max) - w_max) * (h_max + 1 - h_min_max);
				diff    = pixels_diff(s1.p + p1 + p1_r, s2.p + p2 + p2_r, depth, method);
				result += area * diff;

				if(method == COMPARE_SSIM || method == COMPARE_DSSIM)
					set_ps_areas(&p1s, s1.p + p1 + p1_r, &p2s, s2.p + p2 + p2_r, &ps_end, depth, &areas, area, &areas_end);
			}

			if(col_row_changed(w_min, w_min + ws_min, h_min, h_min + hs_min)) {
				area    = (w_max + 1 - w_min_max) * (h_max + 1 - h_min_max);
				diff    = pixels_diff(s1.p + p1 + p1_cr, s2.p + p2 + p2_cr, depth, method);
				result += area * diff;

				if(method == COMPARE_SSIM || method == COMPARE_DSSIM)
					set_ps_areas(&p1s, s1.p + p1 + p1_cr, &p2s, s2.p + p2 + p2_cr, &ps_end, depth, &areas, area, &areas_end);
			}
		}
	}


	result *= MIN(1, (double)s1.width / s2.width) * MIN(1, (double)s1.height / s2.height); // adjust result in case of mismatched aspect ratios

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

	if(method == COMPARE_SSIM || method == COMPARE_DSSIM) {
		const float c1 = SSIM_C1;
		const float c2 = SSIM_C2;

		float mean_p1        = 0.0f;
		float mean_p2        = 0.0f;
		float variance_p1    = 0.0f;
		float variance_p2    = 0.0f;
		float covariance_p12 = 0.0f;

		for(u32 i = 0; i < ps_end; i++) {
			const double current_area = areas[i / depth];

			mean_p1 += current_area * p1s[i];
			mean_p2 += current_area * p2s[i];
		}

		mean_p1 /= size;
		mean_p2 /= size;

		for(u32 i = 0; i < ps_end; i++) {
			const double current_area = areas[i / depth];

			const float diff_p1 = current_area * (p1s[i] - mean_p1);
			const float diff_p2 = current_area * (p2s[i] - mean_p2);

			variance_p1    += diff_p1 * diff_p1;
			variance_p2    += diff_p2 * diff_p2;
			covariance_p12 += diff_p1 * diff_p2;
		}

		variance_p1    /= size;
		variance_p2    /= size;
		covariance_p12 /= size;

		if(method == COMPARE_SSIM) {
			result = _ssim(mean_p1, mean_p2, variance_p1, variance_p2, covariance_p12, c1, c2);
		} else {
			result = _dssim(mean_p1, mean_p2, variance_p1, variance_p2, covariance_p12, c1, c2);
		}
	}


	free(p1s);
	free(p2s);
	free(areas);


	return result;
}

void compare_s2s(Array s1, Array s2, u32 method) {
	printf("[compare_s2s] result = %f\n", _compare_s2s(s1, s2, method));
}

double _compare_s2h(Array s, Hexarray h, u32 method) {
	#define MALLOC_INIT_SIZE 1000000


	const u32 depth = s.depth;
	const u32 size  = s.size;

	double area;
	i32    diff;
	double result = 0.0;

	// Collect pixel values and areas
	u32     ps_end     = 0;
	u32     areas_end  = 0;
	u32     ps_size    = MALLOC_INIT_SIZE;
	u32     areas_size = MALLOC_INIT_SIZE;
	u8*     p1s        =     (u8*)malloc(ps_size    * sizeof(u8));
	u8*     p2s        =     (u8*)malloc(ps_size    * sizeof(u8));
	double* areas      = (double*)malloc(areas_size * sizeof(double));


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


			if(method == COMPARE_SSIM || method == COMPARE_DSSIM)
				realloc_ps_areas(&p1s, &p2s, ps_end, &ps_size, &areas, areas_end, &areas_size);


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

						if(method == COMPARE_SSIM || method == COMPARE_DSSIM)
							set_ps_areas(&p1s, s.p + ps, &p2s, h.p + phc, &ps_end, depth, &areas, area, &areas_end);
					}

					area    = wmr_min * hmr_min - area;
					diff    = pixels_diff(s.p + ps, h.p + ph, depth, method);
					result += area * diff;

					if(method == COMPARE_SSIM || method == COMPARE_DSSIM)
						set_ps_areas(&p1s, s.p + ps, &p2s, h.p + ph, &ps_end, depth, &areas, area, &areas_end);
				} else {
					if(phc >= 0) {
						area    = wmr_min * hmr_min;
						diff    = pixels_diff(s.p + ps, h.p + phc, depth, method);
						result += area * diff;

						if(method == COMPARE_SSIM || method == COMPARE_DSSIM)
							set_ps_areas(&p1s, s.p + ps, &p2s, h.p + phc, &ps_end, depth, &areas, area, &areas_end);
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

						if(method == COMPARE_SSIM || method == COMPARE_DSSIM)
							set_ps_areas(&p1s, s.p + ps, &p2s, h.p + phc, &ps_end, depth, &areas, area, &areas_end);
					}

					area    = wmr_min * hmr_min - area;
					diff    = pixels_diff(s.p + ps, h.p + ph, depth, method);
					result += area * diff;

					if(method == COMPARE_SSIM || method == COMPARE_DSSIM)
						set_ps_areas(&p1s, s.p + ps, &p2s, h.p + ph, &ps_end, depth, &areas, area, &areas_end);
				} else {
					if(phc >= 0 && phc < h.size) {
						area    = wmr_min * hmr_min;
						diff    = pixels_diff(s.p + ps, h.p + phc, depth, method);
						result += area * diff;

						if(method == COMPARE_SSIM || method == COMPARE_DSSIM)
							set_ps_areas(&p1s, s.p + ps, &p2s, h.p + phc, &ps_end, depth, &areas, area, &areas_end);
					}
				}
			} else {
				const u32 ps = depth * (hu  * s.width + wu);
				const u32 ph = depth * (hhu * h.width + whu);

				area    = wmr_min * hmr_min;
				diff    = pixels_diff(s.p + ps, h.p + ph, depth, method);
				result += area * diff;

				if(method == COMPARE_SSIM || method == COMPARE_DSSIM)
					set_ps_areas(&p1s, s.p + ps, &p2s, h.p + ph, &ps_end, depth, &areas, area, &areas_end);
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

	if(method == COMPARE_SSIM || method == COMPARE_DSSIM) {
		const float c1 = SSIM_C1;
		const float c2 = SSIM_C2;

		float mean_p1        = 0.0f;
		float mean_p2        = 0.0f;
		float variance_p1    = 0.0f;
		float variance_p2    = 0.0f;
		float covariance_p12 = 0.0f;

		for(u32 i = 0; i < ps_end; i++) {
			const double current_area = areas[i / depth];

			mean_p1 += current_area * p1s[i];
			mean_p2 += current_area * p2s[i];
		}

		mean_p1 /= size;
		mean_p2 /= size;

		for(u32 i = 0; i < ps_end; i++) {
			const double current_area = areas[i / depth];

			const float diff_p1 = current_area * (p1s[i] - mean_p1);
			const float diff_p2 = current_area * (p2s[i] - mean_p2);

			variance_p1    += diff_p1 * diff_p1;
			variance_p2    += diff_p2 * diff_p2;
			covariance_p12 += diff_p1 * diff_p2;
		}

		variance_p1    /= size;
		variance_p2    /= size;
		covariance_p12 /= size;

		if(method == COMPARE_SSIM) {
			result = _ssim(mean_p1, mean_p2, variance_p1, variance_p2, covariance_p12, c1, c2);
		} else {
			result = _dssim(mean_p1, mean_p2, variance_p1, variance_p2, covariance_p12, c1, c2);
		}
	}


	free(p1s);
	free(p2s);
	free(areas);


	return result;
}

void compare_s2h(Array s, Hexarray h, u32 method) {
	printf("[compare_s2h] result = %f\n", _compare_s2h(s, h, method));
}


