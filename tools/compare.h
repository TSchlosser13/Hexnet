/******************************************************************************
 * compare.h: Array Compare Methods
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


#ifndef COMPARE_H
#define COMPARE_H


/******************************************************************************
 * Includes
 ******************************************************************************/

#include "../misc/types.h"




/******************************************************************************
 * Defines
 ******************************************************************************/


/******************************************************************************
 * Compare methods
 ******************************************************************************/

#define COMPARE_AE     0
#define COMPARE_SE     1
#define COMPARE_MAE    2
#define COMPARE_MSE    3
#define COMPARE_RMSE   4
#define COMPARE_PSNR   5
#define COMPARE_SSIM   6
#define COMPARE_DSSIM  7
#define COMPARE_ERROR -1


/******************************************************************************
 * Calculate (D)SSIM stabilizers C1 and C2
 *
 * - SSIM_BITS        : bits per image channel
 * - SSIM_L           : range of pixel values
 * - SSIM_K1, SSIM_K2 : stabilization factors
 * - SSIM_C1, SSIM_C2 : function stabilizers
 ******************************************************************************/

#define SSIM_BITS 8
#define SSIM_L    ((1 << SSIM_BITS) - 1)
#define SSIM_K1   0.01f
#define SSIM_K2   0.03f
#define SSIM_C1   ((SSIM_K1 * SSIM_L) * (SSIM_K1 * SSIM_L))
#define SSIM_C2   ((SSIM_K2 * SSIM_L) * (SSIM_K2 * SSIM_L))




/******************************************************************************
 * Functions
 ******************************************************************************/

i32 get_compare_method(char* method);

u32          sum(u8* p, u32 size);
float       mean(u8* p, u32 size);
float   variance(u8* p, u32 size);
float covariance(u8* p1, u8* p2, u32 size);

float  _ssim(float mean_x, float mean_y, float variance_x, float variance_y, float covariance, float c1, float c2);
float _dssim(float mean_x, float mean_y, float variance_x, float variance_y, float covariance, float c1, float c2);

u32      ae(u8* p1, u8* p2, u32 size);
u32      se(u8* p1, u8* p2, u32 size);
float   mae(u8* p1, u8* p2, u32 size);
float   mse(u8* p1, u8* p2, u32 size);
float  rmse(u8* p1, u8* p2, u32 size);
float  psnr(u8* p1, u8* p2, u32 size);
float  ssim(u8* p1, u8* p2, u32 size);
float dssim(u8* p1, u8* p2, u32 size);

bool pixels_differ(u8* p1, u8* p2, u32 size);
i32  pixels_diff(u8* p1, u8* p2, u32 size, u32 method);

void set_ps_areas(u8** p1s, u8* p1, u8** p2s, u8* p2, u32* ps_end, u32 depth, double** areas, double area, u32* areas_end);
void realloc_ps_areas(u8** p1s, u8** p2s, u32 ps_end, u32* ps_size, double** areas, u32 areas_end, u32* areas_size);

double _compare_s2s(Array s1, Array s2, u32 method);
void    compare_s2s(Array s1, Array s2, u32 method);
double _compare_s2h(Array s, Hexarray h, u32 method);
void    compare_s2h(Array s, Hexarray h, u32 method);


#endif

