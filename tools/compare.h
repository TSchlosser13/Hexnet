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


#include "../misc/types.h"


#define COMPARE_AE     0
#define COMPARE_SE     1
#define COMPARE_MAE    2
#define COMPARE_MSE    3
#define COMPARE_RMSE   4
#define COMPARE_PSNR   5
#define COMPARE_ERROR -1


i32 get_compare_method(char* method);

u32     ae(u8* p1, u8* p2, u32 size);
u32     se(u8* p1, u8* p2, u32 size);
float  mae(u8* p1, u8* p2, u32 size);
float  mse(u8* p1, u8* p2, u32 size);
float rmse(u8* p1, u8* p2, u32 size);
float psnr(u8* p1, u8* p2, u32 size);

bool pixels_differ(u8* p1, u8* p2, u32 size);
i32  pixels_diff(u8* p1, u8* p2, u32 size, u32 method);

double _compare_s2s(Array s1, Array s2, u32 method);
void    compare_s2s(Array s1, Array s2, u32 method);
double _compare_s2h(Array s, Hexarray h, u32 method);
void    compare_s2h(Array s, Hexarray h, u32 method);


#endif

