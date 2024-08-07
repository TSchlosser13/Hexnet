/******************************************************************************
 * Hexsamp.h: Hexagonale Bildtransformationen
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


#ifndef HEXSAMP_H
#define HEXSAMP_H


#include "../misc/types.h"


float sinc(float x);

// Kernel (Modi: BL / BC / Lanczos / B-Splines (B_3))
float kernel(float x, float y, unsigned int technique);


// Sq -> HIP
void hipsampleColour(RGB_Array rgb_array, RGB_Hexarray* rgb_hexarray,
 unsigned int order, float scale, unsigned int technique);


// HIP -> sq

void* resample(void* args_p);

void sqsampleColour(RGB_Hexarray rgb_hexarray, RGB_Array* rgb_array,
 float radius, float scale, unsigned int technique, unsigned int threads);


#endif

