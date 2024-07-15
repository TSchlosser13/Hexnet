/******************************************************************************
 * filters.h: Hexagonal Filter Banks
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


#ifndef HFBS_H
#define HFBS_H


#include "../misc/defines.h"
#include "../misc/types.h"


// Modi: Low- (1) / High-pass filters (4: 0°, 120°, 240°, (0° + 120° + 240°) / 3)
void filter_4x16(RGB_Hexarray* rgb_hexarray, unsigned int mode);

// Modi: Blurring- / Unblurring-filter
void filter_unblurring(RGB_Hexarray* rgb_hexarray, bool mode);


// Lanczos-Filter (a = Größe des Trägers)

float L(float x, float a);

void filter_lanczos(RGB_Hexarray* rgb_hexarray, float a, float intensity);


#endif

