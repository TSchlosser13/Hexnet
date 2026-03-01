/******************************************************************************
 * precalculations.c: Vorberechnungen
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


#ifndef PRECALCS_H
#define PRECALCS_H


#include "types.h"


// TODO: Notation

float**  pc_reals;
iPoint2d pc_rmn; // pc_reals_min
iPoint2d pc_rmx; // pc_reals_max

unsigned int** pc_spatials;
uPoint2d       pc_spatials_size;
iPoint2d       pc_smn; // pc_spatials_min
iPoint2d       pc_smx; // pc_spatials_max

unsigned int** pc_nearest;

unsigned int** pc_adds;
uPoint3d*      pc_adds_mul;
unsigned int   pc_adds_mul_size;


void precalcs_init(unsigned int order, float scale, float radius);
void precalcs_free();


#endif

