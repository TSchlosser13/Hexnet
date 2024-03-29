/******************************************************************************
 * Hexarray.h: Operations on Hexagonal Arrays
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


#ifndef HEXARRAY_H
#define HEXARRAY_H


/******************************************************************************
 * Includes
 ******************************************************************************/

#include "../misc/types.h"


/******************************************************************************
 * Functions
 ******************************************************************************/

void Hexarray_init(Hexarray* hexarray, u32 width, u32 height, u32 depth, float rad_o);
void Hexarray_init_from_Array(Hexarray* hexarray, Array array, float rad_o);
void Hexarray_init_from_Hexarray(Hexarray* h1, Hexarray h2, float rad_o);
void Hexarray_free(Hexarray* hexarray);

void Hexarray_print_info(Hexarray hexarray, char* title);

void file_to_Hexarray(char* filename, Hexarray* hexarray);
void Hexarray_to_file(Hexarray hexarray, char* filename);


#endif

