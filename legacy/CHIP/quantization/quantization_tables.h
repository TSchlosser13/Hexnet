/******************************************************************************
 * quantization_tables.h: Hexagonale Quantisierungstabellen
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


#ifndef QUANTIZATION_TABLES_H
#define QUANTIZATION_TABLES_H


int qt2_JPEG[7]   = { 12, 14, 16, 13, 12, 16, 11 };
int qt2_custom[7] = {  6,  7,  8,  7,  6,  8,  6 };

int qt5_JPEG[9][9] =
	{ {  16,  11,  10,  16, 124, 140, 151, 161, 186 },
	  {  12,  12,  14,  19, 126, 158, 160, 155, 180 },
	  {  14,  13,  16,  24, 140, 157, 169, 156, 181 },
	  {  14,  17,  22,  29, 151, 187, 180, 162, 187 },
	  {  18,  22,  37,  56, 168, 109, 103, 177, 202 },
	  {  24,  35,  55,  64, 181, 104, 113, 192, 217 },
	  {  49,  64,  78,  87, 103, 121, 120, 101, 126 },
	  {  72,  92,  95,  98, 112, 100, 103, 199, 224 },
	  {  97, 117, 120, 123, 137, 125, 128, 224, 249 } };


#endif

