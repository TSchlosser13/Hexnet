/******************************************************************************
 * quantization.c: Hexagonale Quantisierung
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

#include "../misc/types.h"

#include "quantization_tables.h"

#include "quantization.h"


// Modi: 2D- (81 / 61) / 1D-DCT (exp. / lin. / konst.)
void Quant_custom(void* DCTCoefs, unsigned int mode, unsigned int N, unsigned int qf) {
	float qf_scale;

	if(qf < 50) {
		qf_scale = floorf(5000 / qf) / 100;
	} else {
		qf_scale = (float)(200 - 2 * qf) / 100;
	}

	// DCTH
	if(mode < 2) {
		const unsigned int M = !mode ? 81 : 61;


		const unsigned int i2H5[9][9] =
			{ { 49, 33, 32, 53, 54, 61, 62, 63, 64 },
			  { 50, 34, 28, 31, 26, 25, 65, 66, 67 },
			  { 40, 39, 29, 30, 27, 21, 24, 68, 69 },
			  { 41, 35, 38,  5,  4, 22, 23, 57, 70 },
			  { 51, 36, 37,  6,  0,  3, 19, 18, 58 },
			  { 71, 52, 47, 46,  1,  2, 20, 14, 17 },
			  { 72, 73, 48, 42, 45, 12, 11, 15, 16 },
			  { 74, 75, 76, 43, 44, 13,  7, 10, 59 },
			  { 77, 78, 79, 80, 55, 56,  8,  9, 60 } };

		for(unsigned int h = 0; h < 9; h++) {
			for(unsigned int w = 0; w < 9; w++) {
				qt5_JPEG[w][h] = (int)roundf(qf_scale * qt5_JPEG[w][h]);

				if(qt5_JPEG[w][h] < 1)
					qt5_JPEG[w][h] = 1;
			}
		}


		// Quantisierung und Dequantisierung
		for(unsigned int i = 0; i < ((RGB_Array*)DCTCoefs)->x; i++) {
			int coefs[M][3];

			// Quantisierung
			for(unsigned int h = 0; h < 9; h++) {
				for(unsigned int w = 0; w < 9; w++) {
					if(mode || i2H5[w][h] < 61) {
						coefs[i2H5[w][h]][0] = (int)roundf(((RGB_Array*)DCTCoefs)->p[i][i2H5[w][h]][0] / qt5_JPEG[w][h]);
						coefs[i2H5[w][h]][1] = (int)roundf(((RGB_Array*)DCTCoefs)->p[i][i2H5[w][h]][1] / qt5_JPEG[w][h]);
						coefs[i2H5[w][h]][2] = (int)roundf(((RGB_Array*)DCTCoefs)->p[i][i2H5[w][h]][2] / qt5_JPEG[w][h]);
					}
				}
			}

			// Dequantisierung
			for(unsigned int h = 0; h < 9; h++) {
				for(unsigned int w = 0; w < 9; w++) {
					if(mode || i2H5[w][h] < 61) {
						((RGB_Array*)DCTCoefs)->p[i][i2H5[w][h]][0] = coefs[i2H5[w][h]][0] * qt5_JPEG[w][h];
						((RGB_Array*)DCTCoefs)->p[i][i2H5[w][h]][1] = coefs[i2H5[w][h]][1] * qt5_JPEG[w][h];
						((RGB_Array*)DCTCoefs)->p[i][i2H5[w][h]][2] = coefs[i2H5[w][h]][2] * qt5_JPEG[w][h];
					}
				}
			}
		}
	// 1D-DCTH
	} else {
		const unsigned int M = N == 5 ? 49 : 7;


		int qt[M];

		for(unsigned int i = 0; i < M; i++) {
			if(mode == 2) {
				qt[i] = (int)roundf(qf_scale * (16 + expf(4.431 / (M - 1)))); // exp. (e^x = 84)
			} else if(mode == 3) {
				qt[i] = (int)roundf(qf_scale * (16 + i * 84 / (M - 1)));      // lin.
			} else {
				qt[i] = qt5_JPEG[0][0];                                       // konst.
			}

			if(qt[i] < 1)
				qt[i] = 1;
		}


		// Quantisierung und Dequantisierung
		for(unsigned int i = 0; i < ((fRGB_Hexarray*)DCTCoefs)->size; i += M) {
			int coefs[M][3];

			// Quantisierung
			for(unsigned int p = 0; p < M; p++) {
				coefs[p][0] = (int)roundf(((fRGB_Hexarray*)DCTCoefs)->p[i + p][0] / qt[p]);
				coefs[p][1] = (int)roundf(((fRGB_Hexarray*)DCTCoefs)->p[i + p][1] / qt[p]);
				coefs[p][2] = (int)roundf(((fRGB_Hexarray*)DCTCoefs)->p[i + p][2] / qt[p]);
			}

			// Dequantisierung
			for(unsigned int p = 0; p < M; p++) {
				((fRGB_Hexarray*)DCTCoefs)->p[i + p][0] = (float)(coefs[p][0] * qt[p]);
				((fRGB_Hexarray*)DCTCoefs)->p[i + p][1] = (float)(coefs[p][1] * qt[p]);
				((fRGB_Hexarray*)DCTCoefs)->p[i + p][2] = (float)(coefs[p][2] * qt[p]);
			}
		}
	}
}

