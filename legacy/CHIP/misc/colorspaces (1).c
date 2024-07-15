/******************************************************************************
 * colorspaces.c
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

#include "types.h"

#include "colorspaces.h"


void fRGB2YUV(fRGB_Hexarray* frgb_hexarray) {
	for(unsigned int i = 0; i < frgb_hexarray->size; i++) {
		const float R = frgb_hexarray->p[i][0];
		const float G = frgb_hexarray->p[i][1];
		const float B = frgb_hexarray->p[i][2];

		frgb_hexarray->p[i][0] =  0.299f * R + 0.587f * G + 0.114f * B;
		frgb_hexarray->p[i][1] = -0.147f * R - 0.289f * G + 0.436f * B;
		frgb_hexarray->p[i][2] =  0.615f * R - 0.515f * G - 0.1f   * B;
	}
}

void fYUV2RGB(fRGB_Hexarray* frgb_hexarray) {
	for(unsigned int i = 0; i < frgb_hexarray->size; i++) {
		const float Y = frgb_hexarray->p[i][0];
		const float U = frgb_hexarray->p[i][1];
		const float V = frgb_hexarray->p[i][2];

		frgb_hexarray->p[i][0] = Y              + 1.14f  * V;
		frgb_hexarray->p[i][1] = Y - 0.395f * U - 0.581f * V;
		frgb_hexarray->p[i][2] = Y + 2.032f * U;
	}
}


void fRGB2YCbCr(fRGB_Hexarray* frgb_hexarray) {
	for(unsigned int i = 0; i < frgb_hexarray->size; i++) {
		const float R = frgb_hexarray->p[i][0];
		const float G = frgb_hexarray->p[i][1];
		const float B = frgb_hexarray->p[i][2];

		frgb_hexarray->p[i][0] =  0.257f * R + 0.504f * G + 0.098f * B +  16;
		frgb_hexarray->p[i][1] = -0.148f * R - 0.291f * G + 0.439f * B + 128;
		frgb_hexarray->p[i][2] =  0.439f * R - 0.368f * G - 0.071f * B + 128;
	}
}

void fYCbCr2RGB(RGB_Hexarray* frgb_hexarray) {
	for(unsigned int i = 0; i < frgb_hexarray->size; i++) {
		const float Y  = 1.164f * (frgb_hexarray->p[i][0] -  16); // C = 1.164f * (...)
		const float Cb =           frgb_hexarray->p[i][1] - 128;
		const float Cr =           frgb_hexarray->p[i][2] - 128;

		frgb_hexarray->p[i][0] = Y               + 1.596f * Cr;
		frgb_hexarray->p[i][1] = Y - 0.392f * Cb - 0.813f * Cr;
		frgb_hexarray->p[i][2] = Y + 2.017f * Cb;
	}
}


void iRGB2YCbCr(RGB_Hexarray* rgb_hexarray) {
	for(unsigned int i = 0; i < rgb_hexarray->size; i++) {
		const int R = rgb_hexarray->p[i][0];
		const int G = rgb_hexarray->p[i][1];
		const int B = rgb_hexarray->p[i][2];

		rgb_hexarray->p[i][0] = (int)roundf(  0.257f * R + 0.504f * G + 0.098f * B ) +  16;
		rgb_hexarray->p[i][1] = (int)roundf( -0.148f * R - 0.291f * G + 0.439f * B ) + 128;
		rgb_hexarray->p[i][2] = (int)roundf(  0.439f * R - 0.368f * G - 0.071f * B ) + 128;
	}
}

void iYCbCr2RGB(RGB_Hexarray* rgb_hexarray) {
	for(unsigned int i = 0; i < rgb_hexarray->size; i++) {
		const int C = 1.164f * (rgb_hexarray->p[i][0] -  16);
		const int D =           rgb_hexarray->p[i][1] - 128;
		const int E =           rgb_hexarray->p[i][2] - 128;

		const int R = (const int)roundf( C              + 1.596f * E );
		const int G = (const int)roundf( C - 0.392f * D - 0.813f * E );
		const int B = (const int)roundf( C + 2.017f * D              );

		if(R < 0) {
			rgb_hexarray->p[i][0] =   0;
		} else if(R > 255) {
			rgb_hexarray->p[i][0] = 255;
		} else {
			rgb_hexarray->p[i][0] = R;
		}
		if(G < 0) {
			rgb_hexarray->p[i][1] =   0;
		} else if(G > 255) {
			rgb_hexarray->p[i][1] = 255;
		} else {
			rgb_hexarray->p[i][1] = G;
		}
		if(B < 0) {
			rgb_hexarray->p[i][2] =   0;
		} else if(B > 255) {
			rgb_hexarray->p[i][2] = 255;
		} else {
			rgb_hexarray->p[i][2] = B;
		}
	}
}

