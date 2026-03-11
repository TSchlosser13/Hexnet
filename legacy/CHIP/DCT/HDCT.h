/******************************************************************************
 * HDCT.h: Hexagonale DCT
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


#ifndef HDCT_H
#define HDCT_H


#include "../misc/defines.h"
#include "../misc/types.h"


// Hexagonale DCT HDCT (Makroblockgröße = <N=(2|5)>)

void HDCT_init(float*** psiCos_table, unsigned int N);
void HDCT_free(float*** psiCos_table, unsigned int N);

void  HDCT_N2(  RGB_Hexarray  rgb_hexarray,     fRGB_Hexarray* frgb_HDCTHexarray, float** psiCos_table );
void IHDCT_N2( fRGB_Hexarray frgb_HDCTHexarray,  RGB_Hexarray*  rgb_hexarray,     float** psiCos_table );
void  HDCT_N5(  RGB_Hexarray  rgb_hexarray,      RGB_Array*     rgb_HDCTArray,    float** psiCos_table );
void IHDCT_N5(  RGB_Array     rgb_HDCTArray,     RGB_Hexarray*  rgb_hexarray,     float** psiCos_table );

// Makroblockgröße = <N=(>1)> (TODO?)
void HDCT2_init(float***** psiCos_table, unsigned int N);
void HDCT2_free(float***** psiCos_table, unsigned int N);


// Quadratische DCT DCTH (Makroblockgröße = <N=(2|5)>)

float c(unsigned int x);
float cosu(unsigned int i, unsigned int u, unsigned int N);
float cosv(unsigned int j, unsigned int v, unsigned int M);

void DCTH_init(float*** psiCos_table, unsigned int N);
void DCTH_free(float*** psiCos_table, bool mode, unsigned int N);

void  DCTH(RGB_Hexarray rgb_hexarray,  RGB_Array*    rgb_HDCTArray, float** psiCos_table, unsigned int N);
void IDCTH(RGB_Array    rgb_HDCTArray, RGB_Hexarray* rgb_hexarray,  float** psiCos_table, unsigned int N);


// 1D-DCTH (Modi: Spiral Structure Addressing SSA / SSA Log Space, Makroblockgröße = <N=(2|5)>)

void DCTH2_init(float*** psiCos_table, unsigned int N);
#define DCTH2_free DCTH_free

void  DCTH2(  RGB_Hexarray  rgb_hexarray,     fRGB_Hexarray* frgb_HDCTHexarray, float** psiCos_table, bool mode, unsigned int N );
void IDCTH2( fRGB_Hexarray frgb_HDCTHexarray,  RGB_Hexarray*  rgb_hexarray,     float** psiCos_table, bool mode, unsigned int N );


#endif

