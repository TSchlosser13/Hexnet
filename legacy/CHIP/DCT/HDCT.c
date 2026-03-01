/******************************************************************************
 * HDCT.c: Hexagonale DCT
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


#include <stdio.h>
#include <stdlib.h>

#include <math.h>

#include "../misc/Array.h"
#include "../misc/defines.h"
#include "../misc/types.h"

#include "../core/Hexint.h"
#include "../core/Hexarray.h"

#include "HDCT.h"


const unsigned int i2LogSpace2[7] = { 0, 1, 3, 2, 5, 6, 4 };

const unsigned int i2LogSpace5[49] =
	{  0,  1, 45, 44, 33, 34, 39,
	  35, 36, 24, 23, 19, 20, 11,
	   7,  8, 31, 30,  5,  6, 46,
	  42, 43, 17, 16, 40, 41, 25,
	  21, 22,  3,  2, 12, 13, 32,
	  28, 29, 38, 37, 47, 48, 18,
	  14, 15, 10,  9, 26, 27,  4 };




// Hexagonale DCT HDCT (Makroblockgröße = <N=(2|5)>)


void HDCT_init(float*** psiCos_table, unsigned int N) {
	const unsigned int wh = 2 * N - 1;                 // max. Durchmesser
	const unsigned int M  = 3 * pow(N, 2) - 3 * N + 1; // und #Pixel


	const int i2H2[3][3] =
		{ {  5,  4, -1 },
		  {  6,  0,  3 },
		  { -1,  1,  2 } };

	const int i2H5[9][9] =
		{ { 49, 33, 32, 53, 54, -1, -1, -1, -1 },
		  { 50, 34, 28, 31, 26, 25, -1, -1, -1 },
		  { 40, 39, 29, 30, 27, 21, 24, -1, -1 },
		  { 41, 35, 38,  5,  4, 22, 23, 57, -1 },
		  { 51, 36, 37,  6,  0,  3, 19, 18, 58 },
		  { -1, 52, 47, 46,  1,  2, 20, 14, 17 },
		  { -1, -1, 48, 42, 45, 12, 11, 15, 16 },
		  { -1, -1, -1, 43, 44, 13,  7, 10, 59 },
		  { -1, -1, -1, -1, 55, 56,  8,  9, 60 } };


	*psiCos_table = (float**)malloc(M * sizeof(float*));
	for(unsigned int i = 0; i < M; i++) {
		(*psiCos_table)[i] = (float*)calloc(M, sizeof(float));
	}

	for(unsigned int u = 0; u < wh; u++) {
		for(unsigned int v = 0; v < wh; v++) {
			if(abs(u - v) < N) {
				for(unsigned int i = 0; i < wh; i++) {
					for(unsigned int j = 0; j < wh; j++) {
						if(abs(i - j) < N) {
							if(N == 5) {
								(*psiCos_table)[i2H5[u][v]][i2H5[i][j]] = cos(M_PI / M * \
									((i - (float)N / 2 + 1) * ((4 * (float)N - 2) * u - (N - 1) * v) - \
									(j + 0.5f) * (2 * (float)N * u - (2 * N - 1) * v)));
							} else {
								(*psiCos_table)[i2H2[u][v]][i2H2[i][j]] = cos(M_PI / M * \
									((i - (float)N / 2 + 1) * ((4 * (float)N - 2) * u - (N - 1) * v) - \
									(j + 0.5f) * (2 * (float)N * u - (2 * N - 1) * v)));
							}
						}
					}
				}
			}
		}
	}
}

void HDCT_free(float*** psiCos_table, unsigned int N) {
	const unsigned int M = 3 * pow(N, 2) - 3 * N + 1; // #Pixel

	for(unsigned int i = 0; i < M; i++) {
		free((*psiCos_table)[i]);
	}
	free(*psiCos_table);
}


void HDCT_N2(RGB_Hexarray rgb_hexarray, fRGB_Hexarray* frgb_HDCTHexarray, float** psiCos_table) {
	if(rgb_hexarray.size < 7) {
		printf("\n\nError - HDCT_N2: rgb_hexarray.size (%u) < 7", rgb_hexarray.size);
		exit(EXIT_FAILURE);
	}


	Hexarray_init(frgb_HDCTHexarray, (unsigned int)(logf(rgb_hexarray.size) / logf(7)), 1);

	for(unsigned int p = 0; p < frgb_HDCTHexarray->size; p++) {
		const unsigned int uv = p % 7;

		for(unsigned int ij = 0; ij < 7; ij++) {
			const unsigned int rgb = p - uv + ij;

			frgb_HDCTHexarray->p[p][0] += rgb_hexarray.p[rgb][0] * psiCos_table[uv][ij];
			frgb_HDCTHexarray->p[p][1] += rgb_hexarray.p[rgb][1] * psiCos_table[uv][ij];
			frgb_HDCTHexarray->p[p][2] += rgb_hexarray.p[rgb][2] * psiCos_table[uv][ij];
		}

		if(uv == 5) {
			frgb_HDCTHexarray->p[p][0] *= M_SQRT1_2;
			frgb_HDCTHexarray->p[p][1] *= M_SQRT1_2;
			frgb_HDCTHexarray->p[p][2] *= M_SQRT1_2;
		}
	}
}

void IHDCT_N2(fRGB_Hexarray frgb_HDCTHexarray, RGB_Hexarray* rgb_hexarray, float** psiCos_table) {
	if(rgb_hexarray->size < 7) {
		printf("\n\nError - IHDCT_N2: rgb_hexarray->size (%u) < 7", rgb_hexarray->size);
		exit(EXIT_FAILURE);
	}


	fRGB_Hexarray tempx;

	Hexarray_init(&tempx, (unsigned int)(logf(rgb_hexarray->size) / logf(7)), 1);

	for(unsigned int p = 0; p < tempx.size; p++) {
		const unsigned int ij = p % 7;

		for(unsigned int uv = 0; uv < 7; uv++) {
			const unsigned int rgb = p - ij + uv;

			if(uv != 5) {
				tempx.p[p][0] += frgb_HDCTHexarray.p[rgb][0] * psiCos_table[uv][ij];
				tempx.p[p][1] += frgb_HDCTHexarray.p[rgb][1] * psiCos_table[uv][ij];
				tempx.p[p][2] += frgb_HDCTHexarray.p[rgb][2] * psiCos_table[uv][ij];
			} else {
				tempx.p[p][0] += M_SQRT1_2 * frgb_HDCTHexarray.p[rgb][0] * psiCos_table[uv][ij];
				tempx.p[p][1] += M_SQRT1_2 * frgb_HDCTHexarray.p[rgb][1] * psiCos_table[uv][ij];
				tempx.p[p][2] += M_SQRT1_2 * frgb_HDCTHexarray.p[rgb][2] * psiCos_table[uv][ij];
			}
		}

		tempx.p[p][0] *= 2.0f / 7;
		tempx.p[p][1] *= 2.0f / 7;
		tempx.p[p][2] *= 2.0f / 7;
	}

	for(unsigned int i = 0; i < tempx.size; i++) {
		rgb_hexarray->p[i][0] = (int)abs(round(tempx.p[i][0]));
		rgb_hexarray->p[i][1] = (int)abs(round(tempx.p[i][1]));
		rgb_hexarray->p[i][2] = (int)abs(round(tempx.p[i][2]));
	}

	Hexarray_free(&tempx, 1);
}

void HDCT_N5(RGB_Hexarray rgb_hexarray, RGB_Array* rgb_HDCTArray, float** psiCos_table) {
	if(rgb_hexarray.size < 49) {
		printf("\n\nError - HDCT_N5: rgb_hexarray.size (%u) < 49", rgb_hexarray.size);
		exit(EXIT_FAILURE);
	}


	const unsigned int tmp = rgb_hexarray.size / 49;

	pArray2d_init(rgb_HDCTArray, tmp, 61);


	const unsigned int base[12] = { 33, 34, 41, 36, 31, 26, 44, 13, 18, 17, 10, 9 };

	unsigned int** spatial;
	iPoint2d spatial_size = { .x = 0, .y = 0 };

	fPoint2d ps;
	iPoint2d mx = { .x = 0, .y = 0 };
	iPoint2d mn = { .x = 0, .y = 0 };

	for(unsigned int i = 0; i < rgb_hexarray.size; i++) {
		ps = getSpatial(Hexint_init(i, 0));

		if(ps.x < mn.x) {
			mn.x = (int)ps.x;
		} else if(ps.x > mx.x) {
			mx.x = (int)ps.x;
		}
		if(ps.y < mn.y) {
			mn.y = (int)ps.y;
		} else if(ps.y > mx.y) {
			mx.y = (int)ps.y;
		}
	}

	spatial_size.x = mx.x - mn.x + 1;
	spatial_size.y = mx.y - mn.y + 1;

	spatial = (unsigned int**)malloc(spatial_size.x * sizeof(unsigned int*));
	for(unsigned int i = 0; i < spatial_size.x; i++) {
		spatial[i] = (unsigned int*)malloc(spatial_size.y * sizeof(unsigned int));
	}

	for(unsigned int i = 0; i < rgb_hexarray.size; i++) {
		ps = getSpatial(Hexint_init(i, 0));

		spatial[(int)(ps.x - mn.x)][(int)(ps.y - mn.y)] = i;
	}


	for(unsigned int p = 0; p < tmp * 61; p++) {
		float HDCT[3] = { 0.0f, 0.0f, 0.0f };
		const unsigned int tempp = p / 61;
		const unsigned int uv    = p % 61;
		bool skip = true;

		for(unsigned int ij = 0; ij < 61; ij++) {
			int rgb;

			if(ij < 49) {
				rgb = tempp * 49 + ij;
			} else {
				if(skip) break;

				ps = getSpatial(Hexint_init(tempp * 49 + base[ij - 49], 0));

				if(ij == 49 || ij == 50) {
					ps.y -= 1;
				} else if(ij == 51 || ij == 52) {
					ps.x += 1;
				} else if(ij == 53 || ij == 54) {
					ps.x -= 1;
				} else if(ij == 55 || ij == 56) {
					ps.x += 1;
				} else if(ij == 57 || ij == 58) {
					ps.x -= 1;
				} else {
					ps.y += 1;
				}

				if(spatial[(int)(ps.x - mn.x)][(int)(ps.y - mn.y)] < rgb_hexarray.size) {
					ps.x -= mn.x;
					ps.y -= mn.y;

					rgb = spatial[(int)(ps.x)][(int)(ps.y)];
				} else {
					rgb = -1;
				}
			}

			if(rgb > -1) {
				if(rgb_hexarray.p[rgb][0] || rgb_hexarray.p[rgb][1] || rgb_hexarray.p[rgb][2]) {
					HDCT[0] += rgb_hexarray.p[rgb][0] * psiCos_table[uv][ij];
					HDCT[1] += rgb_hexarray.p[rgb][1] * psiCos_table[uv][ij];
					HDCT[2] += rgb_hexarray.p[rgb][2] * psiCos_table[uv][ij];
					skip = false;
				}
			}
		}

		if(uv != 49) {
			rgb_HDCTArray->p[tempp][uv][0] = (int)round(HDCT[0]);
			rgb_HDCTArray->p[tempp][uv][1] = (int)round(HDCT[1]);
			rgb_HDCTArray->p[tempp][uv][2] = (int)round(HDCT[2]);
		} else {
			rgb_HDCTArray->p[tempp][uv][0] = (int)round(M_SQRT1_2 * HDCT[0]);
			rgb_HDCTArray->p[tempp][uv][1] = (int)round(M_SQRT1_2 * HDCT[1]);
			rgb_HDCTArray->p[tempp][uv][2] = (int)round(M_SQRT1_2 * HDCT[2]);
		}
	}


	for(unsigned int i = 0; i < spatial_size.x; i++) {
		free(spatial[i]);
	}
	free(spatial);
}

void IHDCT_N5(RGB_Array rgb_HDCTArray, RGB_Hexarray* rgb_hexarray, float** psiCos_table) {
	if(rgb_hexarray->size < 49) {
		printf("\n\nError - IHDCT_N5: rgb_hexarray->size (%u) < 49", rgb_hexarray->size);
		exit(EXIT_FAILURE);
	}


	fRGB_Hexarray tempx;

	Hexarray_init(&tempx, (unsigned int)(logf(rgb_hexarray->size) / logf(7)), 1);

	for(unsigned int p = 0; p < rgb_hexarray->size; p++) {
		const unsigned int ij = p % 49;

		for(unsigned int uv = 0; uv < 61; uv++) {
			const unsigned int sub = p / 49;

			if(uv != 49) {
				tempx.p[p][0] += rgb_HDCTArray.p[sub][uv][0] * psiCos_table[uv][ij];
				tempx.p[p][1] += rgb_HDCTArray.p[sub][uv][1] * psiCos_table[uv][ij];
				tempx.p[p][2] += rgb_HDCTArray.p[sub][uv][2] * psiCos_table[uv][ij];
			} else {
				tempx.p[p][0] += M_SQRT1_2 * rgb_HDCTArray.p[sub][uv][0] * psiCos_table[uv][ij];
				tempx.p[p][1] += M_SQRT1_2 * rgb_HDCTArray.p[sub][uv][1] * psiCos_table[uv][ij];
				tempx.p[p][2] += M_SQRT1_2 * rgb_HDCTArray.p[sub][uv][2] * psiCos_table[uv][ij];
			}
		}

		tempx.p[p][0] *= 2.0f / 61;
		tempx.p[p][1] *= 2.0f / 61;
		tempx.p[p][2] *= 2.0f / 61;
	}

	pArray2d_free(&rgb_HDCTArray);

	for(unsigned int i = 0; i < tempx.size; i++) {
		rgb_hexarray->p[i][0] = (int)abs(round(tempx.p[i][0]));
		rgb_hexarray->p[i][1] = (int)abs(round(tempx.p[i][1]));
		rgb_hexarray->p[i][2] = (int)abs(round(tempx.p[i][2]));
	}

	Hexarray_free(&tempx, 1);
}


// Makroblockgröße = <N=(>1)> (TODO?)

void HDCT2_init(float***** psiCos_table, unsigned int N) {
	const unsigned int wh = 2 * N - 1;                 // max. Durchmesser
	const unsigned int M  = 3 * pow(N, 2) - 3 * N + 1; // und #Pixel

	*psiCos_table = (float****)malloc(wh * sizeof(float***));
	for(unsigned int i = 0; i < wh; i++) {
		(*psiCos_table)[i] = (float***)malloc(wh * sizeof(float**));
		for(unsigned int j = 0; j < wh; j++) {
			(*psiCos_table)[i][j] = (float**)malloc(wh * sizeof(float*));
			for(unsigned int k = 0; k < wh; k++) {
				(*psiCos_table)[i][j][k] = (float*)calloc(wh, sizeof(float));
			}
		}
	}

	for(unsigned int u = 0; u < wh; u++) {
		for(unsigned int v = 0; v < wh; v++) {
			if(abs(u - v) < N) {
				for(unsigned int i = 0; i < wh; i++) {
					for(unsigned int j = 0; j < wh; j++) {
						if(abs(i - j) < N) {
							(*psiCos_table)[u][v][i][j] = cos(M_PI / M * \
								((i - (float)N / 2 + 1) * ((4 * (float)N - 2) * u - (N - 1) * v) - \
								(j + 0.5f) * (2 * (float)N * u - (2 * N - 1) * v)));
						}
					}
				}
			}
		}
	}
}

void HDCT2_free(float***** psiCos_table, unsigned int N) {
	const unsigned int wh = 2 * N - 1; // max. Durchmesser

	for(unsigned int i = 0; i < wh; i++) {
		for(unsigned int j = 0; j < wh; j++) {
			for(unsigned int k = 0; k < wh; k++) {
				free((*psiCos_table)[i][j][k]);
			}
			free((*psiCos_table)[i][j]);
		}
		free((*psiCos_table)[i]);
	}
	free(*psiCos_table);
}




// Quadratische DCT DCTH (Makroblockgröße = <N=(2|5)>)


float c(unsigned int x) {
	return x ? 1.0f : M_SQRT1_2;
}

float cosu(unsigned int i, unsigned int u, unsigned int N) {
	return cos((M_PI * (2 * i + 1) * u) / (2 * N));
}

float cosv(unsigned int j, unsigned int v, unsigned int M) {
	return cos((M_PI * (2 * j + 1) * v) / (2 * M));
}


void DCTH_init(float*** psiCos_table, unsigned int N) {
	const unsigned int wh = 2 * N - 1; // max. Durchmesser
	const unsigned int M  = wh * wh;   // und #Pixel


	const int i2H2[3][3] =
		{ { 5, 4, 7 },
		  { 6, 0, 3 },
		  { 8, 1, 2 } };

	const int i2H5[9][9] =
		{ { 49, 33, 32, 53, 54, 61, 62, 63, 64 },
		  { 50, 34, 28, 31, 26, 25, 65, 66, 67 },
		  { 40, 39, 29, 30, 27, 21, 24, 68, 69 },
		  { 41, 35, 38,  5,  4, 22, 23, 57, 70 },
		  { 51, 36, 37,  6,  0,  3, 19, 18, 58 },
		  { 71, 52, 47, 46,  1,  2, 20, 14, 17 },
		  { 72, 73, 48, 42, 45, 12, 11, 15, 16 },
		  { 74, 75, 76, 43, 44, 13,  7, 10, 59 },
		  { 77, 78, 79, 80, 55, 56,  8,  9, 60 } };


	*psiCos_table = (float**)malloc(M * sizeof(float*));
	for(unsigned int i = 0; i < M; i++) {
		(*psiCos_table)[i] = (float*)calloc(M, sizeof(float));
	}

	for(unsigned int u = 0; u < wh; u++) {
		for(unsigned int v = 0; v < wh; v++) {
			for(unsigned int i = 0; i < wh; i++) {
				for(unsigned int j = 0; j < wh; j++) {
					if(N == 5) {
						(*psiCos_table)[i2H5[u][v]][i2H5[i][j]] = c(u) * cosu(i, u, wh) * c(v) * cosv(j, v, wh);
					} else {
						(*psiCos_table)[i2H2[u][v]][i2H2[i][j]] = c(u) * cosu(i, u, wh) * c(v) * cosv(j, v, wh);
					}
				}
			}
		}
	}
}

void DCTH_free(float*** psiCos_table, bool mode, unsigned int N) {
	unsigned int M; // #Pixel

	if(!mode) {
		M = N * (4 * N - 4) + 1;
	} else {
		M = N == 2 ? 7 : 49;
	}

	for(unsigned int i = 0; i < M; i++) {
		free((*psiCos_table)[i]);
	}
	free(*psiCos_table);
}


void DCTH(RGB_Hexarray rgb_hexarray, RGB_Array* rgb_HDCTArray, float** psiCos_table, unsigned int N) {
	const unsigned int wh       = 2 * N - 1;
	const float        factor   = 2 / sqrt(wh * wh);
	const unsigned int M        = wh * wh;
	const unsigned int sub_size = N == 2 ? 7 : 49;
	const unsigned int tmp      = rgb_hexarray.size / sub_size;

	const unsigned int out[32] = {
		17, 16, 25, 24,  8,  9, 33, 32, 48, 43, 40, 41, 41, 35, 38,  5,
		36, 37,  6, 47, 46, 42, 21, 22, 23,  3, 19, 18,  2, 20, 14, 17 };

	pArray2d_init(rgb_HDCTArray, tmp, M);

	for(unsigned int p = 0; p < tmp * M; p++) {
		float HDCT[3] = { 0.0f, 0.0f, 0.0f };
		const unsigned int tempp = p / M;
		const unsigned int uv    = p % M;

		for(unsigned int ij = 0; ij < (N == 5 ? M : sub_size); ij++) {
			unsigned int rgb;

			if(ij < 49) {
				rgb = tempp * sub_size + ij;
			} else {
				rgb = tempp * sub_size + out[ij - 49];
			}

			HDCT[0] += rgb_hexarray.p[rgb][0] * psiCos_table[uv][ij];
			HDCT[1] += rgb_hexarray.p[rgb][1] * psiCos_table[uv][ij];
			HDCT[2] += rgb_hexarray.p[rgb][2] * psiCos_table[uv][ij];
		}

		rgb_HDCTArray->p[tempp][uv][0] = (int)round(factor * HDCT[0]);
		rgb_HDCTArray->p[tempp][uv][1] = (int)round(factor * HDCT[1]);
		rgb_HDCTArray->p[tempp][uv][2] = (int)round(factor * HDCT[2]);
	}
}

void IDCTH(RGB_Array rgb_HDCTArray, RGB_Hexarray* rgb_hexarray, float** psiCos_table, unsigned int N) {
	const unsigned int wh       = 2 * N - 1;
	const float        factor   = 2 / sqrt(wh * wh);
	const unsigned int M        = wh * wh;
	const unsigned int sub_size = N == 2 ? 7 : 49;
	unsigned int       order    = 1;
	fRGB_Hexarray      tempx;

	for(unsigned int i = rgb_hexarray->size; i > 7; i /= 7) {
		order++;
	}
	Hexarray_init(&tempx, order, 1);

	for(unsigned int p = 0; p < rgb_hexarray->size; p++) {
		const unsigned int ij = p % sub_size;

		for(unsigned int uv = 0; uv < M; uv++) {
			const unsigned int sub = p / sub_size;

			tempx.p[p][0] += rgb_HDCTArray.p[sub][uv][0] * psiCos_table[uv][ij];
			tempx.p[p][1] += rgb_HDCTArray.p[sub][uv][1] * psiCos_table[uv][ij];
			tempx.p[p][2] += rgb_HDCTArray.p[sub][uv][2] * psiCos_table[uv][ij];
		}

		tempx.p[p][0] *= factor;
		tempx.p[p][1] *= factor;
		tempx.p[p][2] *= factor;
	}

	pArray2d_free(&rgb_HDCTArray);

	for(unsigned int i = 0; i < tempx.size; i++) {
		rgb_hexarray->p[i][0] = (int)abs(round(tempx.p[i][0]));
		rgb_hexarray->p[i][1] = (int)abs(round(tempx.p[i][1]));
		rgb_hexarray->p[i][2] = (int)abs(round(tempx.p[i][2]));
	}

	Hexarray_free(&tempx, 1);
}




// 1D-DCTH (Modi: Spiral Structure Addressing SSA / SSA Log Space, Makroblockgröße = <N=(2|5)>)


void DCTH2_init(float*** psiCos_table, unsigned int N) {
	const unsigned int M = N == 2 ? 7 : 49; // #Pixel

	*psiCos_table = (float**)malloc(M * sizeof(float*));
	for(unsigned int i = 0; i < M; i++) {
		(*psiCos_table)[i] = (float*)calloc(M, sizeof(float));
	}

	for(unsigned int u = 0; u < M; u++) {
		for(unsigned int i = 0; i < M; i++) {
			(*psiCos_table)[i][u] = c(u) * cosu(i, u, M);
		}
	}
}

#define DCTH2_free DCTH_free


void DCTH2(RGB_Hexarray rgb_hexarray, fRGB_Hexarray* frgb_HDCTHexarray, float** psiCos_table, bool mode, unsigned int N) {
	const unsigned int M      = N == 2 ? 7 : 49;
	const float        factor = sqrt(2.0f / M);

	Hexarray_init(frgb_HDCTHexarray, (unsigned int)(logf(rgb_hexarray.size) / logf(7)), 1);

	for(unsigned int p = 0; p < frgb_HDCTHexarray->size; p++) {
		const unsigned int u = p % M;

		for(unsigned int i = 0; i < M; i++) {
			unsigned int rgb;

			if(!mode) {
				rgb = p - u + i;
			} else {
				if(N == 5) {
					rgb = p - u + i2LogSpace5[i];
				} else {
					rgb = p - u + i2LogSpace2[i];
				}
			}

			frgb_HDCTHexarray->p[p][0] += rgb_hexarray.p[rgb][0] * psiCos_table[i][u];
			frgb_HDCTHexarray->p[p][1] += rgb_hexarray.p[rgb][1] * psiCos_table[i][u];
			frgb_HDCTHexarray->p[p][2] += rgb_hexarray.p[rgb][2] * psiCos_table[i][u];
		}

		frgb_HDCTHexarray->p[p][0] *= factor;
		frgb_HDCTHexarray->p[p][1] *= factor;
		frgb_HDCTHexarray->p[p][2] *= factor;
	}
}

void IDCTH2(fRGB_Hexarray frgb_HDCTHexarray, RGB_Hexarray* rgb_hexarray, float** psiCos_table, bool mode, unsigned int N) {
	const unsigned int M      = N == 2 ? 7 : 49;
	const float        factor = sqrt(2.0f / M);
	fRGB_Hexarray      tempx;

	Hexarray_init(&tempx, (unsigned int)(logf(rgb_hexarray->size) / logf(7)), 1);

	for(unsigned int p = 0; p < tempx.size; p++) {
		const unsigned int i = p % M;

		for(unsigned int u = 0; u < M; u++) {
			unsigned int rgb = p - i + u;

			if(!mode) {
				rgb = p - i + u;
			} else {
				if(N == 5) {
					rgb = p - i + i2LogSpace5[u];
				} else {
					rgb = p - i + i2LogSpace2[u];
				}
			}

			tempx.p[p][0] += frgb_HDCTHexarray.p[rgb][0] * psiCos_table[i][u];
			tempx.p[p][1] += frgb_HDCTHexarray.p[rgb][1] * psiCos_table[i][u];
			tempx.p[p][2] += frgb_HDCTHexarray.p[rgb][2] * psiCos_table[i][u];
		}

		tempx.p[p][0] *= factor;
		tempx.p[p][1] *= factor;
		tempx.p[p][2] *= factor;
	}

	Hexarray_free(&frgb_HDCTHexarray, 1);

	for(unsigned int i = 0; i < tempx.size; i++) {
		rgb_hexarray->p[i][0] = (int)abs(round(tempx.p[i][0]));
		rgb_hexarray->p[i][1] = (int)abs(round(tempx.p[i][1]));
		rgb_hexarray->p[i][2] = (int)abs(round(tempx.p[i][2]));
	}

	Hexarray_free(&tempx, 1);
}

