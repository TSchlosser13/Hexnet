/******************************************************************************
 * Hexarray.h: Funktionen auf Hexarrays
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


#ifndef HEXARRAY_H
#define HEXARRAY_H


#include "../misc/defines.h"
#include "../misc/types.h"


// Modi: RGB_Hexarray (int) / fRGB_Hexarray (float)
void Hexarray_init(void* hexarray, unsigned int order, bool type);
void Hexarray_free(void* hexarray, bool type);
void Hexarray2file(void* hexarray, char* filename, bool type);

// Weitere Ausgaben als PNG
void Hexarray2PNG_1D(RGB_Hexarray hexarray, char* filename);          // eindimensional (auf einer Zeile)
void Hexarray2PNG_2D(RGB_Hexarray hexarray, char* filename);          // zweidimensional mittels schiefer Achse
void Hexarray2PNG_2D_directed(RGB_Hexarray hexarray, char* filename); // zweidimensional mittels kartesischem KOS

// Skalierung: eigens entwickelte / nach HIP
void Hexarray_scale(void* hexarray, bool mode, unsigned int times, bool type); // Modi: herunter / hoch (times x /7 / *7)
void Hexarray_scale_HIP(void* hexarray, bool mode, unsigned int mod);          // Modi: 1 / 7 Versionen (nur Herunterskalierung, mod = Mul.-Modifikator)


#endif

