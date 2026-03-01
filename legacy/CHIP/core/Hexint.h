/******************************************************************************
 * Hexint.h: Funktionen auf Hexints
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


#ifndef HEXINT_H
#define HEXINT_H


#include "../misc/defines.h"
#include "../misc/types.h"


Hexint Hexint_init(int value, bool base7);     // neuer Hexint - zur Basis 7 / 10?
fPoint2d getReal(Hexint self);                 // return cartesian coords of hexint
fPoint2d getPolar(Hexint self);                // return polar coords for a hexint
unsigned int getInt(Hexint self);              // returns a base 10 integer corresponding to a hexint
Hexint neg(Hexint self);                       // Hexint Negation
Hexint add(Hexint self, Hexint object);        // Addition
Hexint mul_int(Hexint self, int object);       // Multiplikation mit Integer
Hexint getNearest(float x, float y);           // find the nearest Hexint to cartesian coords
fPoint3d getHer(Hexint self);                  // returns a 3-tuple using hers coord system
Hexint mul_Hexint(Hexint self, Hexint object); // Multiplikation mit Hexint
Hexint sub(Hexint self, Hexint object);        // Subtraktion
fPoint2d getSpatial(Hexint self);              // return spatial integer coord pair
fPoint2d getFrequency(Hexint self);            // return fequency integer coord pair
fPoint2d getSkew(Hexint self);                 // return integer coord of skewed 60 degree axis


#endif

