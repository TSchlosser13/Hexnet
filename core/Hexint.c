/******************************************************************************
 * Hexint.c: Operations on Hexagonal Pixels
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


#include "Hexint.h"

#include "../misc/types.h"


Hexint int_to_Hexint(u32 u) {
	Hexint h = 0;
	u32    m = 1;

	while(u) {
		h += m * (u % 7);
		u /=  7;
		m *= 10;
	}

	return h;
}

u32 Hexint_to_int(Hexint h) {
	u32 u = 0;
	u32 m = 1;

	while(h) {
		u += m * (h % 10);
		h /= 10;
		m *=  7;
	}

	return u;
}

