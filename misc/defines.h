/******************************************************************************
 * defines.h: Preprocessor Directives
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


#ifndef DEFINES_H
#define DEFINES_H


/******************************************************************************
 * Includes
 ******************************************************************************/

#include <math.h>




/******************************************************************************
 * Defines
 ******************************************************************************/


/******************************************************************************
 * sqrt(3) and sqrt(3) / 2
 ******************************************************************************/

#ifndef M_SQRT3
	#define M_SQRT3   1.7320508076
#endif

#ifndef M_SQRT3_2
	#define M_SQRT3_2 0.8660254038
#endif


/******************************************************************************
 * min(a, b) and max(a, b)
 ******************************************************************************/

#undef  MIN
#define MIN(a, b) ({ \
	__auto_type _a = (a); \
	__auto_type _b = (b); \
	_a < _b ? _a : _b;    \
})

#undef  MAX
#define MAX(a, b) ({ \
	__auto_type _a = (a); \
	__auto_type _b = (b); \
	_a > _b ? _a : _b;    \
})


/******************************************************************************
 * Clock difference
 ******************************************************************************/

#define CLOCK_DIFF(ts1, ts2) ( ts2.tv_sec - ts1.tv_sec + (ts2.tv_nsec - ts1.tv_nsec) / 1000000000.0 )




#endif


