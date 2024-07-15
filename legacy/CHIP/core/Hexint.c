/******************************************************************************
 * Hexint.c: Funktionen auf Hexints
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


#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include <math.h>

#include "../misc/defines.h"
#include "../misc/precalculations.h"
#include "../misc/types.h"

#include "Hexint.h"


#define u8 uint_fast8_t


// Neuer Hexint - zur Basis 7 / 10?
Hexint Hexint_init(int value, bool base7) {
	Hexint hexint;

	if(value > 0) {
		unsigned int me_value = value;
		unsigned int tmp;

		hexint.ndigits = 0;

		if(base7) {
			while(me_value) {
				tmp = me_value % 10;
				if(tmp > 6) {
					printf("\n\nError - Hexint_init: digit (%u) > 6 (value = %i)", tmp, value);
					exit(EXIT_FAILURE);
				}
				me_value = me_value / 10;

				hexint.ndigits++;
			}
			hexint.numstr = value;
		} else {
			unsigned int mul = 1;

			tmp = 0;
			while(me_value) {
				tmp      += (me_value % 7) * mul;
				me_value /=  7;
				mul      *= 10;

				hexint.ndigits++;
			}
			hexint.numstr = tmp;
			hexint.base10 = value;
		}
	} else if(!value) {
		hexint.numstr  = 0;
		hexint.ndigits = 1;
	} else {
		printf("\n\nError - Hexint_init: value (%i) < 0", value);
		exit(EXIT_FAILURE);
	}

	return hexint;
}

// return cartesian coords of hexint
fPoint2d getReal(Hexint self) {
	fPoint2d pc = { .x = 1.0f, .y = 0.0f };
	fPoint2d p = { .x = 0.0f, .y = 0.0f };
	const float sqrt3 = sqrt(3);
	unsigned int value = self.numstr;
	unsigned int tmp;

	for(int i = self.ndigits - 1; i > -1; i--) {
		// compute key points
		if(i < self.ndigits - 1) {
			const float tmpx = pc.x;

			pc.x = 2 * pc.x - sqrt3 * pc.y;
			pc.y = sqrt3 * tmpx + 2 * pc.y;
		}

		// compute rotation
		tmp = value % 10;
		if(tmp == 1) {
			p.x = p.x + pc.x;
			p.y = p.y + pc.y;
		} else if(tmp == 2) {
			p.x = p.x + pc.x / 2 - (sqrt3 * pc.y) / 2;
			p.y = p.y + (sqrt3 * pc.x) / 2 + pc.y / 2;
		} else if(tmp == 3) {
			p.x = p.x - pc.x / 2 - (sqrt3 * pc.y) / 2;
			p.y = p.y + (sqrt3 * pc.x) / 2 - pc.y / 2;
		} else if(tmp == 4) {
			p.x = p.x - pc.x;
			p.y = p.y - pc.y;
		} else if(tmp == 5) {
			p.x = p.x - pc.x / 2 + (sqrt3 * pc.y) / 2;
			p.y = p.y - (sqrt3 * pc.x) / 2 - pc.y / 2;
		} else if(tmp == 6) {
			p.x = p.x + pc.x / 2 + (sqrt3 * pc.y) / 2;
			p.y = p.y - (sqrt3 * pc.x) / 2 + pc.y / 2;
		}
		value = value / 10;
	}

	return p;
}

// return polar coords for a hexint
fPoint2d getPolar(Hexint self) {
	fPoint2d p;

	if(self.numstr) {
		float tmp;

		p = getReal(self);

		tmp = p.x;
		p.x = sqrt(p.x * p.x + p.y * p.y);
		p.y = atan2(p.y, tmp);
	} else {
		p.x = 0.0f;
		p.y = 0.0f;
	}

	return p;
}

// returns a base 10 integer corresponding to a hexint
unsigned int getInt(Hexint self) {
	unsigned int total = 0;
	unsigned int mul = 1;
	unsigned int value = self.numstr;

	for(int i = self.ndigits - 1; i > -1; i--) {
		total = total + (value % 10) * mul;
		mul = mul * 7;
		value = value / 10;
	}

	return total;
}

// Hexint Negation
Hexint neg(Hexint self) {
	unsigned int total = 0;
	unsigned int mul = 1;
	unsigned int value = self.numstr;
	unsigned int tmp;

	for(int i = self.ndigits - 1; i > -1; i--) {
		tmp = value % 10;
		if(tmp == 1) {
			total = total + 4 * mul;
		} else if(tmp == 2) {
			total = total + 5 * mul;
		} else if(tmp == 3) {
			total = total + 6 * mul;
		} else if(tmp == 4) {
			total = total + 1 * mul;
		} else if(tmp == 5) {
			total = total + 2 * mul;
		} else if(tmp == 6) {
			total = total + 3 * mul;
		}
		mul = mul * 10;
		value = value / 10;
	}

	return Hexint_init(total, 1);
}

Hexint add(Hexint self, Hexint object) {
	const unsigned int A[7][7] =
		{ {  0,  1,  2,  3,  4,  5,  6 },
		  {  1, 63, 15,  2,  0,  6, 64 },
		  {  2, 15, 14, 26,  3,  0,  1 },
		  {  3,  2, 26, 25, 31,  4,  0 },
		  {  4,  0,  3, 31, 36, 42,  5 },
		  {  5,  6,  0,  4, 42, 41, 53 },
		  {  6, 64,  1,  0,  5, 53, 52 } };

	u8* numa;
	u8* numb;
	unsigned int maxlen;
	unsigned int value_self = self.numstr;
	unsigned int value_object = object.numstr;
	unsigned int total = 0;
	unsigned int mul = 1;

	if((int)(self.ndigits - object.ndigits) > 0) {
		maxlen = self.ndigits + 1;
	} else {
		maxlen = object.ndigits + 1;
	}

	numa = (u8*)malloc(maxlen * sizeof(u8));
	numb = (u8*)malloc(maxlen * sizeof(u8));

	for(int i = maxlen - 1; i > -1; i--) {
		if(maxlen - self.ndigits) {
			numa[i] = value_self % 10;
		}

		if(maxlen - object.ndigits) {
			numb[i] = value_object % 10;
		}

		if(value_self) {
			value_self = value_self / 10;
		}
		if(value_object) {
			value_object = value_object / 10;
		}
	}

	for(unsigned int i = 0; i < maxlen; i++) {
		const unsigned int ii = maxlen - i - 1;
		unsigned int t = A[numa[ii]][numb[ii]];
		unsigned int carry = t / 10;

		total = total + (t % 10) * mul;

		for(unsigned int j = i + 1; j < maxlen; j++) {
			const unsigned int jj = maxlen - j - 1;

			if(carry) {
				t = A[numa[jj]][carry];
				numa[jj] = t % 10;
				carry = t / 10;
			}
		}
		mul = mul * 10;
	}

	free(numa);
	free(numb);

	return Hexint_init(total, 1);
}


int binary_search_3d(
 uPoint3d* me_pc_adds_mul,
 unsigned int min, unsigned int max,
 unsigned int x, unsigned int y) {
	if(min == max) return -1;

	int me_min = (int)min; unsigned int me_max = max;

	while(1) {
		if(me_min > me_max) return -1;

		const unsigned int me_mid = me_min != me_max ? (me_min + me_max) / 2 : me_min;

		if(x == me_pc_adds_mul[me_mid].x && y == me_pc_adds_mul[me_mid].y) {
			return me_pc_adds_mul[me_mid].z;
		} else if(x < me_pc_adds_mul[me_mid].x) {
			me_max = me_mid - 1;
		} else if(x > me_pc_adds_mul[me_mid].x) {
			me_min = me_mid + 1;
		} else if(y < me_pc_adds_mul[me_mid].y) {
			if(me_min < 0) return -1;

			me_min = me_mid - 1; me_max = me_mid - 1;
		} else {
			if(me_max > max) return -1;

			me_min = me_mid + 1; me_max = me_mid + 1;
		}
	}
}


// Multiplikation mit Integer
Hexint mul_int(Hexint self, int object) {
	Hexint num = Hexint_init(self.numstr, 1);
	Hexint sum = Hexint_init(0, 1);

	if(object < 1) {
		num = neg(num);
	}


	const unsigned int object_abs = (const unsigned int)abs(object);

	if(object_abs > 0) {
		// Nicht po2 (power of two)
		if(object_abs & (object_abs - 1)) {

			Hexint*       hfactors     = NULL;
			unsigned int* ifactors     = NULL;
			unsigned int  factors_size = 0;
			unsigned int  factors_sum  = 0;

			sum = self;

			// Faktoren aufaddieren (= self^(po2 + 1))
			for(unsigned int po2 = 1; po2 <= object_abs / 2; po2 *= 2) {
				factors_size++;
				hfactors =       (Hexint*)realloc(hfactors, factors_size * sizeof(Hexint));
				ifactors = (unsigned int*)realloc(ifactors, factors_size * sizeof(unsigned int*));
				hfactors[factors_size - 1] = sum;
				ifactors[factors_size - 1] = po2;

				// case n : sum = n + n
				switch(sum.numstr) {
					// object =   2
					case      1 : sum = Hexint_init(     63, 1); break;
					case      2 : sum = Hexint_init(     14, 1); break;
					case      4 : sum = Hexint_init(     36, 1); break;
					case      5 : sum = Hexint_init(     41, 1); break;
					// object =   4
					case     63 : sum = Hexint_init(    645, 1); break;
					case     14 : sum = Hexint_init(    156, 1); break;
					case     36 : sum = Hexint_init(    312, 1); break;
					case     41 : sum = Hexint_init(    423, 1); break;
					// object =   8
					case    645 : sum = Hexint_init(    651, 1); break;
					case    156 : sum = Hexint_init(    162, 1); break;
					case    312 : sum = Hexint_init(    324, 1); break;
					case    423 : sum = Hexint_init(    435, 1); break;
					// object =  16
					case    651 : sum = Hexint_init(   5043, 1); break;
					case    162 : sum = Hexint_init(   6054, 1); break;
					case    324 : sum = Hexint_init(   2016, 1); break;
					case    435 : sum = Hexint_init(   3021, 1); break;
					// object =  32
					case   5043 : sum = Hexint_init(  41315, 1); break;
					case   6054 : sum = Hexint_init(  52426, 1); break;
					case   2016 : sum = Hexint_init(  14642, 1); break;
					case   3021 : sum = Hexint_init(  25153, 1); break;
					// object =  64
					case  41315 : sum = Hexint_init(  46511, 1); break;
					case  52426 : sum = Hexint_init(  51622, 1); break;
					case  14642 : sum = Hexint_init(  13244, 1); break;
					case  25153 : sum = Hexint_init(  24355, 1); break;
					// object = 128
					case  46511 : sum = Hexint_init( 430403, 1); break;
					case  51622 : sum = Hexint_init( 540504, 1); break;
					case  13244 : sum = Hexint_init( 160106, 1); break;
					case  24355 : sum = Hexint_init( 210201, 1); break;
					// object = 256
					case 430403 : sum = Hexint_init(3153625, 1); break;
					case 540504 : sum = Hexint_init(4264136, 1); break;
					case 160106 : sum = Hexint_init(6426352, 1); break;
					case 210201 : sum = Hexint_init(1531463, 1); break;
					// order > 7
					default     : sum = add(sum, sum);           break;
				}

				factors_sum = 2 * po2;
			}

			// Rest
			if(object_abs > 2) {
				while(factors_size) {
					const unsigned int rest = object_abs - factors_sum;

					factors_size--;

					if(rest < ifactors[factors_size]) {
						continue;
					} else if(rest > ifactors[factors_size]) {
						const int z_int = binary_search_3d(pc_adds_mul, 0, pc_adds_mul_size - 1, sum.numstr, hfactors[factors_size].numstr);

						if(z_int > -1) {
							sum = Hexint_init(z_int, 1);
						} else {
							sum = add(sum, hfactors[factors_size]);
						}


						factors_sum += ifactors[factors_size];
						continue;
					} else {
						const int z_int = binary_search_3d(pc_adds_mul, 0, pc_adds_mul_size - 1, sum.numstr, hfactors[factors_size].numstr);

						if(z_int > -1) {
							sum = Hexint_init(z_int, 1);
						} else {
							sum = add(sum, hfactors[factors_size]);
						}


						break;
					}
				}
			}

			free(hfactors);
			free(ifactors);

		// und po2
		} else {
			sum = self;

			for(unsigned int po2 = 1; po2 <= object_abs / 2; po2 *= 2) {
				// case n : sum = n + n
				switch(sum.numstr) {
					// object =   2
					case      1 : sum = Hexint_init(     63, 1); break;
					case      2 : sum = Hexint_init(     14, 1); break;
					case      4 : sum = Hexint_init(     36, 1); break;
					case      5 : sum = Hexint_init(     41, 1); break;
					// object =   4
					case     63 : sum = Hexint_init(    645, 1); break;
					case     14 : sum = Hexint_init(    156, 1); break;
					case     36 : sum = Hexint_init(    312, 1); break;
					case     41 : sum = Hexint_init(    423, 1); break;
					// object =   8
					case    645 : sum = Hexint_init(    651, 1); break;
					case    156 : sum = Hexint_init(    162, 1); break;
					case    312 : sum = Hexint_init(    324, 1); break;
					case    423 : sum = Hexint_init(    435, 1); break;
					// object =  16
					case    651 : sum = Hexint_init(   5043, 1); break;
					case    162 : sum = Hexint_init(   6054, 1); break;
					case    324 : sum = Hexint_init(   2016, 1); break;
					case    435 : sum = Hexint_init(   3021, 1); break;
					// object =  32
					case   5043 : sum = Hexint_init(  41315, 1); break;
					case   6054 : sum = Hexint_init(  52426, 1); break;
					case   2016 : sum = Hexint_init(  14642, 1); break;
					case   3021 : sum = Hexint_init(  25153, 1); break;
					// object =  64
					case  41315 : sum = Hexint_init(  46511, 1); break;
					case  52426 : sum = Hexint_init(  51622, 1); break;
					case  14642 : sum = Hexint_init(  13244, 1); break;
					case  25153 : sum = Hexint_init(  24355, 1); break;
					// object = 128
					case  46511 : sum = Hexint_init( 430403, 1); break;
					case  51622 : sum = Hexint_init( 540504, 1); break;
					case  13244 : sum = Hexint_init( 160106, 1); break;
					case  24355 : sum = Hexint_init( 210201, 1); break;
					// object = 256
					case 430403 : sum = Hexint_init(3153625, 1); break;
					case 540504 : sum = Hexint_init(4264136, 1); break;
					case 160106 : sum = Hexint_init(6426352, 1); break;
					case 210201 : sum = Hexint_init(1531463, 1); break;
					// order > 7
					default     : sum = add(sum, sum);           break;
				}
			}
		}
	}


	return sum;
}


// find the nearest Hexint to cartesian coords
Hexint getNearest(float x, float y) {
	const float sqrt3 = sqrt(3);
	Hexint o1 = Hexint_init(1, 1);
	Hexint o2 = Hexint_init(2, 1);
	Hexint h = Hexint_init(0, 1);
	const float r1 = x - y / sqrt3;
	const float r2 = 2 * y / sqrt3;

	if(r1 < 0) {
		o1 = Hexint_init(4, 1);

		if(r1 + floor(r1) > 0.5f) {
			// h = add(h, o1);
			h = Hexint_init(4, 1);
		}
	} else if(r1 - floor(r1) > 0.5f) {
		// h = add(h, o1);
		h = Hexint_init(1, 1);
	}
	if(r2 < 0) {
		o2 = Hexint_init(5, 1);

		if(r2 + floor(r2) > 0.5f) {
			// h = add(h, o2);
			if(r1 < 0) {
				h = Hexint_init(42, 1);
			} else {
				h = Hexint_init(6, 1);
			}
		}
	} else if(r2 - floor(r2) > 0.5f) {
		// h = add(h, o2);
		if(r1 < 0) {
			h = Hexint_init(3, 1);
		} else {
			h = Hexint_init(15, 1);
		}
	}

	h = add(h, mul_int(o1, (int)abs(trunc(r1))));
	h = add(h, mul_int(o2, (int)abs(trunc(r2))));

	return h;
}

// returns a 3-tuple using hers coord system
fPoint3d getHer(Hexint self) {
	fPoint3d c = { .x = 1.0f, .y = 0.0f, .z = -1.0f };
	fPoint3d p = { .x = 0.0f, .y = 0.0f, .z = 0.0f };
	unsigned int value = self.numstr;
	unsigned int tmp;

	for(int i = self.ndigits - 1; i > -1; i--) {
		// compute key points
		if(i < self.ndigits - 1) {
			const fPoint2d tempp = { .x = c.x, .y = c.y };

			c.x = (4 * c.x - 5 * c.y + c.z) / 3;
			c.y = (tempp.x + 4 * c.y - 5 * c.z) / 3;
			c.z = (-5 * tempp.x + tempp.y + 4 * c.z) / 3;
		}

		// compute the rotation
		tmp = value % 10;
		if(tmp == 1) {
			p.x = p.x + c.x;
			p.y = p.y + c.y;
			p.z = p.z + c.z;
		} else if(tmp == 2) {
			p.x = p.x - c.y;
			p.y = p.y - c.z;
			p.z = p.z - c.x;
		} else if(tmp == 3) {
			p.x = p.x + c.z;
			p.y = p.y + c.x;
			p.z = p.z + c.y;
		} else if(tmp == 4) {
			p.x = p.x - c.x;
			p.y = p.y - c.y;
			p.z = p.z - c.z;
		} else if(tmp == 5) {
			p.x = p.x + c.y;
			p.y = p.y + c.z;
			p.z = p.z + c.x;
		} else if(tmp == 6) {
			p.x = p.x - c.z;
			p.y = p.y - c.x;
			p.z = p.z - c.y;
		}
		value = value / 10;
	}

	return p;
}

// Multiplikation mit Hexint
Hexint mul_Hexint(Hexint self, Hexint object) {
	const unsigned int M[7][7] =
		{ { 0, 0, 0, 0, 0, 0, 0 },
		  { 0, 1, 2, 3, 4, 5, 6 },
		  { 0, 2, 3, 4, 5, 6, 1 },
		  { 0, 3, 4, 5, 6, 1, 2 },
		  { 0, 4, 5, 6, 1, 2, 3 },
		  { 0, 5, 6, 1, 2, 3, 4 },
		  { 0, 6, 1, 2, 3, 4, 5 } };

	char* numa;
	char* numb;
	unsigned int maxlen;
	unsigned int value_self = self.numstr;
	unsigned int value_object = object.numstr;
	unsigned int tmp_self = 1;
	unsigned int tmp_object = 1;
	unsigned int* powers;
	Hexint sum = Hexint_init(0, 1);
	unsigned int mul = 1;

	if((int)(self.ndigits - object.ndigits) > 0) {
		maxlen = self.ndigits + 1;
	} else {
		maxlen = object.ndigits + 1;
	}

	numa = (char*)malloc(maxlen * sizeof(char));
	numb = (char*)malloc(maxlen * sizeof(char));

	// pad out with zeros to make strs same length
	for(int i = maxlen - 1; i > -1; i--) {
		if(maxlen - self.ndigits) {
			tmp_self = value_self % 10;
		}
		numa[i] = (char)('0' + tmp_self);

		if(maxlen - object.ndigits) {
			tmp_object = value_object % 10;
		}
		numb[i] = (char)('0' + tmp_object);

		if(value_self) {
			value_self = value_self / 10;
		}
		if(value_object) {
			value_object = value_object / 10;
		}
	}

	powers = (unsigned int*)malloc(maxlen * sizeof(unsigned int));
	for(unsigned int i = 0; i < maxlen; i++) {
		powers[i] = mul;
		if(i < maxlen - 1) {
			mul = mul * 10;
		}
	}
	mul = 1;

	for(unsigned int i = 0; i < maxlen; i++) {
		const unsigned int ii = maxlen - i - 1;
		unsigned int partial = 0;

		mul = powers[i];

		for(unsigned int j = 0; j < maxlen; j++) {
			const unsigned int jj = maxlen - j - 1;

			tmp_self = (unsigned int)(numa[ii] - '0');

			if(tmp_self) {
				partial = partial + M[tmp_self][(int)(numb[jj] - '0')] * mul;
			}
			mul = mul * 10;
		}
		sum = add(sum, Hexint_init(partial, 1));
	}

	free(numa);
	free(numb);
	free(powers);

	return sum;
}

// Subtraktion
Hexint sub(Hexint self, Hexint object) {
	return add(self, neg(object));
}

// return spatial integer coord pair
fPoint2d getSpatial(Hexint self) {
	fPoint2d p2;
	const fPoint3d p3 = getHer(self);

	p2.x = (p3.x + p3.y - 2 * p3.z) / 3;
	p2.y = (-p3.x + 2 * p3.y - p3.z) / 3;

	return p2;
}

// return fequency integer coord pair
fPoint2d getFrequency(Hexint self) {
	fPoint2d p2;
	const fPoint3d p3 = getHer(self);

	p2.x = (-p3.x + 2 * p3.y - p3.z) / 3;
	p2.y = (2 * p3.x - p3.y - p3.z) / 3;

	return p2;
}

// return integer coord of skewed 60 degree axis
fPoint2d getSkew(Hexint self) {
	fPoint2d p2;
	const fPoint3d p3 = getHer(self);

	p2.x = (2 * p3.x - p3.y - p3.z) / 3;
	p2.y = (-p3.x + 2 * p3.y - p3.z) / 3;

	return p2;
}

