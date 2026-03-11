/******************************************************************************
 * precalculations.c: Vorberechnungen
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

#include "defines.h"
#include "types.h"

#include "../core/Hexint.h"

#include "precalculations.h"


// TODO: Notation

float**  pc_reals = NULL;
iPoint2d pc_rmn   = { .x = 0, .y = 0 }; // pc_reals_min
iPoint2d pc_rmx   = { .x = 0, .y = 0 }; // pc_reals_max

unsigned int** pc_spatials      = NULL;
uPoint2d       pc_spatials_size = { .x = 0, .y = 0 };
iPoint2d       pc_smn           = { .x = 0, .y = 0 }; // pc_spatials_min
iPoint2d       pc_smx           = { .x = 0, .y = 0 }; // pc_spatials_max

unsigned int** pc_nearest = NULL;

unsigned int** pc_adds          = NULL;
uPoint3d*      pc_adds_mul      = NULL;
unsigned int   pc_adds_mul_size = 0;


void precalcs_init(unsigned int order, float scale, float radius) {
	const unsigned int size  = pow(7, order);
	const unsigned int size7 = size * 7;

	Hexint   hi;
	fPoint2d pr;
	fPoint2d ps;




	for(unsigned int i = 0; i < size; i++) {
		hi = Hexint_init(i, 0);

		pr = getReal(hi);
		ps = getSpatial(hi);


		if(pr.x < pc_rmn.x) {
			pc_rmn.x = (int)roundf(pr.x);
		} else if(pr.x > pc_rmx.x) {
			pc_rmx.x = (int)roundf(pr.x);
		}
		if(pr.y < pc_rmn.y) {
			pc_rmn.y = (int)roundf(pr.y);
		} else if(pr.y > pc_rmx.y) {
			pc_rmx.y = (int)roundf(pr.y);
		}

		if(ps.x < pc_smn.x) {
			pc_smn.x = (int)ps.x;
		} else if(ps.x > pc_smx.x) {
			pc_smx.x = (int)ps.x;
		}
		if(ps.y < pc_smn.y) {
			pc_smn.y = (int)ps.y;
		} else if(ps.y > pc_smx.y) {
			pc_smx.y = (int)ps.y;
		}
	}

	pc_spatials_size.x = pc_smx.x - pc_smn.x + 1;
	pc_spatials_size.y = pc_smx.y - pc_smn.y + 1;




	const uPoint2d size_out = {
		.x = (unsigned int)roundf((pc_rmx.x - pc_rmn.x) / scale) + 1,
		.y = (unsigned int)roundf((pc_rmx.y - pc_rmn.y) / scale) + 1 };

	const unsigned int i_max = radius > 1.0f ? 49 : 7; // TODO?


	pc_reals = (float**)malloc(size7 * sizeof(float*));

	for(unsigned int i = 0; i < size7; i++)
		pc_reals[i] = (float*)malloc(2 * sizeof(float));


	pc_spatials = (unsigned int**)malloc(pc_spatials_size.x * sizeof(unsigned int*));

	for(unsigned int i = 0; i < pc_spatials_size.x; i++)
		pc_spatials[i] = (unsigned int*)calloc(pc_spatials_size.y, sizeof(unsigned int));


	pc_nearest = (unsigned int**)malloc(size_out.x * sizeof(unsigned int*));

	for(unsigned int i = 0; i < size_out.x; i++)
		pc_nearest[i] = (unsigned int*)malloc(size_out.y * sizeof(unsigned int));


	pc_adds = (unsigned int**)malloc(size7 * sizeof(unsigned int*));

	for(unsigned int i = 0; i < size7; i++)
		pc_adds[i] = (unsigned int*)malloc(i_max * sizeof(unsigned int));




	for(unsigned int i = 0; i < size7; i++) {
		pr = getReal(Hexint_init(i, 0));

		pc_reals[i][0] = pr.x;
		pc_reals[i][1] = pr.y;
	}

	for(unsigned int i = 0; i < size; i++) {
		ps = getSpatial(Hexint_init(i, 0));

		pc_spatials[(int)(ps.x - pc_smn.x)][(int)(ps.y - pc_smn.y)] = i;
	}

	for(unsigned int i = 0; i < size7; i++) {
		const Hexint base = Hexint_init(i, 0);

		for(unsigned int j = 0; j < i_max; j++)
			pc_adds[i][j] = getInt(add(base, Hexint_init(j, 0)));
	}




	FILE* adds_mul = NULL;


	// TODO?

	if(order <= 5) {
		adds_mul = fopen("_LUTs/Hexint_additions_order_5.csv", "r");
	} else {
		char filename[64];

		sprintf(filename, "_LUTs/Hexint_additions_order_%u.csv", order);

		adds_mul = fopen(filename, "r");
	}

	if(adds_mul == NULL) {
		printf("\n > Error - precalcs_init: Precalcs missing (order=%u)\n", order);
		exit(EXIT_FAILURE);
	}


	while(!feof(adds_mul)) {
		pc_adds_mul = (uPoint3d*)realloc(pc_adds_mul, (pc_adds_mul_size + 1) * sizeof(uPoint3d));

		fscanf(adds_mul, "%u %u %u", &pc_adds_mul[pc_adds_mul_size].x, \
			&pc_adds_mul[pc_adds_mul_size].y, &pc_adds_mul[pc_adds_mul_size].z);

		pc_adds_mul_size++;
	}

	fclose(adds_mul);




	for(unsigned int i = 0; i < size_out.y; i++) {
		for(unsigned int j = 0; j < size_out.x; j++) {
			pc_nearest[j][i] = getInt(Hexint_init(
				getNearest(pc_rmn.x + j * scale, pc_rmn.y + i * scale).numstr, 1));
		}
	}
}

void precalcs_free() {
	for(unsigned int i = 0; i < SIZEOF_ARRAY(pc_reals);    i++)
		free(pc_reals[i]);

	free(pc_reals);


	for(unsigned int i = 0; i < SIZEOF_ARRAY(pc_spatials); i++)
		free(pc_spatials[i]);

	free(pc_spatials);


	for(unsigned int i = 0; i < SIZEOF_ARRAY(pc_nearest);  i++)
		free(pc_nearest[i]);

	free(pc_nearest);




	for(unsigned int i = 0; i < SIZEOF_ARRAY(pc_adds);     i++)
		free(pc_adds[i]);

	free(pc_adds);


	free(pc_adds_mul);
}

