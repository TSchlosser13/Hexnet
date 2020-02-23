#!/usr/bin/env python3.7


'''****************************************************************************
 * masks.py: Masks for Hexagonal Layers
 ******************************************************************************
 * v0.1 - 01.03.2019
 *
 * Copyright (c) 2019 Tobias Schlosser (tobias@tobias-schlosser.net)
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
 ****************************************************************************'''


import numpy as np


def build_masks(mask_shape=(3, 3)):
	mask_even_rows = np.zeros(mask_shape, dtype=np.int32)
	mask_odd_rows  = np.zeros(mask_shape, dtype=np.int32)

	mask_size          = min(mask_shape)
	block_side_length  = int(mask_size / 2) + mask_size % 2
	block_size         = 2 * block_side_length - 1
	center             = int(block_size / 2)
	center_is_even_row = 1 - center % 2

	for row in range(block_size):
		d      = abs(center - row)
		offset = int((d + 1) / 2)
		shift  = d % 2

		if center_is_even_row:
			mask_even_rows[row][offset - shift : block_size - offset] = 1
			mask_odd_rows[row][offset : block_size - offset + shift]  = 1
		else:
			mask_even_rows[row][offset : block_size - offset + shift] = 1
			mask_odd_rows[row][offset - shift : block_size - offset]  = 1

	return (mask_even_rows, mask_odd_rows)


def test_build_masks(mask_shape=(3, 3)):
	print(f'test_build_masks: mask_shape={mask_shape}')

	(mask_even_rows, mask_odd_rows) = build_masks(mask_shape)

	print(f'mask_even_rows =\n{mask_even_rows}\nmask_odd_rows =\n{mask_odd_rows}')


if __name__ == '__main__':
	for i in (3, 5, 7):
		test_build_masks(mask_shape=(i, i))


