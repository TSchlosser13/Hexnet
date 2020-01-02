#!/usr/bin/env python3.7


'''****************************************************************************
 * offsets.py: TODO
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


def build_offsets(mask_shape=(3, 3)):
	offsets_even_rows = np.empty(shape=(2, 2), dtype=np.int32)
	offsets_odd_rows  = np.empty(shape=(2, 2), dtype=np.int32)

	mask_size          = min(mask_shape)
	block_side_length  = int(mask_size / 2) + mask_size % 2
	block_size         = 2 * block_side_length - 1
	center             = int(block_size / 2)
	center_is_even_row = 1 - center % 2

	offsets_even_rows[0] = (block_size - 1,                  -(block_side_length - 1))
	offsets_odd_rows[0]  = (block_size - center_is_even_row, -(block_side_length - 1))
	offsets_even_rows[1] = (-1, block_size)
	offsets_odd_rows[1]  = ( 0, block_size)

	return (offsets_even_rows, offsets_odd_rows)


def test_build_offsets(mask_shape=(3, 3)):
	print(f'test_build_offsets: mask_shape={mask_shape}')

	(offsets_even_rows, offsets_odd_rows) = build_offsets(mask_shape)

	print(f'offsets_even_rows =\n{offsets_even_rows}\noffsets_odd_rows =\n{offsets_odd_rows}')


if __name__ == '__main__':
	for i in (3, 5, 7):
		test_build_offsets(mask_shape=(i, i))

