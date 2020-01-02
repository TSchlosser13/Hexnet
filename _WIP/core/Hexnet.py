'''****************************************************************************
 * Hexnet.py: TODO
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


import os

from ctypes import *
from glob   import glob
from tqdm   import tqdm

from misc.misc import print_newline


Hexnet = None


class Array(Structure):
	_fields_ = [
		('width',     c_uint32),
		('height',    c_uint32),
		('depth',     c_uint32),
		('pixels',    c_uint32),
		('size',      c_uint32),
		('len',       c_float),
		('width_sq',  c_float),
		('height_sq', c_float),
		('p',         POINTER(c_uint8))
	]

class Hexarray(Structure):
	_fields_ = [
		('width',      c_uint32),
		('height',     c_uint32),
		('depth',      c_uint32),
		('pixels',     c_uint32),
		('size',       c_uint32),
		('rad_o',      c_float),
		('rad_i',      c_float),
		('dia_o',      c_float),
		('dia_i',      c_float),
		('dist_w',     c_float),
		('dist_h',     c_float),
		('width_hex',  c_float),
		('height_hex', c_float),
		('p',          POINTER(c_uint8))
	]


def Hexnet_load(shared_object='../Hexnet.so'):
	global Hexnet
	Hexnet = CDLL(shared_object)

def Hexnet_init():
	Hexnet_load()
	Hexnet.print_info()
	print_newline()


def _Hexsamp_s2h(filename, output_dir='.', rad_o=1.0, method=0, increase_verbosity=False):
	filename_in  = bytes(filename, encoding='ascii')
	filename_out = os.path.basename(filename).split('.')
	filename_out = '.'.join(filename_out[:-1]) + f'_s2h.{filename_out[-1]}'
	filename_out = os.path.join(output_dir, filename_out)
	filename_out = bytes(filename_out, encoding='ascii')

	array      = Array()
	array_p    = byref(array)
	hexarray   = Hexarray()
	hexarray_p = byref(hexarray)
	rad_o      = c_float(rad_o)
	method     = c_uint32(method)


	Hexnet.file_to_Array(filename_in, array_p)

	if increase_verbosity:
		Hexnet.Array_print_info(array, b'array')

	Hexnet.Hexarray_init_from_Array(hexarray_p, array, rad_o)

	if increase_verbosity:
		Hexnet.Hexarray_print_info(hexarray, b'hexarray')

	Hexnet.Hexsamp_s2h(array, hexarray_p, method)

	Hexnet.Hexarray_to_file(hexarray, filename_out)

	Hexnet.Array_free(array_p)
	Hexnet.Hexarray_free(hexarray_p)


def _Sqsamp_s2s(filename, output_dir='.', width=64, height=None, method=0, increase_verbosity=False):
	filename_in  = bytes(filename, encoding='ascii')
	filename_out = os.path.basename(filename).split('.')
	filename_out = '.'.join(filename_out[:-1]) + f'_s2s.{filename_out[-1]}'
	filename_out = os.path.join(output_dir, filename_out)
	filename_out = bytes(filename_out, encoding='ascii')

	s1     = Array()
	s1_p   = byref(s1)
	s2     = Array()
	s2_p   = byref(s2)
	width  = c_uint32(width)
	method = c_uint32(method)

	if height is None:
		height = width
	else:
		height = c_uint32(height)

	depth = c_uint32(3)
	len   = c_float(1.0)


	Hexnet.file_to_Array(filename_in, s1_p)

	if increase_verbosity:
		Hexnet.Array_print_info(s1, b's1')

	Hexnet.Array_init(s2_p, width, height, depth, len)

	if increase_verbosity:
		Hexnet.Array_print_info(s2, b's2')

	Hexnet.Sqsamp_s2s(s1, s2_p, method)

	Hexnet.Array_to_file(s2, filename_out)

	Hexnet.Array_free(s1_p)
	Hexnet.Array_free(s2_p)


def Hexsamp_s2h(filename_s, output_dir='.', rad_o=1.0, method=0, increase_verbosity=False):
	print(f'Hexsamp_s2h (rad_o={rad_o}, method={method}) for filename in {filename_s} to {output_dir}')

	for filename in tqdm(glob(filename_s)):
		_Hexsamp_s2h(filename, output_dir, rad_o, method, increase_verbosity)


def Sqsamp_s2s(filename_s, output_dir='.', width=64, height=None, method=0, increase_verbosity=False):
	print(f'Sqsamp_s2s (width={width}, height={height}, method={method}) for filename in {filename_s} to {output_dir}')

	for filename in tqdm(glob(filename_s)):
		_Sqsamp_s2s(filename, output_dir, width, height, method, increase_verbosity)

