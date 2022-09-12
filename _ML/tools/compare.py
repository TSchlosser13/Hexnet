#!/usr/bin/env python3.7


'''****************************************************************************
 * compare.py: Array Compare Methods
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


import math

import numpy      as np
import tensorflow as tf


_COMPARE_AE    =  0
_COMPARE_SE    =  1
_COMPARE_MAE   =  2
_COMPARE_MSE   =  3
_COMPARE_RMSE  =  4
_COMPARE_PSNR  =  5
_COMPARE_ERROR = -1


def get_compare_method(method):
	if method == 'AE':
		return _COMPARE_AE
	elif method == 'SE':
		return _COMPARE_SE
	elif method == 'MAE':
		return _COMPARE_MAE
	elif method == 'MSE':
		return _COMPARE_MSE
	elif method == 'RMSE':
		return _COMPARE_RMSE
	elif method == 'PSNR':
		return _COMPARE_PSNR
	else:
		return _COMPARE_ERROR


def ae(p1, p2):
	if not tf.is_tensor(p1):
		ae = np.sum(np.abs(p1 - p2), axis=-1)
	else:
		ae = tf.reduce_sum(tf.abs(p1 - p2), axis=-1)

	return ae

def se(p1, p2):
	diff = p1 - p2

	if not tf.is_tensor(p1):
		se = np.sum(np.multiply(diff, diff), axis=-1)
	else:
		se = tf.reduce_sum(tf.multiply(diff, diff), axis=-1)

	return se

def mae(p1, p2):
	return ae(p1, p2) / p1.size

def mse(p1, p2):
	return se(p1, p2) / p1.size

def rmse(p1, p2):
	return math.sqrt(mse(p1, p2))

def psnr(p1, p2):
	if not tf.is_tensor(p1):
		psnr = 10 * np.log(65025 / mse(p1, p2))
	else:
		psnr = 10 * tf.math.log(1 / mse(p1, p2))

	return psnr


def col_changed(c1, c2):             return int(c1) != int(c2) and (c2) != int(c2)
def row_changed(r1, r2):             return int(r1) != int(r2) and (r2) != int(r2)
def col_row_changed(c1, c2, r1, r2): return col_changed(c1, c2) and row_changed(r1, r2)

def pixels_differ(p1, p2):
	if not tf.is_tensor(p1):
		pixels_differ = not np.equal(p1, p2)
	else:
		pixels_differ = not tf.equal(p1, p2)

	return pixels_differ

def pixels_diff(p1, p2, method):
	if method == _COMPARE_AE or method == _COMPARE_MAE:
		return ae(p1, p2)
	elif method == _COMPARE_SE   or \
	     method == _COMPARE_MSE  or \
	     method == _COMPARE_RMSE or \
	     method == _COMPARE_PSNR :
		return se(p1, p2)
	else:
		return _COMPARE_ERROR


def _compare_s2s(s1, s2, method, s1_shape=None, s2_shape=None):
	s1_is_tensor = tf.is_tensor(s1)
	s2_is_tensor = tf.is_tensor(s2)

	if s1_is_tensor or s2_is_tensor:
		if not s1_is_tensor:
			s1 = tf.convert_to_tensor(s1, dtype=tf.float32)

		if not s2_is_tensor:
			s2 = tf.convert_to_tensor(s2, dtype=tf.float32)

		s12_is_tensor = True

		s1_height = s1_shape[0]
		s2_height = s2_shape[0]
		s1_width  = s1_shape[1]
		s2_width  = s2_shape[1]
		depth     = s1_shape[2]
		size      = depth * s1_width * s1_height
	else:
		if type(s1) is not np.ndarray:
			s1 = np.asarray(s1)

		if type(s2) is not np.ndarray:
			s2 = np.asarray(s2)

		s12_is_tensor = False

		s1_height = s1.shape[1]
		s2_height = s2.shape[1]
		s1_width  = s1.shape[2]
		s2_width  = s2.shape[2]
		depth     = s1.shape[3]
		size      = s1.size


	areas = []
	diffs = []


	if s1_width > s2_width:
		width_min = s2_width
		width_max = s1_width
		p1_c      = 0
		p2_c      = 1
	else:
		width_min = s1_width
		width_max = s2_width
		p1_c      = 1
		p2_c      = 0

	p1w_cr = p1_c
	p2w_cr = p2_c

	if s1_height > s2_height:
		height_min = s2_height
		height_max = s1_height
		p1_r       = 0
		p2_r       = 1
	else:
		height_min = s1_height
		height_max = s2_height
		p1_r       = 1
		p2_r       = 0

	p1h_cr = p1_r
	p2h_cr = p2_r


	ws_min = width_min  / width_max
	ws_max = width_max  / width_min
	hs_min = height_min / height_max
	hs_max = height_max / height_min


	for h_max in range(height_max):
		h_min     = h_max * hs_min
		h_min_max = int(h_min + 1) * hs_max

		for w_max in range(width_max):
			w_min     = w_max * ws_min
			w_min_max = int(w_min + 1) * ws_max


			if s1_width > s2_width:
				p1w = w_max
				p2w = int(w_min)
			else:
				p1w = int(w_min)
				p2w = w_max

			if s1_height > s2_height:
				p1h = h_max
				p2h = int(h_min)
			else:
				p1h = int(h_min)
				p2h = h_max


			areas.append((min(w_max + 1, w_min_max) - w_max) * (min(h_max + 1, h_min_max) - h_max))
			diffs.append(pixels_diff(s1[:, p1h, p1w, :], s2[:, p2h, p2w, :], method))

			if col_changed(w_min, w_min + ws_min):
				areas.append((w_max + 1 - w_min_max) * (min(h_max + 1, h_min_max) - h_max))
				diffs.append(pixels_diff(s1[:, p1h, p1w+p1_c, :], s2[:, p2h, p2w+p2_c, :], method))

			if row_changed(h_min, h_min + hs_min):
				areas.append((min(w_max + 1, w_min_max) - w_max) * (h_max + 1 - h_min_max))
				diffs.append(pixels_diff(s1[:, p1h+p1_r, p1w, :], s2[:, p2h+p2_r, p2w, :], method))

			if col_row_changed(w_min, w_min + ws_min, h_min, h_min + hs_min):
				areas.append((w_max + 1 - w_min_max) * (h_max + 1 - h_min_max))
				diffs.append(pixels_diff(s1[:, p1h+p1h_cr, p1w+p1w_cr, :], s2[:, p2h+p2h_cr, p2w+p2w_cr, :], method))


	if not s12_is_tensor:
		areas  = np.asarray(areas)
		diffs  = np.sum(np.asarray(diffs), axis=-1)
		result = np.sum(np.multiply(areas, diffs))
	else:
		areas  = tf.convert_to_tensor(areas, dtype=tf.float32)
		diffs  = tf.reduce_sum(tf.stack(diffs), axis=-1)
		result = tf.reduce_sum(tf.multiply(areas, diffs))


	result *= min(1, s1_width / s2_width) * min(1, s1_height / s2_height)

	if method == _COMPARE_MAE  or \
	   method == _COMPARE_MSE  or \
	   method == _COMPARE_RMSE or \
	   method == _COMPARE_PSNR :
		result /= size

	if method == _COMPARE_RMSE:
		result = math.sqrt(result)
	elif method == _COMPARE_PSNR:
		if not s12_is_tensor:
			result = 10 * math.log(65025 / result)
		else:
			result = 10 * tf.math.log(1 / result)


	return result


def _compare_s2h(s, h, method, s_shape=None, h_shape=None):
	s_is_tensor = tf.is_tensor(s)
	h_is_tensor = tf.is_tensor(h)

	if s_is_tensor or h_is_tensor:
		if not s_is_tensor:
			s = tf.convert_to_tensor(s, dtype=tf.float32)

		if not h_is_tensor:
			h = tf.convert_to_tensor(h, dtype=tf.float32)

		sh_is_tensor = True

		s_height = s_shape[0]
		h_height = h_shape[0]
		s_width  = s_shape[1]
		h_width  = h_shape[1]
		depth    = s_shape[2]
		s_size   = depth * s_width * s_height
		h_size   = depth * h_width * h_height
	else:
		if type(s) is not np.ndarray:
			s = np.asarray(s)

		if type(h) is not np.ndarray:
			h = np.asarray(h)

		sh_is_tensor = False

		s_height = s.shape[1]
		h_height = h.shape[1]
		s_width  = s.shape[2]
		h_width  = h.shape[2]
		depth    = s.shape[3]
		s_size   = s.size
		h_size   = h.size


	areas = []
	diffs = []


	width_sq  = s_width
	height_sq = s_height

	if h_height > 1:
		h_rad_i_w = (s_width / (h_width - 0.5)) / 2
		h_rad_i_h = ((s_height / (h_height - 0.5)) / 2) / (math.sqrt(3) / 2)
	else:
		h_rad_i_w = (s_width / h_width) / 2
		h_rad_i_h = (s_height / 2) / (math.sqrt(3) / 2)

	h_rad_i = max(h_rad_i_w, h_rad_i_h)

	h_rad_o  = h_rad_i / (math.sqrt(3) / 2)
	h_dia_o  = 2 * h_rad_o
	h_dia_i  = 2 * h_rad_i
	h_dist_w = h_dia_i
	h_dist_h = 1.5 * h_rad_o

	if h_height > 1:
		width_hex  = h_width * h_dia_i + h_rad_i
		height_hex = h_dia_o + (h_height - 1) * h_dist_h
	else:
		width_hex  = h_width * h_dia_i
		height_hex = h_dia_o


	wb = (width_hex  - width_sq)  / 2
	hb = (height_hex - height_sq) / 2

	ws_s = 1.0
	ws_h = h_rad_i
	hs_s = 1.0
	hs_h = h_rad_o / 2
	sr_h = hs_h / ws_h


	wi = 0.0
	hi = 0.0

	while hi < height_sq:
		hh   = hb + hi
		hhm  = hh % hs_h
		hhmr = hs_h - hhm

		hm      = hi % hs_s
		hmr     = hs_s - hm
		hmr_min = min(hmr, hhmr)

		hd = hh % h_dist_h
		hn = max(hd + hmr_min, np.nextafter(hd, hd + 1))

		hu  = int(hi / hs_s)
		hhu = int(hh / h_dist_h)


		while wi < width_sq:
			wh   = wb + wi if not hhu % 2 else wb - h_rad_i + wi
			whm  = wh % ws_h
			whmr = ws_h - whm

			wm      = wi % ws_s
			wmr     = ws_s - wm
			wmr_min = min(wmr, whmr)

			wd = wh % h_dist_w
			wn = max(wd + wmr_min, np.nextafter(wd, wd + 1))

			wu  = int(wi / ws_s)
			whu = int(wh / h_dist_w)


			if hd < hs_h - sr_h * wd:
				w_min = max(wd, (hs_h - hn) / sr_h)
				w_max = min((hs_h - hd) / sr_h, wn)
				h_min = hs_h - sr_h * w_max
				h_max = hs_h - sr_h * w_min

				whuc = whu if hhu % 2 else whu - 1
				hhuc = hhu - 1
				ps   = depth * (hu   * s_width + wu)
				ph   = depth * (hhu  * h_width + whu)
				phc  = depth * (hhuc * h_width + whuc)


				if h_min < hn:
					area = wmr_min * (h_min - hd) + ((w_max - w_min) * (h_max - h_min)) / 2

					if phc >= 0:
						areas.append(area)
						diffs.append(pixels_diff(s[:, hu, wu, :], h[:, hhuc, whuc, :], method))

					areas.append(wmr_min * hmr_min - area)
					diffs.append(pixels_diff(s[:, hu, wu, :], h[:, hhu, whu, :], method))
				else:
					if phc >= 0:
						areas.append(wmr_min * hmr_min)
						diffs.append(pixels_diff(s[:, hu, wu, :], h[:, hhuc, whuc, :], method))
			elif hd < hs_h - sr_h * (h_dist_w - wn):
				w_min = max(wd, (hd + hs_h) / sr_h)
				w_max = min((hn + hs_h) / sr_h, wn)
				h_min = hs_h - sr_h * (h_dist_w - w_min)
				h_max = hs_h - sr_h * (h_dist_w - w_max)

				whuc = whu + 1 if hhu % 2 else whu
				hhuc = hhu - 1
				ps   = depth * (hu   * s_width + wu)
				ph   = depth * (hhu  * h_width + whu)
				phc  = depth * (hhuc * h_width + whuc)


				if h_min < hn:
					area = wmr_min * (h_min - hd) + ((w_max - w_min) * (h_max - h_min)) / 2

					if phc >= 0 and phc < h_size:
						areas.append(area)
						diffs.append(pixels_diff(s[:, hu, wu, :], h[:, hhuc, whuc, :], method))

					areas.append(wmr_min * hmr_min - area)
					diffs.append(pixels_diff(s[:, hu, wu, :], h[:, hhu, whu, :], method))
				else:
					if phc >= 0 and phc < h_size:
						areas.append(wmr_min * hmr_min)
						diffs.append(pixels_diff(s[:, hu, wu, :], h[:, hhuc, whuc, :], method))
			else:
				ps = depth * (hu  * s_width + wu)
				ph = depth * (hhu * h_width + whu)

				areas.append(wmr_min * hmr_min)
				diffs.append(pixels_diff(s[:, hu, wu, :], h[:, hhu, whu, :], method))


			wi = max(wi + wmr_min, np.nextafter(wi + wmr_min, wi + wmr_min + 1))


		wi = 0.0
		hi = max(hi + hmr_min, np.nextafter(hi + hmr_min, hi + hmr_min + 1))


	if not sh_is_tensor:
		areas  = np.asarray(areas)
		diffs  = np.sum(np.asarray(diffs), axis=-1)
		result = np.sum(np.multiply(areas, diffs), axis=-1)
	else:
		areas  = tf.convert_to_tensor(areas, dtype=tf.float32)
		diffs  = tf.reduce_sum(tf.stack(diffs), axis=-1)
		result = tf.reduce_sum(tf.multiply(areas, diffs), axis=-1)


	if method == _COMPARE_MAE  or \
	   method == _COMPARE_MSE  or \
	   method == _COMPARE_RMSE or \
	   method == _COMPARE_PSNR :
		result /= s_size

	if method == _COMPARE_RMSE:
		result = math.sqrt(result)
	elif method == _COMPARE_PSNR:
		if not sh_is_tensor:
			result = 10 * math.log(65025 / result)
		else:
			result = 10 * tf.math.log(1 / result)


	return result




def test_compare_s2s():
	print(f'>> test_compare_s2s')

	s1_shape = (1, 4, 4, 3)
	s2_shape = (1, 2, 2, 3)

	s1     = np.zeros(s1_shape)
	s2     = np.zeros(s2_shape)
	method = 'AE'

	s2[0][0][0][0] = 1

	print(f's1 =\n{s1}\ns2 =\n{s2}\nmethod={method}')

	result          = _compare_s2s(s1, s2, method=get_compare_method(method))
	expected_result = 4.0
	test_passed     = 'test passed' if math.isclose(result, expected_result) else 'test not passed'

	print(f'> result={result:.8f} (expected_result={expected_result}) -> {test_passed}')


def test_compare_s2h():
	print(f'>> test_compare_s2h')

	s_shape = (1, 4, 4, 3)
	h_shape = (1, 2, 2, 3)

	s      = np.zeros(s_shape)
	h      = np.zeros(h_shape)
	method = 'AE'

	s[0][0][0][0] = 1

	print(f's =\n{s}\nh =\n{h}\nmethod={method}')

	result          = _compare_s2h(s, h, method=get_compare_method(method))
	expected_result = 1.0
	test_passed     = 'test passed' if math.isclose(result, expected_result) else 'test not passed'

	print(f'> result={result:.8f} (expected_result={expected_result}) -> {test_passed}')


if __name__ == '__main__':
	test_compare_s2s()
	test_compare_s2h()

