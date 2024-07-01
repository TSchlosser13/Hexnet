/******************************************************************************
 * print.c: Console Outputs
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


/******************************************************************************
 * Includes
 ******************************************************************************/

#include <stdio.h>

#include "console.h"
#include "print.h"

#include "../Hexnet.h"


/******************************************************************************
 * Print info, usage, help, and full help
 ******************************************************************************/

void print_info() {
	puts("Hexnet version " HEXNET_VERSION " Copyright (c) " HEXNET_YEAR_S " " HEXNET_AUTHOR);
}

void print_usage() {
	puts("Usage: ./Hexnet [options]");
}

void print_help() {
	puts(SGM_FC_YELLOW "Use -h to get full help." SGM_TA_RESET);
}

void print_help_full() {
	puts( \
		"-h, --help                            print options                                                                                     \n" \
		"-i, --input       <image>             square or hexagonal pixel based input image                                                       \n" \
		"-o, --output      <image>             hexagonal pixel based output image (s2h); h2s, h2h, and s2s output images' base name              \n" \
		"--s2h-rad         <radius>            enable square to hexagonal image transformation by setting the hexagonal pixels' outer radius     \n" \
		"--h2s-len         <length>            enable hexagonal to square image transformation by setting the square pixels' side length         \n" \
		"--h2h-rad         <radius>            enable hexagonal to hexagonal image transformation by setting the hexagonal pixels' outer radius  \n" \
		"--s2s-res         <width> [<height>]  enable square to square image transformation by setting the output resolution                     \n" \
		"--compare-s2s     <image>             compare square input (i, input) to input square image <image>                                     \n" \
		"--compare-s2h                         compare square input (i, input) to hexagonal output image (o, output) using s2h-rad               \n" \
		"--compare-metric  <metric>            compare-s2s and compare-s2h compare metric: AE / SE / MAE / MSE / RMSE / PSNR / SSIM / DSSIM      \n" \
		"-d, --display                         display hexagonal input (i, input) or output image (o, output) using s2h-rad                      \n" \
		"-v, --verbose                         increase verbosity                                                                                ");
}


