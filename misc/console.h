/******************************************************************************
 * console.h: Console Escape Sequences
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


#ifndef CONSOLE_H
#define CONSOLE_H


// Set Graphics Mode

// Text attributes
#define SGM_TA_RESET   "\x1B[0m"

// Foreground colors
#define SGM_FC_BLACK   "\x1B[30m"
#define SGM_FC_RED     "\x1B[31m"
#define SGM_FC_GREEN   "\x1B[32m"
#define SGM_FC_YELLOW  "\x1B[33m"
#define SGM_FC_BLUE    "\x1B[34m"
#define SGM_FC_MAGENTA "\x1B[35m"
#define SGM_FC_CYAN    "\x1B[36m"
#define SGM_FC_WHITE   "\x1B[37m"


#endif

