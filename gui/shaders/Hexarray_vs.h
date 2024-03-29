/******************************************************************************
 * Hexarray_vs.h: Vertex Shader for the Visualization of Hexagonal Arrays
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


#ifndef HEXARRAY_VS_H
#define HEXARRAY_VS_H


/******************************************************************************
 * Vertex Shader for the Visualization of Hexagonal Arrays
 ******************************************************************************/

const char* Hexarray_vs = \
	"#version 330                                                           \n" \
	"                                                                       \n" \
	"layout(location = 0) in vec2 position;                                 \n" \
	"layout(location = 1) in vec2 position_offset;                          \n" \
	"layout(location = 2) in vec3 color;                                    \n" \
	"                                                                       \n" \
	"uniform mat4 mvp;                                                      \n" \
	"                                                                       \n" \
	"out vec3 color_vs;                                                     \n" \
	"                                                                       \n" \
	"void main() {                                                          \n" \
	"    gl_Position = mvp * vec4(position + position_offset, 1.0f, 1.0f);  \n" \
	"    color_vs    = color;                                               \n" \
	"}                                                                      \n";


#endif


