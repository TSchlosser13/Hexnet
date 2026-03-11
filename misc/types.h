/******************************************************************************
 * types.h: Data Types
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


#ifndef TYPES_H
#define TYPES_H


/******************************************************************************
 * Includes
 ******************************************************************************/

#include <stdbool.h>
#include <stdint.h>

#include <epoxy/gl.h>
#include <gtk/gtk.h>


/******************************************************************************
 * Defines
 ******************************************************************************/

#ifndef i8
	#define i8  int8_t
#endif

#ifndef u8
	#define u8  uint8_t
#endif

#ifndef i32
	#define i32 int32_t
#endif

#ifndef u32
	#define u32 uint32_t
#endif




/******************************************************************************
 * Typedefs
 ******************************************************************************/


/******************************************************************************
 * Hexint
 ******************************************************************************/

typedef u32 Hexint;


/******************************************************************************
 * Array and Hexarray
 ******************************************************************************/

typedef struct {
	u32   width;
	u32   height;
	u32   depth;
	u32   pixels;
	u32   size;
	float len;
	float width_sq;
	float height_sq;
	u8*   p;
} Array;

typedef struct {
	u32   width;
	u32   height;
	u32   depth;
	u32   pixels;
	u32   size;
	float rad_o;
	float rad_i;
	float dia_o;
	float dia_i;
	float dist_w;
	float dist_h;
	float width_hex;
	float height_hex;
	u8*   p;
} Hexarray;


/******************************************************************************
 * gl_area, gl_context, and gui
 ******************************************************************************/

typedef struct {
	gint   width;
	gint   height;
	bool   button_is_pressed;
	gfloat mouse_x;
	gfloat mouse_x_old;
	gfloat mouse_delta_x;
	gfloat mouse_y;
	gfloat mouse_y_old;
	gfloat mouse_delta_y;
} gl_area;

typedef struct {
	GLuint  program;
	GLuint  vao;
	GLuint  mvp;
	GLfloat mvp_matrix[16];
	GLfloat zoom;
} gl_context;

typedef struct {
	GtkWidget* window;
	char*      title;
	u32        width;
	u32        height;
	Hexarray   hexarray;
	gl_area    gl_area;
	gl_context gl_context;
} gui;




#endif

