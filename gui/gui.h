/******************************************************************************
 * gui.h: Graphical User Interface
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


#ifndef GUI_H
#define GUI_H


#include <gtk/gtk.h>

#include "../misc/types.h"


void gui_gl_area_realize(GtkGLArea* area, gpointer data);
void gui_gl_area_render(GtkGLArea* area, GdkGLContext* context, gpointer data);
void gui_gl_area_resize(GtkGLArea* area, gint width, gint height, gpointer data);
void gui_gl_area_interact(GtkWidget* widget, GdkEvent* event, gpointer data);

void gui_update_title(gui* gui);
void gui_activate(GtkApplication* app, gpointer data);
int  gui_run(char* title, u32 width, u32 height, Hexarray hexarray);


#endif


