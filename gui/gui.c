/******************************************************************************
 * gui.c: Graphical User Interface
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
#include <stdlib.h>

#include <epoxy/gl.h>
#include <gtk/gtk.h>

#include "gui.h"

#include "shaders/Hexagon_positions.h"
#include "shaders/Hexarray_fs.h"
#include "shaders/Hexarray_vs.h"

#include "../misc/defines.h"
#include "../misc/types.h"


/******************************************************************************
 * GUI realization (create the associated resources)
 ******************************************************************************/

void gui_gl_area_realize(GtkGLArea* area, gpointer data) {
	#define SCALE_BORDER_WIDTH     0.05f
	#define SCALE_POSITIONS_FACTOR 1.001f


	gui*        gui        = data;
	Hexarray    hexarray   = gui->hexarray;
	gl_context* gl_context = &gui->gl_context;

	GLfloat positions[24];

	u32      positions_offset_size = 2 * hexarray.pixels;
	u32      colors_size           =     hexarray.size;
	GLfloat* positions_offset      = malloc(positions_offset_size * sizeof(GLfloat));
	GLfloat* colors                = malloc(colors_size           * sizeof(GLfloat));

	GLfloat scale           = 1 - 2 * SCALE_BORDER_WIDTH;
	GLfloat offset_center_w = scale;
	GLfloat offset_center_h = scale;


	gtk_gl_area_make_current(area);


	if(hexarray.width_hex > hexarray.height_hex) {
		scale           /= hexarray.width_hex  / hexarray.dia_o;
		offset_center_w -= scale * M_SQRT3_2;
		offset_center_h -= scale             + (hexarray.width_hex  - hexarray.height_hex) / hexarray.width_hex;
	} else {
		scale           /= hexarray.height_hex / hexarray.dia_o;
		offset_center_w -= scale * M_SQRT3_2 + (hexarray.height_hex - hexarray.width_hex)  / hexarray.height_hex;
		offset_center_h -= scale;
	}

	for(u32 i = 0; i < 24; i++)
		positions[i] = SCALE_POSITIONS_FACTOR * scale * Hexagon_positions[i];

	for(u32 h = 0, i = 0; h < hexarray.height; h++) {
		for(u32 w = 0; w < hexarray.width; w++) {
			if(!(h % 2)) {
				positions_offset[i++] = w * scale * M_SQRT3 - offset_center_w;
			} else {
				positions_offset[i++] = w * scale * M_SQRT3 - offset_center_w + scale * M_SQRT3_2;
			}

			positions_offset[i++] = offset_center_h - h * scale * 1.5f;
		}
	}

	for(u32 p = 0; p < hexarray.size; p++)
		colors[p] = hexarray.p[p] / 255.0f;


	GLuint fs = glCreateShader(GL_FRAGMENT_SHADER);
	GLuint vs = glCreateShader(GL_VERTEX_SHADER);

	glShaderSource(fs, 1, &Hexarray_fs, NULL);
	glShaderSource(vs, 1, &Hexarray_vs, NULL);

	glCompileShader(fs);
	glCompileShader(vs);

	gl_context->program = glCreateProgram();

	glAttachShader(gl_context->program, fs);
	glAttachShader(gl_context->program, vs);

	glLinkProgram(gl_context->program);

	glDetachShader(gl_context->program, fs);
	glDetachShader(gl_context->program, vs);

	glDeleteShader(fs);
	glDeleteShader(vs);


	glGenVertexArrays(1, &gl_context->vao);
	glBindVertexArray(gl_context->vao);


	GLuint vbo_positions;
	GLuint vbo_positions_offset;
	GLuint vbo_colors;

	glGenBuffers(1, &vbo_positions);
	glBindBuffer(GL_ARRAY_BUFFER, vbo_positions);
	glBufferData(GL_ARRAY_BUFFER, 24 * sizeof(GLfloat), positions, GL_STATIC_DRAW);
	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);
	glEnableVertexAttribArray(0);

	glGenBuffers(1, &vbo_positions_offset);
	glBindBuffer(GL_ARRAY_BUFFER, vbo_positions_offset);
	glBufferData(GL_ARRAY_BUFFER, positions_offset_size * sizeof(GLfloat), positions_offset, GL_STATIC_DRAW);
	glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, NULL);
	glVertexAttribDivisor(1, 1);
	glEnableVertexAttribArray(1);

	glGenBuffers(1, &vbo_colors);
	glBindBuffer(GL_ARRAY_BUFFER, vbo_colors);
	glBufferData(GL_ARRAY_BUFFER, colors_size * sizeof(GLfloat), colors, GL_STATIC_DRAW);
	glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 0, NULL);
	glVertexAttribDivisor(2, 1);
	glEnableVertexAttribArray(2);


	gl_context->mvp = glGetUniformLocation(gl_context->program, "mvp");

	for(u32 h = 0, i = 0; h < 4; h++) {
		for(u32 w = 0; w < 4; w++) {
			if(w == h && w != 2) {
				gl_context->mvp_matrix[i++] = 1.0f;
			} else {
				gl_context->mvp_matrix[i++] = 0.0f;
			}
		}
	}


	free(positions_offset);
	free(colors);
}


/******************************************************************************
 * GUI rendering
 ******************************************************************************/

void gui_gl_area_render(GtkGLArea* area, GdkGLContext* context, gpointer data) {
	gui*       gui        = data;
	gl_context gl_context = gui->gl_context;

	glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT);

	glUseProgram(gl_context.program);
	glUniformMatrix4fv(gl_context.mvp, 1, GL_FALSE, gl_context.mvp_matrix);
	glBindVertexArray(gl_context.vao);
	glDrawArraysInstanced(GL_TRIANGLES, 0, 12, gui->hexarray.pixels);
}


/******************************************************************************
 * GUI resize
 ******************************************************************************/

void gui_gl_area_resize(GtkGLArea* area, gint width, gint height, gpointer data) {
	gui*        gui        = data;
	gl_context* gl_context = &gui->gl_context;

	gui->gl_area.width  = width;
	gui->gl_area.height = height;

	if(width > height) {
		gl_context->mvp_matrix[0] = (GLfloat)height / width;
		gl_context->mvp_matrix[5] = 1.0f;
	} else {
		gl_context->mvp_matrix[0] = 1.0f;
		gl_context->mvp_matrix[5] = (GLfloat)width  / height;
	}
}


/******************************************************************************
 * GUI interaction
 ******************************************************************************/

void gui_gl_area_interact(GtkWidget* widget, GdkEvent* event, gpointer data) {
	#define SCALING_FACTOR      1.0f / 9
	#define SCALING_FACTOR_UP   SCALING_FACTOR
	#define SCALING_FACTOR_DOWN ( 1 / (1 - SCALING_FACTOR_UP) - 1 )

	GdkEventType event_type = gdk_event_get_event_type(event);
	gui*         gui        = data;
	gl_area*     gl_area    = &gui->gl_area;
	gl_context*  gl_context = &gui->gl_context;

	if(event_type == GDK_ENTER_NOTIFY) {
		if(!gl_area->button_is_pressed) {
			gl_area->mouse_x = event->crossing.x;
			gl_area->mouse_y = event->crossing.y;

			gdk_window_set_cursor(
				gtk_widget_get_window(widget),
				gdk_cursor_new_from_name(gtk_widget_get_display(widget), "grab"));
		} else {
			gdk_window_set_cursor(
				gtk_widget_get_window(widget),
				gdk_cursor_new_from_name(gtk_widget_get_display(widget), "grabbing"));
		}
	} else if(event_type == GDK_LEAVE_NOTIFY) {
		gdk_window_set_cursor(
			gtk_widget_get_window(widget),
			gdk_cursor_new_from_name(gtk_widget_get_display(widget), "default"));
	} else if(event_type == GDK_MOTION_NOTIFY) {
		gl_area->mouse_x_old   = gl_area->mouse_x;
		gl_area->mouse_y_old   = gl_area->mouse_y;
		gl_area->mouse_x       = event->motion.x;
		gl_area->mouse_y       = event->motion.y;
		gl_area->mouse_delta_x = gl_area->mouse_x - gl_area->mouse_x_old;
		gl_area->mouse_delta_y = gl_area->mouse_y - gl_area->mouse_y_old;

		if(gl_area->button_is_pressed) {
			GLfloat translation_speed = 2 * (1 + gl_context->mvp_matrix[11]);

			gl_context->mvp_matrix[12] += translation_speed * gl_area->mouse_delta_x / gl_area->width;
			gl_context->mvp_matrix[13] -= translation_speed * gl_area->mouse_delta_y / gl_area->height;

			gtk_gl_area_queue_render(GTK_GL_AREA(widget));
		}
	} else if(event_type == GDK_BUTTON_PRESS) {
		gl_area->button_is_pressed = true;

		gdk_window_set_cursor(
			gtk_widget_get_window(widget),
			gdk_cursor_new_from_name(gtk_widget_get_display(widget), "grabbing"));
	} else if(event_type == GDK_BUTTON_RELEASE) {
		gl_area->button_is_pressed = false;

		gdk_window_set_cursor(
			gtk_widget_get_window(widget),
			gdk_cursor_new_from_name(gtk_widget_get_display(widget), "grab"));
	} else if(event_type == GDK_SCROLL) {
		if(event->scroll.direction == GDK_SCROLL_UP || event->scroll.direction == GDK_SCROLL_DOWN) {
			if(event->scroll.direction == GDK_SCROLL_UP) {
				gl_context->mvp_matrix[11] -= SCALING_FACTOR_UP * (1 + gl_context->mvp_matrix[11]);
				gl_context->zoom            = 100 / (1 + gl_context->mvp_matrix[11]);

				gdk_window_set_cursor(
					gtk_widget_get_window(widget),
					gdk_cursor_new_from_name(gtk_widget_get_display(widget), "zoom-in"));
			} else {
				gl_context->mvp_matrix[11] += SCALING_FACTOR_DOWN * (1 + gl_context->mvp_matrix[11]);
				gl_context->zoom            = 100 / (1 + gl_context->mvp_matrix[11]);

				gdk_window_set_cursor(
					gtk_widget_get_window(widget),
					gdk_cursor_new_from_name(gtk_widget_get_display(widget), "zoom-out"));
			}

			gtk_gl_area_queue_render(GTK_GL_AREA(widget));
			gui_update_title(gui);
		}
	}
}


/******************************************************************************
 * Update the GUI title
 ******************************************************************************/

void gui_update_title(gui* gui) {
	char title[256];
	sprintf(title, "%s [%.2f%%]", gui->title, gui->gl_context.zoom);
	gtk_window_set_title(GTK_WINDOW(gui->window), title);
}


/******************************************************************************
 * Activate the GUI
 ******************************************************************************/

void gui_activate(GtkApplication* app, gpointer data) {
	GdkPixbuf* icon    = gdk_pixbuf_new_from_resource("/logo.png", NULL);
	GtkWidget* frame   = gtk_frame_new(NULL);
	GtkWidget* gl_area = gtk_gl_area_new();
	gui*       gui     = data;

	gui->window                    = gtk_application_window_new(app);
	gui->gl_area.button_is_pressed = false;
	gui->gl_context.zoom           = 100.0f;


	gtk_widget_add_events(gui->window, GDK_KEY_PRESS_MASK);
	gui_update_title(gui);
	gtk_window_set_default_size(GTK_WINDOW(gui->window), gui->width, gui->height);
	gtk_window_set_icon(GTK_WINDOW(gui->window), icon);
	gtk_container_set_border_width(GTK_CONTAINER(gui->window), 8);
	gtk_frame_set_shadow_type(GTK_FRAME(frame), GTK_SHADOW_IN);

	gtk_widget_add_events(
		gl_area,
		 GDK_ENTER_NOTIFY_MASK | GDK_LEAVE_NOTIFY_MASK | GDK_POINTER_MOTION_MASK |
		 GDK_BUTTON_PRESS_MASK | GDK_BUTTON_RELEASE_MASK |
		 GDK_SCROLL_MASK);

	gtk_container_add(GTK_CONTAINER(gui->window), frame);
	gtk_container_add(GTK_CONTAINER(frame), gl_area);


	g_signal_connect(gl_area, "realize", G_CALLBACK(gui_gl_area_realize), gui);
	g_signal_connect(gl_area, "render",  G_CALLBACK(gui_gl_area_render),  gui);
	g_signal_connect(gl_area, "resize",  G_CALLBACK(gui_gl_area_resize),  gui);

	g_signal_connect(gl_area, "enter-notify-event",   G_CALLBACK(gui_gl_area_interact), gui);
	g_signal_connect(gl_area, "leave-notify-event",   G_CALLBACK(gui_gl_area_interact), gui);
	g_signal_connect(gl_area, "motion-notify-event",  G_CALLBACK(gui_gl_area_interact), gui);
	g_signal_connect(gl_area, "button-press-event",   G_CALLBACK(gui_gl_area_interact), gui);
	g_signal_connect(gl_area, "button-release-event", G_CALLBACK(gui_gl_area_interact), gui);
	g_signal_connect(gl_area, "scroll-event",         G_CALLBACK(gui_gl_area_interact), gui);


	gtk_widget_show_all(gui->window);
}


/******************************************************************************
 * Start the GUI
 ******************************************************************************/

int gui_run(char* title, u32 width, u32 height, Hexarray hexarray) {
	GtkApplication* app;
	gui             gui;
	int             status;

	gui.title    = title;
	gui.width    = width;
	gui.height   = height;
	gui.hexarray = hexarray;

	gtk_disable_setlocale();

	app = gtk_application_new("hexnet.gui", G_APPLICATION_NON_UNIQUE);
	g_signal_connect(app, "activate", G_CALLBACK(gui_activate), &gui);
	status = g_application_run(G_APPLICATION(app), 0, NULL);
	g_object_unref(app);

	return status;
}

