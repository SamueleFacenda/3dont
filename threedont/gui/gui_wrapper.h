#ifndef GUI_WRAPPER_H
#define GUI_WRAPPER_H

#include <Python.h>
#include "controller_wrapper.h"
#include "main_layout.h"
#include <QApplication>
#include <thread>

typedef struct {
  PyObject_HEAD ControllerWrapper *controllerWrapper;
  MainLayout *mainLayout;
  QApplication *app;
  std::thread guiThread;

} GuiWrapperObject;

static bool pyListToQStringList(PyObject *pyList, QStringList &qStringList, const std::string &name = "List");

static void GuiWrapper_dealloc(GuiWrapperObject *self);

static PyObject *GuiWrapper_new(PyTypeObject *type, PyObject *args, PyObject *kwds);

static int GuiWrapper_init(GuiWrapperObject *self, PyObject *args, PyObject *kwds);

static PyObject *GuiWrapper_run(GuiWrapperObject *self, PyObject *args);

static PyObject *GuiWrapper_get_viewer_server_port(GuiWrapperObject *self, PyObject *args);

static PyObject *GuiWrapper_view_node_details(GuiWrapperObject *self, PyObject *args);

static PyObject *GuiWrapper_plot_tabular(GuiWrapperObject *self, PyObject *args);

static PyObject *GuiWrapper_set_statusbar_content(GuiWrapperObject *self, PyObject *args);

static PyObject *GuiWrapper_set_query_error(GuiWrapperObject *self, PyObject *args);

static PyObject *GuiWrapper_set_legend(GuiWrapperObject *self, PyObject *args);

static PyObject *GuiWrapper_set_project_list(GuiWrapperObject *self, PyObject *args);

static PyObject *GuiWrapper_get_properties_mapping(GuiWrapperObject *self, PyObject *args);

extern PyTypeObject GuiWrapperType;

PyMODINIT_FUNC PyInit_gui(void);

#endif // GUI_WRAPPER_H
