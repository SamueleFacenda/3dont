#ifndef THREEDONT_PYQLEVER_H
#define THREEDONT_PYQLEVER_H

#include <Python.h>
#include <qlever/libqlever/Qlever.h>

typedef struct {
  PyObject_HEAD
  // Add members here as needed
} PyQleverObject;

static void PyQlever_dealloc(PyQleverObject *self);
static PyObject *PyQlever_new(PyTypeObject *type, PyObject *args, PyObject *kwds);
static int PyQlever_init(PyQleverObject *self, PyObject *args, PyObject *kwds);

extern PyTypeObject PyQleverType;
PyMODINIT_FUNC PyInit_pyqlever(void);

#endif // THREEDONT_PYQLEVER_H
