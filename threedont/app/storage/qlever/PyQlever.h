#ifndef THREEDONT_PYQLEVER_H
#define THREEDONT_PYQLEVER_H

#include <Python.h>
#include <qlever/libqlever/Qlever.h>

// Qlever Wrapper Object (abstract_storage)
typedef struct {
  PyObject_HEAD
  // Add members here as needed
} PyQleverObject;

static void PyQlever_dealloc(PyQleverObject *self);
static PyObject *PyQlever_new(PyTypeObject *type, PyObject *args, PyObject *kwds);
static int PyQlever_init(PyQleverObject *self, PyObject *args, PyObject *kwds);

static PyObject *PyQlever_setup_storage(PyQleverObject *self, PyObject *args);
// static PyObject *PyQlever_query(PyQleverObject *self, PyObject *args); this is handled in python, creating a QueryResult object
static PyObject *PyQlever_is_empty(PyQleverObject *self, PyObject *args);
static PyObject *PyQlever_bind_to_path(PyQleverObject *self, PyObject *args);
static PyObject *PyQlever_load_file(PyQleverObject *self, PyObject *args);
static PyObject *PyQlever_update(PyQleverObject *self, PyObject *args);

extern PyTypeObject PyQleverType;
PyMODINIT_FUNC PyInit_pyqlever(void);


// QueryResult Object
typedef struct {
    PyObject_HEAD
    PyQleverObject* qlever;
} PyQleverQueryResultObject;

static void PyQleverQueryResult_dealloc(PyQleverQueryResultObject *self);
static PyObject *PyQleverQueryResult_new(PyTypeObject *type, PyObject *args, PyObject *kwds);
static int PyQleverQueryResult_init(PyQleverQueryResultObject *self, PyObject *args, PyObject *kwds);

static PyObject *PyQleverQueryResult_append_chunk(PyQleverQueryResultObject *self, PyObject *args);
static PyObject *PyQleverQueryResult_perform_query(PyQleverQueryResultObject *self, PyObject *args);
static PyObject *PyQleverQueryResult_len(PyQleverQueryResultObject *self, PyObject *args);
static PyObject *PyQleverQueryResult_iter(PyQleverQueryResultObject *self, PyObject *args);
static PyObject *PyQleverQueryResult_getitem(PyQleverQueryResultObject *self, PyObject *args);
static PyObject *PyQleverQueryResult_tuple_iterator(PyQleverQueryResultObject *self, PyObject *args);
static PyObject *PyQleverQueryResult_vars(PyQleverQueryResultObject *self, PyObject *args);
static PyObject *PyQleverQueryResult_has_var(PyQleverQueryResultObject *self, PyObject *args);

extern PyTypeObject PyQleverQueryResultType;
PyMODINIT_FUNC PyInit_pyqlever_query_result(void);

#endif // THREEDONT_PYQLEVER_H
