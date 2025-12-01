#include "PyQlever.h"

static void PyQleverQueryResult_dealloc(PyQleverQueryResultObject *self) {
    Py_XDECREF(self->qlever);
    Py_TYPE(self)->tp_free((PyObject *) self);
}

static PyObject *PyQleverQueryResult_new(PyTypeObject *type, PyObject *args, PyObject *kwds) {
    PyQleverQueryResultObject *self;
    self = (PyQleverQueryResultObject *) type->tp_alloc(type, 0);
    if (self != nullptr) {
        self->qlever = nullptr;
    }
    return (PyObject *) self;
}

static int PyQleverQueryResult_init(PyQleverQueryResultObject *self, PyObject *args, PyObject *kwds) {
    PyObject *qleverObj;
    if (!PyArg_ParseTuple(args, "O", &qleverObj)) {
        return -1;
    }

    if (!PyObject_TypeCheck(qleverObj, &PyQleverType)) {
        PyErr_SetString(PyExc_TypeError, "Expected a Qlever object");
        return -1;
    }

    Py_INCREF(qleverObj);
    self->qlever = (PyQleverObject *) qleverObj;

    return 0;
}

static PyMethodDef PyQleverQueryResult_methods[] = {
    // Define methods for PyQleverQueryResult here
    {nullptr}  // Sentinel
};

PyTypeObject PyQleverQueryResultType = {
  .ob_base = PyVarObject_HEAD_INIT(nullptr, 0)
  .tp_name = "pyqlever.QleverQueryResult",
  .tp_basicsize = sizeof(PyQleverQueryResultObject),
  .tp_itemsize = 0,
  .tp_dealloc = (destructor) PyQleverQueryResult_dealloc,
  .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
  .tp_doc = "Query result wrapper and iterator",
  .tp_methods = PyQleverQueryResult_methods,
  .tp_init = (initproc) PyQleverQueryResult_init,
  .tp_new = PyQleverQueryResult_new};