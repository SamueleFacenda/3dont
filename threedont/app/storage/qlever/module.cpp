#include "PyQlever.h"

#define PY_ARRAY_UNIQUE_SYMBOL PyQlever_ARRAY_API
#include <numpy/arrayobject.h>

static PyModuleDef pyqlevermodule = {
        PyModuleDef_HEAD_INIT,
        "pyqlever",
        "Python bindings for Qlever storage",
        -1,
};

PyMODINIT_FUNC PyInit_pyqlever(void) {
  import_array();

  PyObject *m;
  if (PyType_Ready(&PyQleverType) < 0)
    return nullptr;
  if (PyType_Ready(&PyQleverQueryResultType) < 0)
    return nullptr;
  if (PyType_Ready(&PyQleverQueryResultTupleIteratorType) < 0)
    return nullptr;

  m = PyModule_Create(&pyqlevermodule);
  if (m == nullptr)
    return nullptr;

  Py_INCREF(&PyQleverType);
  PyModule_AddObject(m, "Qlever", (PyObject *) &PyQleverType);

  Py_INCREF(&PyQleverQueryResultType);
  PyModule_AddObject(m, "QleverQueryResult", (PyObject *) &PyQleverQueryResultType);

  Py_INCREF(&PyQleverQueryResultTupleIteratorType);
  PyModule_AddObject(m, "QleverQueryResultTupleIterator", (PyObject *) &PyQleverQueryResultTupleIteratorType);

  return m;
}