#include "PyQlever.h"

#define PY_ARRAY_UNIQUE_SYMBOL PyQlever_ARRAY_API
#define NO_IMPORT_ARRAY
#include <numpy/arrayobject.h>

#include "CsvStringParser.h"

#include <regex>

/**
  auto result = qlever->query(heritage_query + " LIMIT 10", ad_utility::MediaType::csv);
p,x,y,z,r,g,b
http://www.semanticweb.org/matteocodiglione/ontologies/2024/9/Heritage_Ontology/Neptune_Temple_Paestum_predicted#1,31.8442,20.499,8.84419,26880.0,23040.0,21504.0
http://www.semanticweb.org/matteocodiglione/ontologies/2024/9/Heritage_Ontology/Neptune_Temple_Paestum_predicted#10,30.4088,19.883,8.83623,23808.0,17920.0,15872.0
http://www.semanticweb.org/matteocodiglione/ontologies/2024/9/Heritage_Ontology/Neptune_Temple_Paestum_predicted#100,30.5682,20.095,8.76665,24064.0,15360.0,8704.0
http://www.semanticweb.org/matteocodiglione/ontologies/2024/9/Heritage_Ontology/Neptune_Temple_Paestum_predicted#1000,27.1402,20.81,8.71594,26112.0,17408.0,10240.0
http://www.semanticweb.org/matteocodiglione/ontologies/2024/9/Heritage_Ontology/Neptune_Temple_Paestum_predicted#10000,32.9632,7.06799,10.0808,25856.0,15360.0,7680.0
http://www.semanticweb.org/matteocodiglione/ontologies/2024/9/Heritage_Ontology/Neptune_Temple_Paestum_predicted#100000,26.31,20.104,3.82942,17152.0,11776.0,8704.0
http://www.semanticweb.org/matteocodiglione/ontologies/2024/9/Heritage_Ontology/Neptune_Temple_Paestum_predicted#1000000,50.5543,17.251,2.20139,30720.0,29440.0,30208.0
http://www.semanticweb.org/matteocodiglione/ontologies/2024/9/Heritage_Ontology/Neptune_Temple_Paestum_predicted#1000001,50.2631,17.187,2.43121,13824.0,13824.0,12544.0
http://www.semanticweb.org/matteocodiglione/ontologies/2024/9/Heritage_Ontology/Neptune_Temple_Paestum_predicted#1000002,50.7738,17.186,2.24453,24064.0,24064.0,20992.0
http://www.semanticweb.org/matteocodiglione/ontologies/2024/9/Heritage_Ontology/Neptune_Temple_Paestum_predicted#1000003,50.3173,17.453,2.2362,26624.0,24064.0,23808.0


 */
static void PyQleverQueryResult_dealloc(PyQleverQueryResultObject *self) {
  Py_DECREF(self->qleverObj);
  for (auto& array : self->result)
    Py_DECREF(array.second);
  self->result.clear();

  Py_TYPE(self)->tp_free((PyObject *) self);
}

static PyObject *PyQleverQueryResult_new(PyTypeObject *type, PyObject *args, PyObject *kwds) {
  PyQleverQueryResultObject *self;
  self = (PyQleverQueryResultObject *) type->tp_alloc(type, 0);
  if (self != nullptr) {
    self->qleverObj = nullptr;
    self->qlever = nullptr;
  }
  return (PyObject *) self;
}

std::pair<int, int> getResultShape(std::string log) {
  std::cout << log << std::endl;
  std::regex pattern(R"(Result has size (\d+) x (\d+))");
  std::smatch matches;
  if (std::regex_search(log, matches, pattern)) {
    int rows = std::stoi(matches[1]);;
    int cols = std::stoi(matches[2]);;
    return {rows, cols};
  }

  return {0, 0}; // no match found
}

static int PyQleverQueryResult_init(PyQleverQueryResultObject *self, PyObject *args, PyObject *kwds) {
  PyObject *qleverObj;
  if (!PyArg_ParseTuple(args, "O", &qleverObj))
    return -1;

  if (!PyObject_TypeCheck(qleverObj, &PyQleverType)) {
    PyErr_SetString(PyExc_TypeError, "Expected a Qlever object");
    return -1;
  }

  Py_INCREF(qleverObj);
  self->qleverObj = qleverObj;
  // call get_ref_
  PyObject* refObj = PyObject_CallMethod(qleverObj, "get_ref_", nullptr);
  if (refObj == nullptr) {
    PyErr_SetString(PyExc_RuntimeError, "Failed to get Qlever reference");
    return -1;
  }
  void* ptr = PyCapsule_GetPointer(refObj, "qlever.Qlever");
  self->qlever = static_cast<qlever::Qlever*>(ptr);
  Py_DECREF(refObj);

  return 0;
}

static PyObject *PyQleverQueryResult_append_chunk(PyQleverQueryResultObject *self, PyObject *args) {
  // TODO
  Py_RETURN_NONE;
}

static PyObject *PyQleverQueryResult_perform_query(PyQleverQueryResultObject *self, PyObject *args) {
  char* queryStr;
  if (!PyArg_ParseTuple(args, "s", &queryStr))
    return nullptr;

  std::ostringstream logStream;
  ad_utility::LogstreamChoice::get().setStream(&logStream);
  std::cout << "Performing query: " << queryStr << std::endl;

  std::string result = self->qlever->query(queryStr, ad_utility::MediaType::csv);

  ad_utility::LogstreamChoice::get().setStream(&std::cout); // reset log stream
  auto [rows, cols] = getResultShape(logStream.str());
  CsvStringParser parser(result, rows, cols);
  parser.parse();

  self->isStringColumn = parser.getIsStringColumn(); // append the results to the object
  auto varNames = parser.getColNames();
  for (auto& name : varNames)
    self->result[name] = nullptr; // initialize with null pointers

  auto parsed = parser.getResult();

  // create dictionary
  PyObject *resultDict = PyDict_New();
  for (size_t i = 0; i < self->result.size(); i++)
    PyDict_SetItemString(resultDict, varNames[i].c_str(), parsed[i]);

  PyObject *returnTuple = PyTuple_New(2);
  PyTuple_SetItem(returnTuple, 0, resultDict); // steals reference
  PyTuple_SetItem(returnTuple, 1, PyLong_FromSize_t(rows)); // steals reference
  return returnTuple;
}

static PyObject *PyQleverQueryResult_len(PyQleverQueryResultObject *self, PyObject *args) {
  return PyLong_FromSize_t(PyArray_DIM((PyArrayObject*)self->result[0], 0));
}

static PyObject *PyQleverQueryResult_iter(PyQleverQueryResultObject *self, PyObject *args) {
  // TODO
  Py_RETURN_NONE;
}

static PyObject *PyQleverQueryResult_getitem(PyQleverQueryResultObject *self, PyObject *key) {
  if (!PyUnicode_Check(key)) {
    PyErr_SetString(PyExc_TypeError, "Key must be a string");
    return nullptr;
  }
  const char* varName = PyUnicode_AsUTF8(key);

  if (!self->result.contains(varName)) {
    PyErr_Format(PyExc_KeyError, "Variable '%s' not found", varName);
    return nullptr; // variable not found
  }

  PyObject* array = self->result[varName];
  if (array == nullptr) {
    PyErr_Format(PyExc_RuntimeError, "Variable '%s' has no data", varName);
    return nullptr; // variable has no data
  }

  Py_INCREF(array);
  return array;
}

static PyObject *PyQleverQueryResult_tuple_iterator(PyQleverQueryResultObject *self, PyObject *args) {
  // get a list of strings representing the variable names
  std::vector<std::string> varNames;
  PyObject* varList;
  if (!PyArg_ParseTuple(args, "O", &varList))
    return nullptr;

  if (!PyList_Check(varList)) {
    PyErr_SetString(PyExc_TypeError, "Expected a list of variable names");
    return nullptr;
  }

  Py_ssize_t varCount = PyList_Size(varList);
  for (Py_ssize_t i = 0; i < varCount; i++) {
    PyObject* item = PyList_GetItem(varList, i);
    if (!PyUnicode_Check(item)) {
      PyErr_SetString(PyExc_TypeError, "Variable names must be strings");
      return nullptr;
    }
    varNames.emplace_back(PyUnicode_AsUTF8(item));
  }

  // TODO use custom iterator that yields MemoryMaps

  Py_RETURN_NONE;
}

static PyObject *PyQleverQueryResult_vars(PyQleverQueryResultObject *self, PyObject *args) {
  auto pyList = PyList_New(self->result.size());
  auto it = self->result.begin();
  for (int i = 0; i < self->result.size(); i++) {
    PyObject* pyStr = PyUnicode_FromString(it->first.c_str());
    PyList_SetItem(pyList, i, pyStr); // steals reference
    ++it;
  }
  return pyList;
}

static PyObject *PyQleverQueryResult_has_var(PyQleverQueryResultObject *self, PyObject *args) {
  char* varName;
  if (!PyArg_ParseTuple(args, "s", &varName))
    return nullptr;

  if (self->result.contains(varName))
    Py_RETURN_TRUE;
  else
    Py_RETURN_FALSE;
}

static PyMethodDef PyQleverQueryResult_methods[] = {
        {"_perform_query", (PyCFunction) PyQleverQueryResult_perform_query, METH_VARARGS,"Perform a SPARQL query and store the result."},
        {"tuple_iterator", (PyCFunction) PyQleverQueryResult_tuple_iterator, METH_VARARGS, "Return an iterator that yields rows as tuples."},
        {"vars", (PyCFunction) PyQleverQueryResult_vars, METH_NOARGS, "Return the list of variable names in the result."},
        {"has_var", (PyCFunction) PyQleverQueryResult_has_var, METH_O, "Check if a variable is in the result."},
        {nullptr} // Sentinel
};

static PySequenceMethods PyQleverQueryResult_as_sequence = {
  .sq_length = (lenfunc)PyQleverQueryResult_len,
  .sq_item = (ssizeargfunc)PyQleverQueryResult_getitem,
};

static PyMappingMethods PyQleverQueryResult_as_mapping = {
  .mp_length = (lenfunc)PyQleverQueryResult_len,
  .mp_subscript = (binaryfunc)PyQleverQueryResult_getitem,
};

PyTypeObject PyQleverQueryResultType = {
        .ob_base = PyVarObject_HEAD_INIT(nullptr, 0)
                           .tp_name = "pyqlever.QleverQueryResult",
        .tp_basicsize = sizeof(PyQleverQueryResultObject),
        .tp_itemsize = 0,
        .tp_dealloc = (destructor) PyQleverQueryResult_dealloc,
        .tp_as_sequence = &PyQleverQueryResult_as_sequence,
        .tp_as_mapping = &PyQleverQueryResult_as_mapping,
        .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
        .tp_doc = "Query result wrapper and iterator",
        .tp_iter = (getiterfunc)PyQleverQueryResult_iter,
        .tp_methods = PyQleverQueryResult_methods,
        .tp_init = (initproc) PyQleverQueryResult_init,
        .tp_new = PyQleverQueryResult_new};