#include "PyQlever.h"

#include <chrono>

static void PyQlever_dealloc(PyQleverObject *self) {
  delete self->qlever;
  delete self->config;
  Py_TYPE(self)->tp_free((PyObject *) self);
}

static PyObject *PyQlever_new(PyTypeObject *type, PyObject *args, PyObject *kwds) {
  PyQleverObject *self;
  self = (PyQleverObject *) type->tp_alloc(type, 0);
  if (self != nullptr) {
    self->qlever = nullptr;
    self->config = nullptr;
  }
  return (PyObject *) self;
}

static int PyQlever_init(PyQleverObject *self, PyObject *args, PyObject *kwds) {
  int maxMemoryGb = 4; // default
  char* prefix = nullptr;
  static char *kwlist[] = {"prefix", "max_memory_gb", nullptr};
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "s|i" , kwlist, &prefix, &maxMemoryGb)) {
    return -1;
  }

  self->config = new qlever::IndexBuilderConfig();

  self->config->kbIndexName_ = "dev";
  self->config->memoryLimit_ = ad_utility::MemorySize::gigabytes(maxMemoryGb);
  self->config->vocabType_ = ad_utility::VocabularyType(ad_utility::VocabularyType::Enum::OnDiskCompressed);
  self->config->prefixesForIdEncodedIris_ = {std::string(prefix)};
  self->config->validate();

  return 0;
}

static PyObject *PyQlever_setup_storage(PyQleverObject *self, PyObject *args, PyObject *kwds) {
  const char *identifier;
  static char *kwlist[] = {"identifier", nullptr};
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "s", kwlist, &identifier))
    return nullptr;

  self->config->inputFiles_.emplace_back(std::string(""), qlever::Filetype::NQuad, identifier, true);

  Py_RETURN_NONE;
}

static PyObject *PyQlever_is_empty(PyQleverObject *self, PyObject *args) {
  if (self->qlever == nullptr)
    Py_RETURN_TRUE;

  Py_RETURN_FALSE;
}

static PyObject *PyQlever_bind_to_path(PyQleverObject *self, PyObject *args) {
  const char *path;
  if (!PyArg_ParseTuple(args, "s", &path))
    return nullptr;

  using namespace std::filesystem;

  self->config->baseName_ = std::string(path);
  if (!self->config->baseName_.ends_with("/"))
    self->config->baseName_ += "/";
  self->config->baseName_ += "dev";
  std::cout << "Binding QLever to path: " << self->config->baseName_ << std::endl;

  if (exists(path) && is_directory(path) && !is_empty(path)) {
    // load existing index
    try {
      self->qlever = new qlever::Qlever(qlever::EngineConfig(*self->config));
    } catch (const std::exception& e) {
      std::cerr << "Loading the index failed: " << e.what() << std::endl;
      PyErr_SetString(PyExc_RuntimeError, e.what());
      return nullptr;
    }
  }

  Py_RETURN_NONE;
}

static PyObject *PyQlever_load_file(PyQleverObject *self, PyObject *args) {
  const char *filePath;
  if (!PyArg_ParseTuple(args, "s", &filePath))
    return nullptr;

  // Requires the input in N-Quad format for now
  self->config->inputFiles_[0].filename_ = std::string(filePath);

  try {
    qlever::Qlever::buildIndex(*self->config);
    self->qlever = new qlever::Qlever(qlever::EngineConfig(*self->config));
  } catch (const std::exception& e) {
    std::cerr << "Building the index failed: " << e.what() << std::endl;
    PyErr_SetString(PyExc_RuntimeError, e.what());
    return nullptr;
  }

  Py_RETURN_NONE;
}

static PyObject *PyQlever_update(PyQleverObject *self, PyObject *args) {
  char* updateQuery;
  if (!PyArg_ParseTuple(args, "s", &updateQuery))
    return nullptr;

  self->qlever->query(std::string(updateQuery));

  Py_RETURN_NONE;
}

static PyObject *PyQlever_get_ref_(PyQleverObject *self, PyObject *args) {
  QleverRefObject* refObj = PyObject_New(QleverRefObject, &QleverRefType);
  if (refObj == nullptr)
    return nullptr;
  refObj->qlever = self->qlever;
  return reinterpret_cast<PyObject *>(refObj);
}

static PyMethodDef PyQlever_methods[] = {
        {"setup_storage", (PyCFunction) PyQlever_setup_storage, METH_VARARGS | METH_KEYWORDS, "Sets up the storage"},
        {"is_empty", (PyCFunction) PyQlever_is_empty, METH_NOARGS, "Checks if the storage is empty"},
        {"bind_to_path", (PyCFunction) PyQlever_bind_to_path, METH_VARARGS, "Binds the storage to a given path"},
        {"load_file", (PyCFunction) PyQlever_load_file, METH_VARARGS, "Loads data from a specified file"},
        {"update", (PyCFunction) PyQlever_update, METH_VARARGS, "Update query on the storage"},
        {"get_ref_", (PyCFunction) PyQlever_get_ref_, METH_NOARGS, "Get a reference to the underlying Qlever instance"},
        {nullptr}};

PyTypeObject PyQleverType = {
        .ob_base = PyVarObject_HEAD_INIT(nullptr, 0)
                           .tp_name = "pyqlever.Qlever",
        .tp_basicsize = sizeof(PyQleverObject),
        .tp_itemsize = 0,
        .tp_dealloc = (destructor) PyQlever_dealloc,
        .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
        .tp_doc = "Wrapper for the GUI",
        .tp_methods = PyQlever_methods,
        .tp_init = (initproc) PyQlever_init,
        .tp_new = PyQlever_new};