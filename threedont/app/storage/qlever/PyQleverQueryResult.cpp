#include "PyQlever.h"

#include "CsvStringParser.h"

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
  Py_TYPE(self)->tp_free((PyObject *) self);
}

static PyObject *PyQleverQueryResult_new(PyTypeObject *type, PyObject *args, PyObject *kwds) {
  PyQleverQueryResultObject *self;
  self = (PyQleverQueryResultObject *) type->tp_alloc(type, 0);
  return (PyObject *) self;
}

std::pair<int, int> getResultShape(std::string log) {
  std::cout << "Log stream:\n" << log << std::endl;
}

static int PyQleverQueryResult_init(PyQleverQueryResultObject *self, PyObject *args, PyObject *kwds) {
  PyObject *qleverObj;
  char* queryStr;
  if (!PyArg_ParseTuple(args, "Os", &qleverObj, &queryStr))
    return -1;

  if (!PyObject_TypeCheck(qleverObj, &PyQleverType)) {
    PyErr_SetString(PyExc_TypeError, "Expected a Qlever object");
    return -1;
  }


  auto qlever = reinterpret_cast<PyQleverObject *>(qleverObj); // TODO this might not work, it's a python object

  std::stringstream logStream;
  ad_utility::LogstreamChoice::get().setStream(&logStream);

  std::string result = qlever->qlever->query(queryStr, ad_utility::MediaType::csv);

  ad_utility::LogstreamChoice::get().setStream(&std::cout); // reset log stream
  auto [rows, cols] = getResultShape(logStream.str());
  CsvStringParser parser(result, rows, cols);
  parser.parse();

  self->isStringColumn = parser.getIsStringColumn();
  self->colNames = parser.getColNames();
  self->result = parser.getResult();

  return 0;
}

static PyMethodDef PyQleverQueryResult_methods[] = {
        // Define methods for PyQleverQueryResult here
        {nullptr} // Sentinel
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