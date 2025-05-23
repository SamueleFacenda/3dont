#ifndef THREEDONT_CONTROLLER_WRAPPER_H
#define THREEDONT_CONTROLLER_WRAPPER_H

#include <Python.h>
#include <cstdarg>
#include <stdexcept>
#include <string>

class ControllerWrapper {
private:
  PyObject *controller;
  inline static std::string neededMethods[] = {
          "select_query",
          "scalar_query",
          "connect_to_server",
          "stop",
          "view_point_details",
          "view_node_details",
          "start",
          "annotate_node",
          "select_all_subjects",
          "natural_language_query",
          "tabular_query"};

  static void callPythonMethod(PyObject *object, const char *methodName, const char *format, ...);

public:
  explicit ControllerWrapper(PyObject *controller);
  ~ControllerWrapper();

  void selectQuery(const std::string &query);
  void scalarQuery(const std::string &query);
  void connectToServer(const std::string &url, const std::string &ontologyNamespace);
  void stop();
  void viewPointDetails(unsigned int index);
  void viewNodeDetails(const std::string &node_id);
  void scalarWithPredicate(const std::string &predicate);
  void start();
  void annotateNode(const std::string &subject, const std::string &predicate, const std::string &object);
  void selectAllSubjects(const std::string &predicate, const std::string &object);
  void tabularQuery(const std::string &query);
  void naturalLanguageQuery(const std::string &query);
};

#endif // THREEDONT_CONTROLLER_WRAPPER_H