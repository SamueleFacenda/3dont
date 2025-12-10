#pragma once

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include <string>
#include <vector>
#include <array>

// TODO make this configurable
#define PARSER_THREADS 4

/**
 * parses a CSV string and provides access to its data.
 * Only supports strings and floats.
 */
class CsvStringParser {
public:
  CsvStringParser(const std::string& csvString, int resultLen, int numCols)
      : data(csvString.data()), length(static_cast<int>(csvString.size())),
        rows(resultLen), cols(numCols) {
    init();
  }
  void parse();
  std::vector<PyObject*> getResult() const { return result; }
  std::vector<std::string> getColNames() const { return colNames; }
  std::vector<bool> getIsStringColumn() const { return isStringColumn; }
private:
  const char* data;
  int length, rows, cols;
  std::vector<std::array<std::vector<const char*>,PARSER_THREADS>> stringColumns; // [col][thread][rel row]
  std::vector<std::array<std::vector<size_t>,PARSER_THREADS>> stringLengths;
  std::vector<std::array<std::vector<double>,PARSER_THREADS>> floatColumns;
  std::vector<bool> isStringColumn;
  std::vector<std::string> colNames;
  std::vector<PyObject*> result;

  void worker(int threadId, int start, int end);
  void merge();
  PyObject* mergeStringColumn(int col);
  PyObject* mergeFloatColumn(int col);

  void init();
};