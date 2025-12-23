#pragma once

#include <Python.h>

#include <string>
#include <vector>
#include <array>

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
  int length, rows, cols, threadsCount;
  std::vector<std::vector<std::vector<const char*>>> stringColumns; // [col][thread][rel row]
  std::vector<std::vector<std::vector<size_t>>> stringLengths;
  std::vector<std::vector<std::vector<float>>> floatColumns;
  std::vector<bool> isStringColumn;
  std::vector<std::string> colNames;
  std::vector<PyObject*> result;

  void worker(int threadId, int start, int end);
  void merge();
  void computeNumRows();
  PyObject* mergeStringColumn(int col);
  PyObject* mergeFloatColumn(int col);

  void processHeader();
  void allocateColumnsSpace();
  void detectColumnType();
  void init();
};