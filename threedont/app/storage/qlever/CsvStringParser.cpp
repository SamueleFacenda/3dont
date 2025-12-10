#include "CsvStringParser.h"

#include "fast_float/fast_float.h"

#include <iostream>
#include <thread>

void CsvStringParser::init() {
  stringColumns.resize(cols);
  stringLengths.resize(cols);
  floatColumns.resize(cols);
  isStringColumn.resize(cols, false);
  result.resize(cols);

  char* current = const_cast<char*>(data);
  int colIt = 0;
  while (colIt < cols) {
    isStringColumn[colIt] = *current == 'h'; // starts with http

    colIt++;
    while (colIt < cols && *current++ != ','); // there are no spaces
  }
}

void CsvStringParser::parse() {
  std::vector<std::thread> workers;
  int chunkSize = length / PARSER_THREADS;
  for (int threadId = 0; threadId < PARSER_THREADS; threadId++) {
    int start = threadId * chunkSize;
    int end = (threadId == PARSER_THREADS - 1) ? length : (threadId + 1) * chunkSize;
    workers.emplace_back(&CsvStringParser::worker, this, threadId, start, end);
  }
  for (auto& worker : workers)
    worker.join();

  merge();
}

void CsvStringParser::worker(int threadId, int start, int end) {
  char const *current = const_cast<char *>(data) + start; // move to start row
  if (threadId != 0) {
    while (*current++ != '\n'); // move to next line
  }
  while (current < data + length && current < data + end) {
    for (int colIt = 0; colIt < cols; colIt++) {
      if (isStringColumn[colIt]) {
        const char* strIt = current;
        while (*current != ',' && *current != '\n') current++;
        size_t len = current - strIt;
        stringColumns[colIt][threadId].push_back(strIt);
        stringLengths[colIt][threadId].push_back(len);
      } else {
        auto [ptr, ec] = fast_float::from_chars(current, data + length, floatColumns[colIt][threadId].emplace_back());
        current = ptr;
        if (ec == std::errc()) {
          std::cerr << "Error parsing float in CSV at char " << current - data << ", col " << colIt << std::endl;
          exit(1);
        }
      }
      current++; // skip comma or newline
    }
  }
}

PyObject* CsvStringParser::mergeStringColumn(int col) {
  int maxSize = 0;
  for (int t = 0; t < PARSER_THREADS; t++)
    if (stringColumns[col][t].size() > maxSize)
      maxSize = stringColumns[col][t].size();

  npy_intp dims[1] = {static_cast<npy_intp>(rows)};

  PyArray_Descr *descr = PyArray_DescrNewFromType(NPY_STRING);
  PyDataType_SET_ELSIZE(descr, maxSize);

  PyObject* array = PyArray_SimpleNewFromDescr(1, dims, descr);
  char *data = (char *) PyArray_DATA((PyArrayObject *) array);

  size_t idx = 0;
  for (int t = 0; t < PARSER_THREADS; t++) {
    for (size_t i = 0; i < stringColumns[col][t].size(); i++) {
      const char *src = stringColumns[col][t][i];
      size_t len = stringLengths[col][t][i];
      char *dest = data + idx * maxSize;

      memcpy(dest, src, len);
      // Zero-pad remaining bytes
      if (len < maxSize)
        memset(dest + len, 0, maxSize - len);

      idx++;
    }
  }
  return array;
}

PyObject* CsvStringParser::mergeFloatColumn(int col) {
  std::vector<double> mergedFloats;
  mergedFloats.reserve(rows);
  for (int t = 0; t < PARSER_THREADS; t++)
    mergedFloats.insert(mergedFloats.end(),
                        floatColumns[col][t].begin(),
                        floatColumns[col][t].end());
  npy_intp dims[1] = {static_cast<npy_intp>(mergedFloats.size())};
  PyObject *array = PyArray_SimpleNewFromData(1, dims, NPY_DOUBLE, mergedFloats.data());
  return array;
}

void CsvStringParser::merge() {
  for (int col=0; col < cols; col++) {
    if (isStringColumn[col])
      result[col] = mergeStringColumn(col);
    else
      result[col] = mergeFloatColumn(col);
  }
}