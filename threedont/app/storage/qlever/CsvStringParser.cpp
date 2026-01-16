#include "CsvStringParser.h"

#include "fast_float/fast_float.h"

#define PY_ARRAY_UNIQUE_SYMBOL PyQlever_ARRAY_API
#define NO_IMPORT_ARRAY
#include <numpy/arrayobject.h>

#include <iostream>
#include <thread>
#include <algorithm>
#include <climits>

#define FALLBACK_PARSER_THREADS 4

void CsvStringParser::processHeader() {
  const char* current = data;
  if (cols == -1) // unknown, parse until newline
    cols = INT_MAX;
  for (int colIt = 0; colIt < cols; colIt++) {
    const char *start = current;
    while (*current != ',' && *current != '\n')
      current++;

    std::string colName(start, current - start);
    colNames.push_back(colName);

    if (*current == '\n') {
      current++;        // skip newline
      cols = colIt + 1; // adjust cols
      break;
    }
    current++; // skip comma or newline
  }

  length -= (current - data); // adjust length to skip header
  data = current;
}

void CsvStringParser::allocateColumnsSpace() {
  stringColumns.resize(cols);
  stringLengths.resize(cols);
  floatColumns.resize(cols);
  isStringColumn.resize(cols, false);
  result.resize(cols);

  threadsCount = std::thread::hardware_concurrency();
  if (threadsCount == 0)
    threadsCount = FALLBACK_PARSER_THREADS;
  for (int colIt = 0; colIt < cols; colIt++) {
    stringColumns[colIt].resize(threadsCount);
    stringLengths[colIt].resize(threadsCount);
    floatColumns[colIt].resize(threadsCount);
  }
}

void CsvStringParser::detectColumnType() {
  const char* current = data;
  for (int colIt = 0; colIt < cols; colIt++) {
    isStringColumn[colIt] = *current == 'h';       // starts with http

    while (colIt < cols - 1 && *current++ != ','); // there are no spaces
  }
}

void CsvStringParser::init() {
  processHeader();

  allocateColumnsSpace();

  if (rows == 0)
    return; // no data

  detectColumnType();
}

void CsvStringParser::computeNumRows() {
  if (cols == 0)
    return; // no columns
  rows = 0;
  for (int t = 0; t < threadsCount; t++) {
    if (isStringColumn[0])
      rows += stringColumns[0][t].size();
    else
      rows += floatColumns[0][t].size();
  }
}


void CsvStringParser::parse() {
  Py_BEGIN_ALLOW_THREADS
  std::vector<std::thread> workers;
  size_t chunkSize = length / threadsCount;
  int chunkRows = rows / threadsCount; // only a rough estimate
  for (int threadId = 0; threadId < threadsCount; threadId++) {
    if (rows != -1) { // preallocate only if we know the number of rows
      for (int colIt = 0; colIt < cols; colIt++) {
        // preallocate guess size
        if (isStringColumn[colIt]) {
          stringColumns[colIt][threadId].reserve(chunkRows);
          stringLengths[colIt][threadId].reserve(chunkRows);
        } else {
          floatColumns[colIt][threadId].reserve(chunkRows);
        }
      }
    }

    size_t start = threadId * chunkSize;
    size_t end = (threadId == threadsCount - 1) ? length : (threadId + 1) * chunkSize;
    workers.emplace_back(&CsvStringParser::worker, this, threadId, start, end);
  }
  for (auto& worker : workers)
    worker.join();

  computeNumRows(); // input num rows is not reliable!!
  Py_END_ALLOW_THREADS
  merge();
}

void CsvStringParser::worker(int threadId, size_t start, size_t end) {
  char const *current = const_cast<char *>(data) + start; // move to start row
  if (threadId != 0) {
    while (*current++ != '\n'); // move to next line
  }
  while (current < data + length && current <= data + end) {
    for (int colIt = 0; colIt < cols; colIt++) {
      if (isStringColumn[colIt]) {
        const char* strIt = current;
        while (*current != ',' && *current != '\n') current++;
        size_t len = current - strIt;
        stringColumns[colIt][threadId].push_back(strIt);
        stringLengths[colIt][threadId].push_back(len);
      } else {
        auto [ptr, ec] = fast_float::from_chars(current, data + length, floatColumns[colIt][threadId].emplace_back());
        if (ec != std::errc()) {
          std::cerr << "Error parsing float in CSV at char " << current - data << ", col " << colIt << " with value: ";
          std::cerr << std::make_error_code(ec).message() << std::endl;
          std::cerr << "Data snippet [-10,|,10]: ";
          for (int i = -10; i < 10; i++) {
            if (i == 0) std::cerr << "|";
            std::cerr << *(current + i);
          }
          std::cerr << std::endl;
          exit(1);
        }
        current = ptr;
      }
      current++; // skip comma or newline
    }
  }
}

PyObject* CsvStringParser::mergeStringColumn(int col) {
  size_t maxSize = 0;
  for (int t = 0; t < threadsCount; t++) {
    if (!stringLengths[col][t].empty())
      maxSize = std::max(maxSize, *std::ranges::max_element(stringLengths[col][t]));
  }
  maxSize++; // for null terminator
  npy_intp dims[1] = {static_cast<npy_intp>(rows)};

  PyArray_Descr *descr = PyArray_DescrNewFromType(NPY_STRING);
  PyDataType_SET_ELSIZE(descr, maxSize);

  PyObject* array = PyArray_SimpleNewFromDescr(1, dims, descr);
  char *data = (char *) PyArray_DATA((PyArrayObject *) array);

  size_t idx = 0;
  for (int t = 0; t < threadsCount; t++) {
    for (size_t i = 0; i < stringColumns[col][t].size(); i++) {
      const char *src = stringColumns[col][t][i];
      size_t len = stringLengths[col][t][i];
      char *dest = data + idx * maxSize;

      memcpy(dest, src, len);
      // Zero-pad remaining bytes
      memset(dest + len, 0, maxSize - len);

      idx++;
    }
  }
  return array;
}

PyObject* CsvStringParser::mergeFloatColumn(int col) {
  npy_intp dims[1] = {static_cast<npy_intp>(rows)};
  PyObject *array = PyArray_SimpleNew(1, dims, NPY_FLOAT);
  if (rows == 0)
    return array;

  float *data = (float*)PyArray_DATA((PyArrayObject*)array);

  size_t idx = 0;
  for (int t = 0; t < threadsCount; t++) {
    size_t chunkSize = floatColumns[col][t].size();
    memcpy(data + idx, floatColumns[col][t].data(), chunkSize * sizeof(float));
    idx += chunkSize;
  }

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