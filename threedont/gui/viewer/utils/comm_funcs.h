#ifndef __COMMFUNCS_H__
#define __COMMFUNCS_H__
#include <QTcpSocket>
#include <vector>

namespace comm {
  template<typename T>
  struct TypeCode {
    static const unsigned char value = 0;
  };
  template<>
  struct TypeCode<char> {
    static const unsigned char value = 1;
  };
  template<>
  struct TypeCode<float> {
    static const unsigned char value = 2;
  };
  template<>
  struct TypeCode<int> {
    static const unsigned char value = 3;
  };
  template<>
  struct TypeCode<unsigned int> {
    static const unsigned char value = 4;
  };

  template<typename T>
  QByteArray sendError(const char *msg, const quint64 size) {
    QByteArray out;
    // send data type
    unsigned char dataType = 0;
    out.append((char *) &dataType, 1);

    // send number of dimensions
    quint64 numDims = 1;
    out.append((char *) &numDims, sizeof(quint64));

    // send dimensions
    out.append((char *) &size, sizeof(quint64));

    // send array elements
    out.append((char *) msg, sizeof(char) * size);
    return out;
  }

  // Template functions remain in header
  template<typename T>
  QByteArray sendScalar(const T value) {
    // send data type
    unsigned char dataType = TypeCode<T>::value;
    QByteArray out;
    out.append((char *) &dataType, 1);

    // send number of dimensions
    quint64 numDims = 1;
    out.append((char *) &numDims, sizeof(quint64));

    // send dimensions
    quint64 numElts = 1;
    out.append((char *) &numElts, sizeof(quint64));

    // send data
    out.append((char *) &value, sizeof(T));
    return out;
  }

  template<typename T>
  QByteArray sendArray(const T *source, const quint64 size) {
    QByteArray out;
    // send data type
    unsigned char dataType = TypeCode<T>::value;
    out.append((char *) &dataType, 1);

    // send number of dimensions
    quint64 numDims = 1;
    out.append((char *) &numDims, sizeof(quint64));

    // send dimensions
    out.append((char *) &size, sizeof(quint64));

    // send array elements
    out.append((char *) source, sizeof(T) * size);
    return out;
  }

  template<typename T>
  QByteArray sendMatrix(const T *source, // in column major order
                  const quint64 numRows, const quint64 numCols) {
    QByteArray out;
    // send data type
    unsigned char dataType = TypeCode<T>::value;
    out.append((char *) &dataType, 1);

    // send number of dimensions
    quint64 numDims = 2;
    out.append((char *) &numDims, sizeof(quint64));

    // send dimensions
    quint64 dims[2] = {numRows, numCols};
    out.append((char *) &dims[0], 2 * sizeof(quint64));

    // send array elements
    out.append((char *) source, sizeof(T) * numRows * numCols);
    return out;
  }
} // namespace comm
#endif // __COMMFUNCS_H__
