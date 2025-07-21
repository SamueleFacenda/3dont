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

  // Non-template functions moved to source file
  void receiveBytes(char *destination, const qint64 bytesExpected,
                    QTcpSocket *clientConnection);
  void sendBytes(const char *source, const qint64 size,
                 QTcpSocket *clientConnection);
  void sendError(const char *msg, const quint64 size,
                 QTcpSocket *clientConnection);

  // Template functions remain in header
  template<typename T>
  void sendScalar(const T value, QTcpSocket *clientConnection) {
    // send data type
    unsigned char dataType = TypeCode<T>::value;
    sendBytes((char *) &dataType, 1, clientConnection);

    // send number of dimensions
    quint64 numDims = 1;
    sendBytes((char *) &numDims, sizeof(quint64), clientConnection);

    // send dimensions
    quint64 numElts = 1;
    sendBytes((char *) &numElts, sizeof(quint64), clientConnection);

    // send data
    sendBytes((char *) &value, sizeof(T), clientConnection);
  }

  template<typename T>
  void sendArray(const T *source, const quint64 size,
                 QTcpSocket *clientConnection) {
    // send data type
    unsigned char dataType = TypeCode<T>::value;
    sendBytes((char *) &dataType, 1, clientConnection);

    // send number of dimensions
    quint64 numDims = 1;
    sendBytes((char *) &numDims, sizeof(quint64), clientConnection);

    // send dimensions
    sendBytes((char *) &size, sizeof(quint64), clientConnection);

    // send array elements
    sendBytes((char *) source, sizeof(T) * size, clientConnection);
  }

  template<typename T>
  void sendMatrix(const T *source, // in column major order
                  const quint64 numRows, const quint64 numCols,
                  QTcpSocket *clientConnection) {
    // send data type
    unsigned char dataType = TypeCode<T>::value;
    sendBytes((char *) &dataType, 1, clientConnection);

    // send number of dimensions
    quint64 numDims = 2;
    sendBytes((char *) &numDims, sizeof(quint64), clientConnection);

    // send dimensions
    quint64 dims[2] = {numRows, numCols};
    sendBytes((char *) &dims[0], 2 * sizeof(quint64), clientConnection);

    // send array elements
    sendBytes((char *) source, sizeof(T) * numRows * numCols, clientConnection);
  }

  template<typename T>
  void sendMultiDimArray(const T *source, const std::vector<quint64> &dims,
                         QTcpSocket *clientConnection) {
    // send data type
    unsigned char dataType = TypeCode<T>::value;
    sendBytes((char *) &dataType, 1, clientConnection);

    // send number of dimensions
    quint64 numDims = dims.size();
    sendBytes((char *) &numDims, sizeof(quint64), clientConnection);

    // send dimensions
    quint64 numElts = 1;
    for (std::size_t i = 0; i < dims.size(); i++) {
      sendBytes((char *) &dims[i], sizeof(quint64), clientConnection);
      numElts *= dims[i];
    }

    // send array elements
    sendBytes((char *) source, sizeof(T) * numElts, clientConnection);
  }
} // namespace comm
#endif // __COMMFUNCS_H__
