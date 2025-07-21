#include "comm_funcs.h"
#include <QDebug>

namespace comm {

void receiveBytes(char *destination, const qint64 bytesExpected,
                  QTcpSocket *clientConnection) {
    // notes: read() can read just part of buffer
    // but waitForReadyRead() unblocks only on receiving *new* data
    // not-yet-read data in buffer is not considered new
    qint64 bytesReceived = 0;
    while (bytesReceived < bytesExpected) {
        qint64 received =
            clientConnection->read(destination, bytesExpected - bytesReceived);
        if (received == 0) clientConnection->waitForReadyRead(-1);
        if (received == -1) {
            qDebug() << "error during socket read()";
            exit(1);
        }
        bytesReceived += received;
        destination += received;
    }
}

void sendBytes(const char *source, const qint64 size,
               QTcpSocket *clientConnection) {
    qint64 bytesLeft = size;
    const char *buf = source;
    while (bytesLeft > 0) {
        qint64 bytesSent = clientConnection->write(buf, bytesLeft);
        if (bytesSent == -1) {
            qDebug() << "error during socket write()";
            exit(1);
        }
        buf += bytesSent;
        bytesLeft -= bytesSent;
        clientConnection->waitForBytesWritten();
    }
}

void sendError(const char *msg, const quint64 size,
               QTcpSocket *clientConnection) {
    // send data type
    unsigned char dataType = 0;
    sendBytes((char *) &dataType, 1, clientConnection);

    // send number of dimensions
    quint64 numDims = 1;
    sendBytes((char *) &numDims, sizeof(quint64), clientConnection);

    // send dimensions
    sendBytes((char *) &size, sizeof(quint64), clientConnection);

    // send array elements
    sendBytes((char *) msg, sizeof(char) * size, clientConnection);
}

} // namespace comm
