#ifndef __OPENGLFUNCS_H__
#define __OPENGLFUNCS_H__
#include <QOpenGLFunctions>
#include <QOpenGLFunctions_3_3_Core>

class OpenGLFuncs : public QOpenGLFunctions_3_3_Core {
  // extends QOpenGLFunctions with some helper and error checking functions
public:
  OpenGLFuncs();
  GLint getBufferSize(GLuint bufferId);
  void checkError();
  void printFramebufferStatus();
};

#endif // __OPENGLFUNCS_H__
