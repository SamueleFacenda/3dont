#ifndef __LOOKAT_H__
#define __LOOKAT_H__
#include "../camera/qt_camera.h"
#include "../utils/opengl_funcs.h"
#include <QOpenGLContext>
#include <QOpenGLShaderProgram>
#include <QWindow>

class LookAt : protected OpenGLFuncs {
public:
  LookAt();
  ~LookAt();
  void draw(const QtCamera &camera);
  void setVisible(bool visible);
  bool getVisible() const;

private:
  void compileProgram();
  void initializeBuffers();

  QOpenGLShaderProgram _program;
  bool _visible;
  GLuint _vao;
  GLuint _vbo_positions;
  GLuint _vbo_colors;
};

#endif // __LOOKAT_H__
