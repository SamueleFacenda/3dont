#ifndef __BACKGROUND_H__
#define __BACKGROUND_H__
#include "../utils/opengl_funcs.h"
#include <QOpenGLContext>
#include <QOpenGLShaderProgram>
#include <QWindow>

class Background : protected OpenGLFuncs {
public:
  Background();
  void draw();
  void setColorTop(QVector4D c);
  void setColorBottom(QVector4D c);
  QVector4D getColorTop() const;
  QVector4D getColorBottom() const;

private:
  void compileProgram();

  QOpenGLShaderProgram _program;
  QVector4D _bg_color_top;
  QVector4D _bg_color_bottom;
};

#endif // __BACKGROUND_H__
