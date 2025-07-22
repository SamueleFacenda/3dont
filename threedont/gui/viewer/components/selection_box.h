#ifndef __SELECTIONBOX_H__
#define __SELECTIONBOX_H__
#include "../utils/opengl_funcs.h"
#include <QOpenGLContext>
#include <QOpenGLShaderProgram>
#include <QPointF>
#include <QRectF>
#include <QWindow>

class SelectionBox : protected OpenGLFuncs {
public:
  enum SelectMode { ADD = 0,
                    SUB = 1,
                    NONE = 2 };

  SelectionBox();
  ~SelectionBox();

  void draw();
  void click(QPointF p, SelectMode select_mode);
  void drag(QPointF p);
  void release();

  bool active() const;
  bool empty() const;
  const QRectF &getBox() const;
  SelectMode getType() const;

private:
  void compileProgram();
  void initializeBuffers();

  QOpenGLShaderProgram _program;
  SelectMode _select_mode;
  QPointF _anchor;
  QRectF _box;
  GLuint _vao;
  GLuint _vbo_vertices;
  GLuint _vbo_indices;
};

#endif // __SELECTIONBOX_H__
