#ifndef THREEDONT_ALTERNATIVE_FRAME_BUFFER_H
#define THREEDONT_ALTERNATIVE_FRAME_BUFFER_H

#include "opengl_funcs.h"
#include <QOpenGLBuffer>
#include <QOpenGLShaderProgram>
#include <QOpenGLVertexArrayObject>

class AlternativeFrameBuffer : public OpenGLFuncs {
public:
  AlternativeFrameBuffer();
  void bind();
  void unbind();
  void setupBuffers(int w, int h);
  void displayTexture();

private:
  GLuint _fbo;
  GLuint _texture;
  GLuint _depth_buffer;
  QOpenGLShaderProgram *_display_texture_program;
  QOpenGLVertexArrayObject *_display_quad_vao;
  QOpenGLBuffer *_display_quad_vbo;
  QOpenGLBuffer *_display_quad_ebo;
};

#endif // THREEDONT_ALTERNATIVE_FRAME_BUFFER_H
