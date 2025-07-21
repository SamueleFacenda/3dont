#include "alternative_frame_buffer.h"
#include <QDebug>

AlternativeFrameBuffer::AlternativeFrameBuffer() {
  initializeOpenGLFunctions();
  glGenFramebuffers(1, &_fbo);
  glGenTextures(1, &_texture);
  glGenRenderbuffers(1, &_depth_buffer);

  _display_texture_program = new QOpenGLShaderProgram();
  _display_texture_program->addShaderFromSourceCode(QOpenGLShader::Vertex,
                                                    "#version 330 core\n"
                                                    "layout (location = 0) in vec2 a_pos;\n"
                                                    "layout (location = 1) in vec2 a_texCoord;\n"
                                                    "out vec2 v_texCoord;\n"
                                                    "void main() {\n"
                                                    "    gl_Position = vec4(a_pos, 0.0, 1.0);\n"
                                                    "    v_texCoord = a_texCoord;\n"
                                                    "}\n");
  _display_texture_program->addShaderFromSourceCode(QOpenGLShader::Fragment,
                                                    "#version 330 core\n"
                                                    "in vec2 v_texCoord;\n"
                                                    "uniform sampler2D u_texture;\n"
                                                    "out vec4 fragColor;\n"
                                                    "void main() {\n"
                                                    "    fragColor = texture(u_texture, v_texCoord);\n"
                                                    "}\n");
  _display_texture_program->link();

  // Define constant quad data
  float vertices[] = {
          // positions     // texture coords
          -1.0f, -1.0f, 0.0f, 0.0f,
          1.0f, -1.0f, 1.0f, 0.0f,
          1.0f, 1.0f, 1.0f, 1.0f,
          -1.0f, 1.0f, 0.0f, 1.0f};
  unsigned int indices[] = {0, 1, 2, 2, 3, 0};

  _display_quad_vao = new QOpenGLVertexArrayObject();
  _display_quad_vao->create();
  _display_quad_vao->bind(); // Bind the VAO to store the following settings

  _display_quad_vbo = new QOpenGLBuffer(QOpenGLBuffer::VertexBuffer);
  _display_quad_vbo->create();
  _display_quad_vbo->bind();
  _display_quad_vbo->allocate(vertices, sizeof(vertices)); // Allocate and upload data now

  _display_quad_ebo = new QOpenGLBuffer(QOpenGLBuffer::IndexBuffer);
  _display_quad_ebo->create();
  _display_quad_ebo->bind();
  _display_quad_ebo->allocate(indices, sizeof(indices)); // Allocate and upload data now

  // Configure vertex attributes. This state is stored in the VAO.
  // Position attribute
  glEnableVertexAttribArray(0);
  glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void *) nullptr);
  // Texture coordinate attribute
  glEnableVertexAttribArray(1);
  glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void *) (2 * sizeof(float)));

  _display_quad_vao->release(); // Unbind the VAO
}

void AlternativeFrameBuffer::bind() {
  glBindFramebuffer(GL_FRAMEBUFFER, _fbo);
}

void AlternativeFrameBuffer::unbind() {
  glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void AlternativeFrameBuffer::setupBuffers(int w, int h) {
  // Set up color texture
  glBindTexture(GL_TEXTURE_2D, _texture);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

  // Set up depth buffer
  glBindRenderbuffer(GL_RENDERBUFFER, _depth_buffer);
  glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, w, h);

  // Set up framebuffer
  glBindFramebuffer(GL_FRAMEBUFFER, _fbo);
  glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, _texture, 0);
  glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, _depth_buffer);

  // Check framebuffer completeness
  if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
    qDebug() << "Fine render framebuffer not complete!";

  // Restore default framebuffer
  glBindFramebuffer(GL_FRAMEBUFFER, 0);
  glBindTexture(GL_TEXTURE_2D, 0);
  glBindRenderbuffer(GL_RENDERBUFFER, 0);
}

void AlternativeFrameBuffer::displayTexture() {
  // --- Save and Set OpenGL State ---
  GLboolean depth_test_enabled = glIsEnabled(GL_DEPTH_TEST);
  GLboolean blend_enabled = glIsEnabled(GL_BLEND);
  GLint current_program;
  glGetIntegerv(GL_CURRENT_PROGRAM, &current_program);

  glDisable(GL_DEPTH_TEST);
  glDisable(GL_BLEND);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  // --- Bind and Draw ---
  _display_texture_program->bind();
  _display_texture_program->setUniformValue("u_texture", 0);

  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_2D, _texture);

  // All we need to do is bind the VAO. All VBO/EBO/Attribute data is already set.
  _display_quad_vao->bind();
  glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, nullptr);
  _display_quad_vao->release();

  _display_texture_program->release();

  // --- Restore OpenGL State (no changes here) ---
  if (depth_test_enabled) glEnable(GL_DEPTH_TEST);
  if (blend_enabled) glEnable(GL_BLEND);
  glUseProgram(current_program);
}
