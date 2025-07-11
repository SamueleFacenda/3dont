#ifndef THREEDONT_ALTERNATIVE_FRAME_BUFFER_H
#define THREEDONT_ALTERNATIVE_FRAME_BUFFER_H

#include "opengl_funcs.h"

class AlternativeFrameBuffer: public OpenGLFuncs {
public:
  AlternativeFrameBuffer() {
    initializeOpenGLFunctions();
    glGenFramebuffers(1, &_fine_render_fbo);
    glGenTextures(1, &_fine_render_texture);
    glGenRenderbuffers(1, &_fine_render_depth_buffer);

    _display_texture_program = new QOpenGLShaderProgram();
    _display_texture_program->addShaderFromSourceCode(QOpenGLShader::Vertex,
                                                      "#version 330 core\n"
                                                      "layout (location = 0) in vec2 a_pos;\n"
                                                      "layout (location = 1) in vec2 a_texCoord;\n"
                                                      "out vec2 v_texCoord;\n"
                                                      "void main() {\n"
                                                      "    gl_Position = vec4(a_pos, 0.0, 1.0);\n"
                                                      "    v_texCoord = a_texCoord;\n"
                                                      "}\n"
    );
    _display_texture_program->addShaderFromSourceCode(QOpenGLShader::Fragment,
                                                      "#version 330 core\n"
                                                      "in vec2 v_texCoord;\n"
                                                      "uniform sampler2D u_texture;\n"
                                                      "out vec4 fragColor;\n"
                                                      "void main() {\n"
                                                      "    fragColor = texture(u_texture, v_texCoord);\n"
                                                      "}\n"
    );
    _display_texture_program->link();

    // Create VAO/VBO for full-screen quad
    _display_quad_vao = new QOpenGLVertexArrayObject();
    _display_quad_vao->create();
    _display_quad_vbo = new QOpenGLBuffer(QOpenGLBuffer::VertexBuffer);
    _display_quad_vbo->create();
    _display_quad_ebo = new QOpenGLBuffer(QOpenGLBuffer::IndexBuffer);
    _display_quad_ebo->create();

    _display_quad_vao->bind();
    _display_quad_vbo->bind();
    _display_quad_vbo->allocate(4 * 4 * sizeof(float));

    _display_quad_ebo->bind();
    unsigned int indices[] = {0, 1, 2, 2, 3, 0};
    _display_quad_ebo->allocate(indices, sizeof(indices));
    _display_quad_vao->release();
  }

  void bind() {
    glBindFramebuffer(GL_FRAMEBUFFER, _fine_render_fbo);
  }

  void unbind() {
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
  }

  void setupFineRenderBuffers(int w, int h) {
    // Set up color texture
    glBindTexture(GL_TEXTURE_2D, _fine_render_texture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    // Set up depth buffer
    glBindRenderbuffer(GL_RENDERBUFFER, _fine_render_depth_buffer);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, w, h);

    // Set up framebuffer
    glBindFramebuffer(GL_FRAMEBUFFER, _fine_render_fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, _fine_render_texture, 0);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, _fine_render_depth_buffer);

    // Check framebuffer completeness
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
      qDebug() << "Fine render framebuffer not complete!";
    }

    // Restore default framebuffer
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glBindTexture(GL_TEXTURE_2D, 0);
    glBindRenderbuffer(GL_RENDERBUFFER, 0);
  }

  void displayFineRenderTexture() {
    // Save current OpenGL state
    GLboolean depth_test_enabled = glIsEnabled(GL_DEPTH_TEST);
    GLboolean blend_enabled = glIsEnabled(GL_BLEND);
    GLint current_program;
    glGetIntegerv(GL_CURRENT_PROGRAM, &current_program);

    // Disable depth testing and blending for full-screen quad
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_BLEND);

    // Clear the screen
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // Bind the fine render texture
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, _fine_render_texture);

    // Use a simple shader program to display the texture
    // You'll need to create this shader program in initializeGL()
    if (_display_texture_program) {
      _display_texture_program->bind();
      _display_texture_program->setUniformValue("u_texture", 0);

      // Render full-screen quad (vertices from -1 to 1 in NDC)
      float vertices[] = {
              -1.0f, -1.0f,  0.0f, 0.0f,  // Bottom-left
              1.0f, -1.0f,  1.0f, 0.0f,  // Bottom-right
              1.0f,  1.0f,  1.0f, 1.0f,  // Top-right
              -1.0f,  1.0f,  0.0f, 1.0f   // Top-left
      };
      unsigned int indices[] = {
              0, 1, 2,
              2, 3, 0
      };

      // You'll need to create VAO/VBO for this quad in initializeGL()
      _display_quad_vao->bind();
      _display_quad_vbo->bind();
      _display_quad_vbo->write(0, vertices, sizeof(vertices));

      glEnableVertexAttribArray(0);
      glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
      glEnableVertexAttribArray(1);
      glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));

      glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

      _display_quad_vao->release();
      _display_texture_program->release();
    }

    // Restore OpenGL state
    if (depth_test_enabled) glEnable(GL_DEPTH_TEST);
    if (blend_enabled) glEnable(GL_BLEND);
    glUseProgram(current_program);
  }

private:

  GLuint _fine_render_fbo;
  GLuint _fine_render_texture;
  GLuint _fine_render_depth_buffer;
  QOpenGLShaderProgram* _display_texture_program;
  QOpenGLVertexArrayObject* _display_quad_vao;
  QOpenGLBuffer* _display_quad_vbo;
  QOpenGLBuffer* _display_quad_ebo;
};


#endif // THREEDONT_ALTERNATIVE_FRAME_BUFFER_H
