#ifndef __LOOKAT_H__
#define __LOOKAT_H__
#include "opengl_funcs.h"
#include "qt_camera.h"
#include <QOpenGLContext>
#include <QOpenGLShaderProgram>
#include <QWindow>

class LookAt : protected OpenGLFuncs {
public:
  LookAt()
      : _visible(true) {
    initializeOpenGLFunctions();
    compileProgram();
  }
  void draw(const QtCamera &camera) {
    if (!_visible) return;

    QVector3D lookat = camera.getLookAtPosition();
    float d = 0.0625 * camera.getCameraDistance();
    vltools::Box3<float> lookatBox(lookat.x() - d, lookat.x() + d,
                                   lookat.y() - d, lookat.y() + d,
                                   lookat.z() - d, lookat.z() + d);

    _program.bind();
    _program.setUniformValue("mvp", camera.computeMVPMatrix(lookatBox));
    _program.setUniformValue("d", d);
    _program.setUniformValue("lookat", lookat);

    float positions[18] = {
            0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f,
            0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f};
    float colors[18] = {
            1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f,
            0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f,
            0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f};

    GLuint buffer_positions;
    glGenBuffers(1, &buffer_positions);
    glBindBuffer(GL_ARRAY_BUFFER, buffer_positions);
    glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 18, positions, GL_STATIC_DRAW);
    _program.enableAttributeArray("position");
    _program.setAttributeArray("position", GL_FLOAT, 0, 3);

    GLuint buffer_colors;
    glGenBuffers(1, &buffer_colors);
    glBindBuffer(GL_ARRAY_BUFFER, buffer_colors);
    glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 18, colors, GL_STATIC_DRAW);
    _program.enableAttributeArray("color");
    _program.setAttributeArray("color", GL_FLOAT, 0, 3);

    glLineWidth(2.0f);
    glDrawArrays(GL_LINES, 0, 6);

    _program.disableAttributeArray("position");
    _program.disableAttributeArray("color");
    glDeleteBuffers(1, &buffer_positions);
    glDeleteBuffers(1, &buffer_colors);
  }

  void setVisible(bool visible) { _visible = visible; }
  bool getVisible() const { return _visible; }

private:
  void compileProgram() {
    std::string vsCode =
            "#version 330 core\n"
            "uniform float d;\n"
            "uniform vec3 lookat;\n"
            "uniform mat4 mvp;\n"
            "layout(location = 0) in vec3 position;\n"
            "layout(location = 1) in vec3 color;\n"
            "out vec3 vcolor;\n"
            "void main() {\n"
            "  gl_Position = mvp * vec4(d * position + lookat, 1.0);\n"
            "  vcolor = color;\n"
            "}\n";
    std::string fsCode =
            "#version 330 core\n"
            "in vec3 vcolor;\n"
            "out vec4 fragColor;\n"
            "void main() {\n"
            "  fragColor = vec4(vcolor, 1.0);\n"
            "}\n";
    _program.addShaderFromSourceCode(QOpenGLShader::Vertex, vsCode.c_str());
    _program.addShaderFromSourceCode(QOpenGLShader::Fragment, fsCode.c_str());
    _program.link();
  }

  QOpenGLShaderProgram _program;
  bool _visible;
};

#endif // __LOOKAT_H__
