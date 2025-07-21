#include "background.h"

Background::Background()
    : _bg_color_top(0.0f, 0.0f, 0.0f, 1.0f),
      _bg_color_bottom(0.23f, 0.23f, 0.44f, 1.0f) {
  initializeOpenGLFunctions();
  compileProgram();
  initializeBuffers();
}

Background::~Background() {
  glDeleteVertexArrays(1, &_vao);
  glDeleteBuffers(1, &_vbo_vertices);
  glDeleteBuffers(1, &_vbo_indices);
}

void Background::draw() {
  glDepthMask(GL_FALSE);
  glDisable(GL_DEPTH_TEST);

  _program.bind();
  _program.setUniformValue("colorBottom", _bg_color_bottom);
  _program.setUniformValue("colorTop", _bg_color_top);
  
  glBindVertexArray(_vao);
  glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
  glBindVertexArray(0);

  glEnable(GL_DEPTH_TEST);
  glDepthMask(GL_TRUE);
}

void Background::setColorTop(QVector4D c) { _bg_color_top = c; }
void Background::setColorBottom(QVector4D c) { _bg_color_bottom = c; }
QVector4D Background::getColorTop() const { return _bg_color_top; }
QVector4D Background::getColorBottom() const { return _bg_color_bottom; }

void Background::compileProgram() {
  std::string vsCode =
          "#version 330 core\n"
          "\n"
          "layout(location = 0) in vec4 position;\n"
          "out vec2 coordinate;\n"
          "void main() {\n"
          "  gl_Position = vec4(2.0 * position.xy - 1.0, 0.0, 1.0);\n"
          "  coordinate = position.xy;\n"
          "}\n";
  std::string fsCode =
          "#version 330 core\n"
          "\n"
          "uniform vec4 colorBottom;\n"
          "uniform vec4 colorTop;\n"
          "in vec2 coordinate;\n"
          "out vec4 fragColor;\n"
          "void main() {\n"
          "  fragColor = mix(colorBottom, colorTop, coordinate.y);\n"
          "}\n";

  _program.addShaderFromSourceCode(QOpenGLShader::Vertex, vsCode.c_str());
  _program.addShaderFromSourceCode(QOpenGLShader::Fragment, fsCode.c_str());
  _program.link();
}

void Background::initializeBuffers() {
  float points[12] = {
          0.0f, 0.0f, 0.0f,
          1.0f, 0.0f, 0.0f,
          1.0f, 1.0f, 0.0f,
          0.0f, 1.0f, 0.0f};
  
  unsigned int indices[6] = {
          0, 1, 2,
          0, 2, 3};

  // Generate and bind VAO
  glGenVertexArrays(1, &_vao);
  glBindVertexArray(_vao);

  // Create and setup vertex buffer
  glGenBuffers(1, &_vbo_vertices);
  glBindBuffer(GL_ARRAY_BUFFER, _vbo_vertices);
  glBufferData(GL_ARRAY_BUFFER, sizeof(points), points, GL_STATIC_DRAW);
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);
  glEnableVertexAttribArray(0);

  // Create and setup index buffer
  glGenBuffers(1, &_vbo_indices);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, _vbo_indices);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

  glBindVertexArray(0);
}
