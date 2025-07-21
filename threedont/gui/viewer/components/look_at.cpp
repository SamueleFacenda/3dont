#include "look_at.h"

LookAt::LookAt()
    : _visible(true) {
  initializeOpenGLFunctions();
  compileProgram();
  initializeBuffers();
}

LookAt::~LookAt() {
  glDeleteVertexArrays(1, &_vao);
  glDeleteBuffers(1, &_vbo_positions);
  glDeleteBuffers(1, &_vbo_colors);
}

void LookAt::draw(const QtCamera &camera) {
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

  glBindVertexArray(_vao);

  GLfloat lineWidthRange[2];
  glGetFloatv(GL_LINE_WIDTH_RANGE, lineWidthRange);
  float clampedWidth = std::clamp(2.0f, lineWidthRange[0], lineWidthRange[1]);
  qDebug() << "Line width range:" << lineWidthRange[0] << "to" << lineWidthRange[1];
  glLineWidth(clampedWidth);
  glDrawArrays(GL_LINES, 0, 6);

  glBindVertexArray(0);
}

void LookAt::setVisible(bool visible) { _visible = visible; }
bool LookAt::getVisible() const { return _visible; }

void LookAt::compileProgram() {
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

void LookAt::initializeBuffers() {
  float positions[18] = {
          0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f,
          0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f,
          0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f};
  float colors[18] = {
          1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f,
          0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f,
          0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f};

  // Generate and bind VAO
  glGenVertexArrays(1, &_vao);
  glBindVertexArray(_vao);

  // Create and setup position buffer
  glGenBuffers(1, &_vbo_positions);
  glBindBuffer(GL_ARRAY_BUFFER, _vbo_positions);
  glBufferData(GL_ARRAY_BUFFER, sizeof(positions), positions, GL_STATIC_DRAW);
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);
  glEnableVertexAttribArray(0);

  // Create and setup color buffer
  glGenBuffers(1, &_vbo_colors);
  glBindBuffer(GL_ARRAY_BUFFER, _vbo_colors);
  glBufferData(GL_ARRAY_BUFFER, sizeof(colors), colors, GL_STATIC_DRAW);
  glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, 0);
  glEnableVertexAttribArray(1);

  glBindVertexArray(0);
}
