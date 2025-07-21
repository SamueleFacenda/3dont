#include "selection_box.h"

SelectionBox::SelectionBox()
    : _select_mode(NONE) {
  initializeOpenGLFunctions();
  compileProgram();
  initializeBuffers();
}

SelectionBox::~SelectionBox() {
  glDeleteVertexArrays(1, &_vao);
  glDeleteBuffers(1, &_vbo_vertices);
  glDeleteBuffers(1, &_vbo_indices);
}

void SelectionBox::draw() {
  if (_select_mode == NONE) return;
  glDisable(GL_DEPTH_TEST);
  glDepthMask(GL_FALSE);

  _program.bind();
  _program.setUniformValue("box_min", _box.topLeft());
  _program.setUniformValue("box_max", _box.bottomRight());
  
  glBindVertexArray(_vao);
  glDrawElements(GL_LINE_STRIP, 5, GL_UNSIGNED_INT, 0);
  glBindVertexArray(0);

  glEnable(GL_DEPTH_TEST);
  glDepthMask(GL_TRUE);
}

void SelectionBox::click(QPointF p, SelectMode select_mode) {
  _select_mode = select_mode;
  _anchor = p;
  _box = QRectF(p, p);
}

void SelectionBox::drag(QPointF p) {
  _box = QRectF(p, _anchor);
  _box = _box.normalized();
}

void SelectionBox::release() {
  _select_mode = NONE;
  _box.setWidth(0.0f);
  _box.setHeight(0.0f);
}

bool SelectionBox::active() const {
  return _select_mode != NONE;
}

bool SelectionBox::empty() const {
  return _box.isEmpty();
}

const QRectF &SelectionBox::getBox() const {
  return _box;
}

SelectionBox::SelectMode SelectionBox::getType() const {
  return _select_mode;
}

void SelectionBox::compileProgram() {
  std::string vsCode =
          "#version 330 core\n"
          "uniform vec2 box_min;\n"
          "uniform vec2 box_max;\n"
          "layout(location = 0) in vec3 position;\n"
          "void main() {\n"
          "  gl_Position = vec4(position.xy * (box_max - box_min) + box_min, 0.0, 1.0);\n"
          "}\n";
  std::string fsCode =
          "#version 330 core\n"
          "out vec4 fragColor;\n"
          "void main() {\n"
          "  fragColor = vec4(1.0, 1.0, 0.0, 1.0);\n"
          "}\n";
  _program.addShaderFromSourceCode(QOpenGLShader::Vertex, vsCode.c_str());
  _program.addShaderFromSourceCode(QOpenGLShader::Fragment, fsCode.c_str());
  _program.link();
}

void SelectionBox::initializeBuffers() {
  float points[12] = {0.0f, 0.0f, 0.0f,
                      1.0f, 0.0f, 0.0f,
                      1.0f, 1.0f, 0.0f,
                      0.0f, 1.0f, 0.0f};
  unsigned int indices[5] = {0, 1, 2, 3, 0};

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
