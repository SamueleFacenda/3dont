#include "point_cloud.h"
#include <QOpenGLWidget>
#include <QVector3D>
#include <algorithm>
#include <limits>

PointCloud::PointCloud(QOpenGLWidget *parent)
    : _parent(parent),
      _point_size(0.0f),
      _num_points(0),
      _vao(0),
      _buffer_positions(0),
      _buffer_colors(0),
      _buffer_sizes(0),
      _buffer_selection_mask(0),
      _buffer_octree_ids(0),
      _texture_color_map(0),
      _color_map(4, 1.0f),
      _color_map_min(0.0f),
      _color_map_max(1.0f),
      _color_map_auto(true) {
  initializeOpenGLFunctions();
  compileProgram();
  initializeVAO();
}

PointCloud::~PointCloud() {
  clearPoints();
  if (_vao)
    glDeleteVertexArrays(1, &_vao);
  if (_texture_color_map)
    glDeleteTextures(1, &_texture_color_map);
}

void PointCloud::loadPoints(std::vector<float> &positions) {
  _positions.swap(positions);
  _num_points = _positions.size() / 3;

  _octree.buildTree(_positions, _sizes, 32);
  _octree_ids.reserve(_num_points);

  _full_box = vltools::Box3<float>();
  _full_box.addPoints(&_positions[0], _num_points);

  // Bind VAO for buffer setup
  glBindVertexArray(_vao);

  // create a buffer for storing position vectors
  glGenBuffers(1, &_buffer_positions);
  glBindBuffer(GL_ARRAY_BUFFER, _buffer_positions);
  glBufferData(GL_ARRAY_BUFFER, sizeof(float) * _positions.size(), (GLvoid *) &_positions[0], GL_STATIC_DRAW);
  GLint position_attrib = _program.attributeLocation("position");
  glVertexAttribPointer(position_attrib, 3, GL_FLOAT, GL_FALSE, 0, 0);
  glEnableVertexAttribArray(position_attrib);

  // create a buffer for storing color vectors
  glGenBuffers(1, &_buffer_colors);

  // create a buffer for storing per point scalars
  glGenBuffers(1, &_buffer_scalars);

  // create buffer for storing centroid sizes
  glGenBuffers(1, &_buffer_sizes);
  glBindBuffer(GL_ARRAY_BUFFER, _buffer_sizes);
  glBufferData(GL_ARRAY_BUFFER, _sizes.size() * sizeof(float), (GLvoid *) &_sizes[0], GL_STATIC_DRAW);
  GLint size_attrib = _program.attributeLocation("size");
  glVertexAttribPointer(size_attrib, 1, GL_FLOAT, GL_FALSE, 0, 0);
  glEnableVertexAttribArray(size_attrib);

  // create buffer for storing selection mask
  glGenBuffers(1, &_buffer_selection_mask);
  glBindBuffer(GL_ARRAY_BUFFER, _buffer_selection_mask);
  glBufferData(GL_ARRAY_BUFFER, _positions.size() / 3 * sizeof(float), nullptr, GL_STATIC_DRAW);
  GLint selection_attrib = _program.attributeLocation("selected");
  glVertexAttribPointer(selection_attrib, 1, GL_FLOAT, GL_FALSE, 0, 0);
  glEnableVertexAttribArray(selection_attrib);

  // create buffer for storing point indices obtained from octree
  glGenBuffers(1, &_buffer_octree_ids);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, _buffer_octree_ids);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, _num_points * sizeof(unsigned int), nullptr, GL_DYNAMIC_DRAW);

  glBindVertexArray(0);

  _attributes.reset();
  initColors();
}

void PointCloud::clearPoints() {
  clearAttributes();
  if (_num_points == 0) return;
  _num_points = 0;
  _positions.clear();
  _sizes.clear();
  _octree_ids.clear();
  _selected_ids.clear();
  _full_box = vltools::Box3<float>();
  glDeleteBuffers(1, &_buffer_positions);
  glDeleteBuffers(1, &_buffer_colors);
  glDeleteBuffers(1, &_buffer_scalars);
  glDeleteBuffers(1, &_buffer_sizes);
  glDeleteBuffers(1, &_buffer_selection_mask);
  glDeleteBuffers(1, &_buffer_octree_ids);
  _octree.buildTree(_positions, _sizes, 32);
  _attributes.reset();
}

void PointCloud::loadAttributes(const std::vector<char> &data) {
  _attributes.set(data, _octree);
  initColors();
}

void PointCloud::loadAttributes(const std::vector<float> &attr, quint64 attr_size, quint64 attr_dim) {
  _attributes.set(attr, attr_size, attr_dim);
  initColors();
}

void PointCloud::clearAttributes() {
  _attributes.reset();
  initColors();
}

void PointCloud::draw(const QtCamera &camera, const SelectionBox *selection_box) {
  queryLOD(_octree_ids, camera, FAST_RENDERING_LOD);
  if (_octree_ids.empty())
    return;
  draw(&_octree_ids[0], _octree_ids.size(), camera, selection_box);
}

void PointCloud::draw(const unsigned int *indices, unsigned int num_points,
                      const QtCamera &camera, const SelectionBox *selection_box) {
  if (_num_points == 0 || num_points == 0) return;

  // box should be in normalized device coordinates
  glEnable(GL_PROGRAM_POINT_SIZE);
  glDisable(GL_DEPTH_TEST);
  glDepthMask(GL_FALSE);
  glEnable(GL_BLEND);

  glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ZERO, GL_ONE);
  glBlendEquation(GL_FUNC_ADD);

  _program.bind();
  _program.setUniformValue("width", (float) _parent->devicePixelRatio() * _parent->width());
  _program.setUniformValue("height", (float) _parent->devicePixelRatio() * _parent->height());
  _program.setUniformValue("point_size", _point_size);
  _program.setUniformValue("mvpMatrix", camera.computeMVPMatrix(_full_box));
  _program.setUniformValue("box_min", selection_box ? selection_box->getBox().topLeft() : QPointF());
  _program.setUniformValue("box_max", selection_box ? selection_box->getBox().bottomRight() : QPointF());
  _program.setUniformValue("eye", camera.getCameraPosition());
  _program.setUniformValue("view", camera.getViewVector());
  _program.setUniformValue("image_t", camera.getTop());
  _program.setUniformValue("box_select_mode", selection_box ? selection_box->getType() : SelectionBox::NONE);
  _program.setUniformValue("projection_mode", camera.getProjectionMode());
  _program.setUniformValue("color_map", 0);
  _program.setUniformValue("scalar_min", _color_map_min);
  _program.setUniformValue("scalar_max", _color_map_max);
  _program.setUniformValue("color_map_n", _color_map.size() / 4.0f);

  glActiveTexture(GL_TEXTURE0 + 0);
  glBindTexture(GL_TEXTURE_2D, _texture_color_map);

  glBindVertexArray(_vao);
  
  // Handle color and scalar attributes dynamically
  int curr_attr_idx = (int) _attributes.currentIndex();
  bool use_color_map = _attributes.dim(curr_attr_idx) == 1;
  bool broadcast_attr = _attributes.size(curr_attr_idx) == 1;
  
  if (use_color_map) {
    GLint color_attrib = _program.attributeLocation("color");
    _program.setAttributeValue("color", QVector4D(1.0f, 1.0f, 1.0f, 1.0f));
    glDisableVertexAttribArray(color_attrib);
  } else if (broadcast_attr) {
    const std::vector<float> &v = _attributes[curr_attr_idx];
    GLint color_attrib = _program.attributeLocation("color");
    _program.setAttributeValue("color", QVector4D(v[0], v[1], v[2], v[3]));
    glDisableVertexAttribArray(color_attrib);
  } else {
    glBindBuffer(GL_ARRAY_BUFFER, _buffer_colors);
    GLint color_attrib = _program.attributeLocation("color");
    glVertexAttribPointer(color_attrib, 4, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(color_attrib);
  }
  
  if (!use_color_map) {
    GLint scalar_attrib = _program.attributeLocation("scalar");
    _program.setAttributeValue("scalar", 1.0f);
    glDisableVertexAttribArray(scalar_attrib);
  } else if (broadcast_attr) {
    GLint scalar_attrib = _program.attributeLocation("scalar");
    _program.setAttributeValue("scalar", _attributes[curr_attr_idx][0]);
    glDisableVertexAttribArray(scalar_attrib);
  } else {
    glBindBuffer(GL_ARRAY_BUFFER, _buffer_scalars);
    GLint scalar_attrib = _program.attributeLocation("scalar");
    glVertexAttribPointer(scalar_attrib, 1, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(scalar_attrib);
  }

  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, _buffer_octree_ids);
  glBufferSubData(GL_ELEMENT_ARRAY_BUFFER, 0, sizeof(unsigned int) * num_points, (GLvoid *) indices);
  glDrawElements(GL_POINTS, num_points, GL_UNSIGNED_INT, nullptr);

  glBindVertexArray(0);

  glEnable(GL_DEPTH_TEST);
  glDepthMask(GL_TRUE);
  glDisable(GL_BLEND);
  glDisable(GL_PROGRAM_POINT_SIZE);
}

void PointCloud::queryLOD(std::vector<unsigned int> &indices, const QtCamera &camera, float lod_multiplier) const {
  if (_num_points == 0) {
    indices.clear();
    return;
  }
  
  float min_z_near = 0.1f;
  if (camera.getProjectionMode() == QtCamera::PERSPECTIVE)
    _octree.getIndices(indices, camera, camera.getVerticalFOV(), -min_z_near,
                       _parent->width(), _parent->height(), lod_multiplier);
  else {
    float t = camera.getCameraDistance() * tan(0.5f * camera.getVerticalFOV());
    float r = (float) _parent->width() / _parent->height() * t;
    _octree.getIndicesOrtho(indices, camera, r, t, _parent->height(), lod_multiplier);
  }
}

void PointCloud::queryNearPoint(std::vector<unsigned int> &indices, const QPointF &screen_pos, const QtCamera &camera) const {
  Octree::ProjectionMode projection_mode =
          camera.getProjectionMode() == QtCamera::PERSPECTIVE
                  ? Octree::PERSPECTIVE
                  : Octree::ORTHOGRAPHIC;
  _octree.getClickIndices(
          indices, screen_pos.x(), _parent->height() - screen_pos.y() - 1.0f, 5.0f,
          _parent->width(), _parent->height(), camera.getVerticalFOV(), 0.1f,
          camera, projection_mode);
}

void PointCloud::setSelected(const std::vector<unsigned int> &selected_ids) {
  // expects indices into original array of points
  _selected_ids.clear();
  for (std::size_t i = 0; i < selected_ids.size(); i++)
    _selected_ids.push_back(_octree.getIndicesR()[selected_ids[i]]);
  std::sort(_selected_ids.begin(), _selected_ids.end()); // increasing order
  updateSelectionMask();
}

void PointCloud::getSelected(std::vector<unsigned int> &selected_ids) const {
  // returns indices into original array of points
  // (prior to reshuffling by octree)
  selected_ids.reserve(_selected_ids.size());
  selected_ids.clear();
  for (std::size_t i = 0; i < _selected_ids.size(); i++)
    if (_selected_ids[i] < _num_points)
      selected_ids.push_back(_octree.getIndices()[_selected_ids[i]]);
    else
      break;
}

void PointCloud::selectNearPoint(const QPointF &screen_pos, const QtCamera &camera, bool deselect) {
  std::vector<unsigned int> new_indices;
  queryNearPoint(new_indices, screen_pos, camera);
  if (new_indices.empty()) return;
  if (deselect)
    removeIndices(_selected_ids, new_indices);
  else
    mergeIndices(_selected_ids, new_indices);
  updateSelectionMask();
}

void PointCloud::deselectNearPoint(const QPointF &screen_pos, const QtCamera &camera) {
  selectNearPoint(screen_pos, camera, true);
}

void PointCloud::selectInBox(const SelectionBox &selection_box, const QtCamera &camera) {
  if (selection_box.getType() == SelectionBox::NONE) return;
  QMatrix4x4 mvp = camera.computeMVPMatrix(_full_box);
  QRectF rect = selection_box.getBox();
  std::vector<unsigned int> new_indices;
  // check all centroids in addition to all points
  for (unsigned int i = 0; i < _positions.size() / 3; i++) {
    float *v = &_positions[3 * i];
    QVector4D p(v[0], v[1], v[2], 1);
    p = mvp * p;
    p /= p.w();
    if (rect.contains(QPointF(p.x(), p.y())) && p.z() > -1.0f && p.z() < 1.0f)
      new_indices.push_back(i);
  }
  if (selection_box.getType() == SelectionBox::ADD)
    mergeIndices(_selected_ids, new_indices);
  else // selection_box.getType() == SelectionBox::SUB
    removeIndices(_selected_ids, new_indices);
  updateSelectionMask();
}

void PointCloud::clearSelected() {
  _selected_ids.clear();
  updateSelectionMask();
}

QVector3D PointCloud::computeSelectionCentroid() {
  // returns bounding box centroid if no points are selected
  QVector3D centroid;
  std::size_t num_selected = 0;
  for (std::size_t i = 0; i < _selected_ids.size(); i++) {
    if (_selected_ids[i] >= _num_points) break;
    num_selected++;
    float *v = &_positions[3 * _selected_ids[i]];
    centroid += QVector3D(v[0], v[1], v[2]);
  }
  if (num_selected == 0)
    return QVector3D(0.5f * (_full_box.x(0) + _full_box.x(1)),
                     0.5f * (_full_box.y(0) + _full_box.y(1)),
                     0.5f * (_full_box.z(0) + _full_box.z(1)));
  else
    return centroid / num_selected;
}

// Getters
const std::vector<float> &PointCloud::getPositions() const {
  return _positions;
}

const std::vector<unsigned int> &PointCloud::getSelectedIds() const {
  return _selected_ids;
}

const PointAttributes &PointCloud::getAttributes() const {
  return _attributes;
}

const vltools::Box3<float> &PointCloud::getBox() const {
  return _full_box;
}

float PointCloud::getFloor() const {
  return _num_points == 0 ? 0.0f : _full_box.min(2);
}

std::size_t PointCloud::getNumPoints() const {
  return _num_points;
}

std::size_t PointCloud::getNumSelected() const {
  return countSelected(_selected_ids, (unsigned int) _num_points);
}

std::size_t PointCloud::getCurrentAttributeIndex() const {
  return _attributes.currentIndex();
}

std::size_t PointCloud::getNumAttributes() const {
  return _attributes.numAttributes();
}

// Setters
void PointCloud::setPointSize(float point_size) {
  _point_size = point_size;
}

void PointCloud::setColorMap(const std::vector<float> &color_map) {
  _color_map.resize(color_map.size());
  std::copy(color_map.begin(), color_map.end(), _color_map.begin());
  initColors();
}

void PointCloud::setColorMapScale(float color_map_min, float color_map_max) {
  // scale_min >= scale_max understood to mean auto
  if (color_map_min >= color_map_max) {
    _color_map_auto = true;
    // automatically set [_color_map_min, _color_map_max]
    // according to the attribute type
    int curr_attr_idx = (int) _attributes.currentIndex();
    bool use_color_map = _attributes.dim(curr_attr_idx) == 1;
    bool broadcast_attr = _attributes.size(curr_attr_idx) == 1;
    const std::vector<float> &attr = _attributes[curr_attr_idx];
    if (!use_color_map) {
      _color_map_min = 0.0f;
      _color_map_max = 1.0f;
    } else if (broadcast_attr) {
      _color_map_min = attr[0] - 1.0f;
      _color_map_max = attr[0] + 1.0f;
    } else {
      _color_map_min = std::numeric_limits<float>::max();
      _color_map_max = std::numeric_limits<float>::lowest();
      for (std::size_t i = 0; i < attr.size(); i++) {
        if (attr[i] == attr[i]) { // skip if attr[i] is NaN
          _color_map_min = qMin(_color_map_min, attr[i]);
          _color_map_max = qMax(_color_map_max, attr[i]);
        }
      }
    }
  } else {
    _color_map_auto = false;
    _color_map_min = color_map_min;
    _color_map_max = color_map_max;
  }
}

void PointCloud::setCurrentAttributeIndex(std::size_t index) {
  bool index_changed = index != _attributes.currentIndex();
  _attributes.setCurrentIndex(index);
  if (index_changed)
    initColors();
}

// Private methods
void PointCloud::compileProgram() {
  std::string vsCode =
          "#version 330 core\n"
          "\n"
          "uniform float point_size;\n"
          "uniform float width;\n"
          "uniform float height;\n"
          "uniform vec2 box_min;\n"
          "uniform vec2 box_max;\n"
          "uniform int draw_selection_box;\n"
          "uniform int box_select_mode;  // 0 - add, 1 - remove, 2 - no box\n"
          "uniform mat4 mvpMatrix;\n"
          "uniform sampler2D color_map;\n"
          "uniform float scalar_min;\n"
          "uniform float scalar_max;\n"
          "uniform float color_map_n;\n"
          "uniform int projection_mode;\n"
          "uniform vec3 eye;\n"
          "uniform vec3 view;\n"
          "uniform float image_t;\n"
          "\n"
          "layout(location = 0) in vec3 position;\n"
          "layout(location = 1) in vec4 color;\n"
          "layout(location = 2) in float scalar;\n"
          "layout(location = 3) in float size;\n"
          "layout(location = 4) in float selected;\n"
          "\n"
          "out vec4 frag_color;\n"
          "out vec2 frag_center;\n"
          "out float inner_radius;\n"
          "out float outer_radius;\n"
          "\n"
          "void main() {\n"
          "  vec4 p = mvpMatrix * vec4(position, 1.0);\n"
          "  frag_center = 0.5 * (p.xy / p.w + 1.0) * vec2(width, height);\n"
          "  gl_Position = p;\n"
          "  p /= p.w;\n"
          "  float tex_coord = clamp((scalar - scalar_min) / (scalar_max - scalar_min), 0.0, 1.0);\n"
          "  tex_coord = (tex_coord - 0.5) * (color_map_n - 1.0) / color_map_n + 0.5;\n"
          "  vec4 color_s = tex_coord != tex_coord ? vec4(0, 0, 0, 1) : texture(color_map, vec2(tex_coord, 0.5));\n"
          "  vec4 color_r = color_s * color;\n"
          "  if (box_select_mode == 2)\n"
          "    frag_color = selected == 1.0 ? vec4(1, 1, 0, 1) : color_r;\n"
          "  else {\n"
          "    bool inBox = p.x < box_max.x && p.x > box_min.x && p.y < box_max.y && p.y > box_min.y && p.z < 1.0 && p.z > -1.0;\n"
          "    if (box_select_mode == 0)\n"
          "      frag_color = (inBox || selected == 1.0) ? vec4(1, 1, 0, 1) : color_r;\n"
          "    else\n"
          "      frag_color = (!inBox && selected == 1.0) ? vec4(1, 1, 0, 1) : color_r;\n"
          "  }\n"
          "  float d = abs(dot(position.xyz - eye,view));\n"
          "  if (projection_mode == 1) d = 1.0;\n"
          "  if (size == 0.0) {\n"
          "    inner_radius = point_size / d * height / (2.0 * image_t);\n"
          "    outer_radius = inner_radius + 1.0;\n"
          "  } else {\n"
          "    inner_radius = 0.5 * size / d * height / (2.0 * image_t);\n"
          "    outer_radius = max(1.0, 2.0 * inner_radius);\n"
          "  }\n"
          "  gl_PointSize = outer_radius * 2.0;\n"
          "}\n";
  std::string fsCode =
          "#version 330 core\n"
          "\n"
          "in vec4 frag_color;\n"
          "in float inner_radius;\n"
          "in float outer_radius;\n"
          "\n"
          "out vec4 fragColor;\n"
          "\n"
          "void main() {\n"
          "  vec2 centeredCoord = gl_PointCoord - vec2(0.5);\n"
          "  float dist = length(centeredCoord) * outer_radius * 2.0;\n"
          "  float weight = clamp((outer_radius - dist) / (outer_radius - inner_radius), 0.0, 1.0);\n"
          "  fragColor = frag_color * vec4(1.0, 1.0, 1.0, weight);\n"
          "}\n";

  _program.addShaderFromSourceCode(QOpenGLShader::Vertex, vsCode.c_str());
  _program.addShaderFromSourceCode(QOpenGLShader::Fragment, fsCode.c_str());
  _program.link();
}

void PointCloud::initColors() {
  // prepare OpenGL buffers and textures for current attribute set
  // four cases:           use colormap  upload array to gpu
  //   1. scalar           Y             N
  //   2. rgba             N             N
  //   3. array of scalar  Y             Y
  //   4. array of rgba    N             Y
  int curr_attr_idx = (int) _attributes.currentIndex();
  bool use_color_map = _attributes.dim(curr_attr_idx) == 1;
  bool broadcast_attr = _attributes.size(curr_attr_idx) == 1;
  const std::vector<float> &attr = _attributes[curr_attr_idx];

  if (_color_map_auto)
    setColorMapScale(1.0f, 0.0f);

  glActiveTexture(GL_TEXTURE0);
  glDeleteTextures(1, &_texture_color_map);
  glGenTextures(1, &_texture_color_map);

  glBindTexture(GL_TEXTURE_2D, _texture_color_map);

  if (use_color_map) {
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, (int)_color_map.size() / 4, 1, 0, GL_RGBA,
                 GL_FLOAT, &_color_map[0]);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  } else {
    float white[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 1, 1, 0, GL_RGBA, GL_FLOAT, white);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  }

  // Upload attribute buffer (scalar or RGBA) if needed
  if (!broadcast_attr) {
    GLuint attr_buffer = use_color_map ? _buffer_scalars : _buffer_colors;
    glBindBuffer(GL_ARRAY_BUFFER, attr_buffer);
    glBufferData(GL_ARRAY_BUFFER, sizeof(float) * attr.size(), &attr[0], GL_STATIC_DRAW);
  }

  glBindTexture(GL_TEXTURE_2D, 0);
}

void PointCloud::updateSelectionMask() {
  std::vector<float> selection_mask(_positions.size() / 3, 0.0f);
  for (std::size_t i = 0; i < _selected_ids.size(); i++)
    selection_mask[_selected_ids[i]] = 1.0f;
  glBindBuffer(GL_ARRAY_BUFFER, _buffer_selection_mask);
  glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(float) * _positions.size() / 3, (GLvoid *) &selection_mask[0]);
}

void PointCloud::initializeVAO() {
  glGenVertexArrays(1, &_vao);
}

void PointCloud::mergeIndices(std::vector<unsigned int> &x,
                             const std::vector<unsigned int> &y,
                             bool xor_merge) {
  // assumes x and y are sorted in increasing order
  std::vector<unsigned int> temp;
  temp.reserve(x.size() + y.size());
  std::size_t y_idx = 0;
  for (std::size_t i = 0; i < x.size(); i++) {
    while (y_idx < y.size() && y[y_idx] < x[i]) temp.push_back(y[y_idx++]);
    if (y_idx == y.size()) {
      temp.insert(temp.end(), &x[i], &x[i] + (x.size() - i));
      break;
    }
    if (y[y_idx] == x[i]) {
      y_idx++;
      if (xor_merge) continue;
    }
    temp.push_back(x[i]);
  }
  if (y_idx < y.size())
    temp.insert(temp.end(), &y[y_idx], &y[y_idx] + (y.size() - y_idx));
  x.swap(temp);
}

void PointCloud::removeIndices(std::vector<unsigned int> &x,
                              const std::vector<unsigned int> &y) {
  std::vector<unsigned int> temp;
  temp.reserve(x.size());
  std::size_t y_idx = 0;
  for (std::size_t i = 0; i < x.size(); i++) {
    while (y_idx < y.size() && y[y_idx] < x[i]) y_idx++;
    if (y_idx == y.size()) {
      temp.insert(temp.end(), &x[i], &x[i] + (x.size() - i));
      break;
    }
    if (y[y_idx] == x[i])
      y_idx++;
    else
      temp.push_back(x[i]);
  }
  x.swap(temp);
}

std::size_t PointCloud::countSelected(const std::vector<unsigned int> &x,
                                     unsigned int y) {
  // note _selected_ids may contain centroid, we desire non-centroid ids
  // find number of items in _selected_ids (sorted)
  // that is less than _num_points
  if (x.empty()) return 0;
  std::size_t a = 0;
  std::size_t b = x.size() - 1;
  if (x[b] < y) return b + 1;
  if (x[a] >= y) return 0;
  // at this point, we know x.size() >= 2
  // invariances:
  // 1. x[b] >= y
  // 2. x[a] < y
  while (b > a + 1) {
    std::size_t c = (a + b) / 2;
    if (x[c] < y)
      a = c;
    else
      b = c;
  }
  return a + 1;
}
