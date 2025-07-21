#include "point_cloud.h"
#include <QOpenGLWidget>
#include <QVector3D>
#include <algorithm>

PointCloud::PointCloud(QOpenGLWidget *parent)
    : _parent(parent),
      _point_size(0.0f),
      _num_points(0),
      _buffer_positions(0),
      _buffer_colors(0),
      _buffer_sizes(0),
      _buffer_selection_mask(0),
      _buffer_octree_ids(0),
      _color_map(4, 1.0f),
      _color_map_min(0.0f),
      _color_map_max(1.0f),
      _color_map_auto(true) {
  initializeOpenGLFunctions();
  compileProgram();
}

PointCloud::~PointCloud() {
  clearPoints();
}

void PointCloud::loadPoints(std::vector<float> &positions) {
  // warning: this function modifies positions and colors
  _positions.swap(positions);
  _num_points = _positions.size() / 3;

  _octree.buildTree(_positions, _sizes, 32);
  _octree_ids.reserve(_num_points);

  _full_box = vltools::Box3<float>();
  _full_box.addPoints(&_positions[0], _num_points);

  // create a buffer for storing position vectors
  glGenBuffers(1, &_buffer_positions);
  glBindBuffer(GL_ARRAY_BUFFER, _buffer_positions);
  glBufferData(GL_ARRAY_BUFFER, sizeof(float) * _positions.size(), (GLvoid *) &_positions[0], GL_STATIC_DRAW);

  // create a buffer for storing color vectors
  glGenBuffers(1, &_buffer_colors);

  // create a buffer for storing per point scalars
  glGenBuffers(1, &_buffer_scalars);

  // create buffer for storing centroid sizes
  glGenBuffers(1, &_buffer_sizes);
  glBindBuffer(GL_ARRAY_BUFFER, _buffer_sizes);
  glBufferData(GL_ARRAY_BUFFER, _sizes.size() * sizeof(float), (GLvoid *) &_sizes[0], GL_STATIC_DRAW);

  // create buffer for storing selection mask
  glGenBuffers(1, &_buffer_selection_mask);
  glBindBuffer(GL_ARRAY_BUFFER, _buffer_selection_mask);
  glBufferData(GL_ARRAY_BUFFER, _positions.size() / 3 * sizeof(float), nullptr, GL_STATIC_DRAW);

  // create buffer for storing point indices obtained from octree
  glGenBuffers(1, &_buffer_octree_ids);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, _buffer_octree_ids);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, _sizes.size() * sizeof(unsigned int), nullptr, GL_DYNAMIC_DRAW);

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
  updateColors();
}

void PointCloud::clearAttributes() {
  _attributes.reset();
  updateColors();
}

void PointCloud::draw(const QtCamera &camera, const SelectionBox *selection_box) {
  if (_num_points == 0) return;

  // get points at appropriate LOD
  queryLOD(_octree_ids, camera, FAST_RENDERING_LOD);

  // upload to GPU
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, _buffer_octree_ids);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, _octree_ids.size() * sizeof(unsigned int),
               &_octree_ids[0], GL_DYNAMIC_DRAW);

  draw(&_octree_ids[0], _octree_ids.size(), camera, selection_box);
}

void PointCloud::draw(const unsigned int *indices, unsigned int num_indices,
                      const QtCamera &camera, const SelectionBox *selection_box) {
  if (_num_points == 0 || num_indices == 0) return;

  // Set up OpenGL state
  glEnable(GL_DEPTH_TEST);
  glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);

  // Use shader program
  _program.bind();

  // Set uniforms
  QMatrix4x4 mvp = camera.computeMVPMatrix(_full_box);
  _program.setUniformValue("mvp", mvp);
  _program.setUniformValue("point_size", _point_size);

  // Bind vertex attributes
  _program.enableAttributeArray("position");
  glBindBuffer(GL_ARRAY_BUFFER, _buffer_positions);
  _program.setAttributeBuffer("position", GL_FLOAT, 0, 3);

  _program.enableAttributeArray("color");
  glBindBuffer(GL_ARRAY_BUFFER, _buffer_colors);
  _program.setAttributeBuffer("color", GL_FLOAT, 0, 4);

  _program.enableAttributeArray("size");
  glBindBuffer(GL_ARRAY_BUFFER, _buffer_sizes);
  _program.setAttributeBuffer("size", GL_FLOAT, 0, 1);

  _program.enableAttributeArray("selection_mask");
  glBindBuffer(GL_ARRAY_BUFFER, _buffer_selection_mask);
  _program.setAttributeBuffer("selection_mask", GL_FLOAT, 0, 1);

  // Draw points
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, _buffer_octree_ids);
  glDrawElements(GL_POINTS, num_indices, GL_UNSIGNED_INT, (void *) 0);

  // Clean up
  _program.disableAttributeArray("position");
  _program.disableAttributeArray("color");
  _program.disableAttributeArray("size");
  _program.disableAttributeArray("selection_mask");
  _program.release();

  glDisable(GL_VERTEX_PROGRAM_POINT_SIZE);
}

void PointCloud::queryLOD(std::vector<unsigned int> &indices, const QtCamera &camera, float lod_multiplier) const {
  if (_num_points == 0) {
    indices.clear();
    return;
  }

  float vfov = camera.getVerticalFOV();
  float aspect = camera.getAspectRatio();
  float z_near = 0.1f; // Default near plane

  _octree.getIndices(indices, camera, vfov, z_near,
                     _parent->width(), _parent->height(), lod_multiplier);
}

void PointCloud::queryNearPoint(std::vector<unsigned int> &indices, const QPointF &screen_pos, const QtCamera &camera) const {
  if (_num_points == 0) {
    indices.clear();
    return;
  }

  float screen_radius = 10.0f; // pixels
  float vfov = camera.getVerticalFOV();
  float near_clip = 0.1f;

  _octree.getClickIndices(indices, screen_pos.x(), screen_pos.y(), screen_radius,
                          _parent->width(), _parent->height(), vfov, near_clip,
                          camera, camera.getProjectionMode() == QtCamera::PERSPECTIVE ? Octree::PERSPECTIVE : Octree::ORTHOGRAPHIC);
}

void PointCloud::setSelected(const std::vector<unsigned int> &selected_ids) {
  _selected_ids = selected_ids;
  updateSelectionMask();
}

void PointCloud::getSelected(std::vector<unsigned int> &selected_ids) const {
  selected_ids = _selected_ids;
}

void PointCloud::selectNearPoint(const QPointF &screen_pos, const QtCamera &camera, bool deselect) {
  std::vector<unsigned int> near_indices;
  queryNearPoint(near_indices, screen_pos, camera);

  if (!near_indices.empty()) {
    unsigned int point_id = near_indices[0];

    auto it = std::find(_selected_ids.begin(), _selected_ids.end(), point_id);
    if (it != _selected_ids.end()) {
      if (deselect)
        _selected_ids.erase(it);
    } else {
      if (!deselect)
        _selected_ids.push_back(point_id);
    }
    updateSelectionMask();
  }
}

void PointCloud::selectInBox(const SelectionBox &selection_box, const QtCamera &camera) {
  // Implementation would require box intersection testing with points
  // This is a simplified version
  updateSelectionMask();
}

void PointCloud::clearSelected() {
  _selected_ids.clear();
  updateSelectionMask();
}

QVector3D PointCloud::computeSelectionCentroid() const {
  if (_selected_ids.empty()) return QVector3D(0, 0, 0);

  QVector3D centroid(0, 0, 0);
  for (unsigned int id: _selected_ids)
    if (id < _num_points)
      centroid += QVector3D(_positions[3 * id], _positions[3 * id + 1], _positions[3 * id + 2]);
  return centroid / _selected_ids.size();
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
  return _full_box.min(2); // Z minimum
}

std::size_t PointCloud::getNumPoints() const {
  return _num_points;
}

std::size_t PointCloud::getNumSelected() const {
  return _selected_ids.size();
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
  _color_map = color_map;
  updateColors();
}

void PointCloud::setColorMapScale(float color_map_min, float color_map_max) {
  _color_map_min = color_map_min;
  _color_map_max = color_map_max;
  _color_map_auto = false;
  updateColors();
}

void PointCloud::setCurrentAttributeIndex(std::size_t index) {
  _attributes.setCurrentIndex(index);
  updateColors();
}

// Private methods
void PointCloud::compileProgram() {
  QString vertexShader = R"(
        #version 330 core
        layout(location = 0) in vec3 position;
        layout(location = 1) in vec4 color;
        layout(location = 2) in float size;
        layout(location = 3) in float selection_mask;

        uniform mat4 mvp;
        uniform float point_size;

        out vec4 v_color;
        out float v_selection;

        void main() {
            gl_Position = mvp * vec4(position, 1.0);
            gl_PointSize = point_size * size;
            v_color = color;
            v_selection = selection_mask;
        }
    )";

  QString fragmentShader = R"(
        #version 330 core
        in vec4 v_color;
        in float v_selection;

        out vec4 fragColor;

        void main() {
            vec4 color = v_color;
            if (v_selection > 0.5) {
                color = mix(color, vec4(1.0, 1.0, 0.0, 1.0), 0.5);
            }
            fragColor = color;
        }
    )";

  _program.addShaderFromSourceCode(QOpenGLShader::Vertex, vertexShader);
  _program.addShaderFromSourceCode(QOpenGLShader::Fragment, fragmentShader);
  _program.link();
}

void PointCloud::initColors() {
  updateColors();
}

void PointCloud::updateColors() {
  if (_num_points == 0) return;

  std::vector<float> colors(_num_points * 4, 1.0f); // RGBA

  // Apply color mapping based on attributes
  for (size_t i = 0; i < _num_points; ++i) {
    // Default white color
    colors[4 * i] = 1.0f;     // R
    colors[4 * i + 1] = 1.0f; // G
    colors[4 * i + 2] = 1.0f; // B
    colors[4 * i + 3] = 1.0f; // A
  }

  // Upload colors to GPU
  glBindBuffer(GL_ARRAY_BUFFER, _buffer_colors);
  glBufferData(GL_ARRAY_BUFFER, colors.size() * sizeof(float), &colors[0], GL_STATIC_DRAW);
}

void PointCloud::updateSelectionMask() {
  if (_num_points == 0) return;

  std::vector<float> mask(_num_points, 0.0f);

  for (unsigned int id: _selected_ids)
    if (id < _num_points)
      mask[id] = 1.0f;

  glBindBuffer(GL_ARRAY_BUFFER, _buffer_selection_mask);
  glBufferData(GL_ARRAY_BUFFER, mask.size() * sizeof(float), &mask[0], GL_STATIC_DRAW);
}

QVector3D PointCloud::screenToWorld(const QPointF &screen_pos, const QtCamera &camera) const {
  // Convert screen coordinates to world coordinates
  // This is a simplified implementation
  return QVector3D(screen_pos.x(), screen_pos.y(), 0.0f);
}
