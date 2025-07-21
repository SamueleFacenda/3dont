#ifndef __POINTCLOUD_H__
#define __POINTCLOUD_H__
#include "../camera/qt_camera.h"
#include "../data/octree.h"
#include "../data/point_attributes.h"
#include "../utils/box3.h"
#include "../utils/opengl_funcs.h"
#include "../utils/timer.h"
#include "selection_box.h"
#include <QOpenGLContext>
#include <QOpenGLShaderProgram>
#include <QOpenGLWidget>
#include <vector>

// from 0 to 1 (1 is best quality)
#define FAST_RENDERING_LOD 0.35f

class PointCloud : protected OpenGLFuncs {
public:
  PointCloud(QOpenGLWidget *parent);
  ~PointCloud();

  void loadPoints(std::vector<float> &positions);
  void clearPoints();
  void loadAttributes(const std::vector<char> &data);
  void clearAttributes();

  void draw(const QtCamera &camera, const SelectionBox *selection_box = nullptr);
  void draw(const unsigned int *indices, unsigned int num_indices, const QtCamera &camera, const SelectionBox *selection_box = nullptr);

  void queryLOD(std::vector<unsigned int> &indices, const QtCamera &camera, float lod_multiplier = 1.0f) const;
  void queryNearPoint(std::vector<unsigned int> &indices, const QPointF &screen_pos, const QtCamera &camera) const;

  void setSelected(const std::vector<unsigned int> &selected_ids);
  void getSelected(std::vector<unsigned int> &selected_ids) const;
  void selectNearPoint(const QPointF &screen_pos, const QtCamera &camera, bool deselect = false);
  void selectInBox(const SelectionBox &selection_box, const QtCamera &camera);
  void clearSelected();

  QVector3D computeSelectionCentroid() const;

  // Getters
  const std::vector<float> &getPositions() const;
  const std::vector<unsigned int> &getSelectedIds() const;
  const PointAttributes &getAttributes() const;
  const vltools::Box3<float> &getBox() const;
  float getFloor() const;
  std::size_t getNumPoints() const;
  std::size_t getNumSelected() const;
  std::size_t getCurrentAttributeIndex() const;
  std::size_t getNumAttributes() const;

  // Setters
  void setPointSize(float point_size);
  void setColorMap(const std::vector<float> &color_map);
  void setColorMapScale(float color_map_min, float color_map_max);
  void setCurrentAttributeIndex(std::size_t index);

private:
  void compileProgram();
  void initColors();
  void updateColors();
  void updateSelectionMask();
  QVector3D screenToWorld(const QPointF &screen_pos, const QtCamera &camera) const;

  QOpenGLWidget *_parent;
  QOpenGLShaderProgram _program;

  float _point_size;
  std::size_t _num_points;
  std::vector<float> _positions;
  std::vector<float> _sizes;
  std::vector<unsigned int> _octree_ids;
  std::vector<unsigned int> _selected_ids;

  Octree _octree;
  PointAttributes _attributes;
  vltools::Box3<float> _full_box;

  GLuint _buffer_positions;
  GLuint _buffer_colors;
  GLuint _buffer_scalars;
  GLuint _buffer_sizes;
  GLuint _buffer_selection_mask;
  GLuint _buffer_octree_ids;

  std::vector<float> _color_map;
  float _color_map_min;
  float _color_map_max;
  bool _color_map_auto;
};

#endif // __POINTCLOUD_H__
