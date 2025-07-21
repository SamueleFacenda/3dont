#ifndef __FLOORGRID_H__
#define __FLOORGRID_H__
#include "../camera/qt_camera.h"
#include "../utils/opengl_funcs.h"
#include <QOpenGLContext>
#include <QOpenGLShaderProgram>
#include <QOpenGLWidget>
#include <QVector3D>
#include <QVector4D>
#include <exception>
#include <iostream>
#include <math.h>

class SectorCreateException : public std::exception {
public:
  const char *what() const throw();
};

/*! \brief
 *  Acute (angle < pi) sector class (in radians)
 *  Restricting to acute angles to ensure sector intersection results in at
 *    most one sector
 */
class Sector {
public:
  Sector();
  Sector(float start, float end);

  operator bool();
  bool empty() const;

  /* \brief
   * returns true if angle is in sector (including _start and _end)
   */
  bool contains(float angle) const;

  /* \brief
   * returns the sector intersection between this and other
   */
  Sector intersect(const Sector &other) const;

  float getStart() const;
  float getEnd() const;
  static float pi();
  static float rad2deg(float rad);
  static float deg2rad(float deg);
  friend std::ostream &operator<<(std::ostream &, const Sector &);

private:
  /* \brief
   * _start is in [0, 2*pi) and _end is in [_start, _start+2*pi]
   */
  static void normalize(float &start, float &end);

  /* \brief
   * normalizes angle x to be in [0, 2*pi)
   */
  static float normalize(float x);

  float _start;
  float _end;
};

class FloorGrid : protected OpenGLFuncs {
public:
  FloorGrid(QOpenGLWidget *parent);
  ~FloorGrid();

  void draw(const QtCamera &camera);
  void draw(const QtCamera &camera, float z_floor);

  // Getters
  float getCellSize() const;
  float getLineWeight() const;
  QVector4D getLineColor() const;
  QVector4D getFloorColor() const;
  float getFloorLevel() const;
  bool getVisible() const;

  // Setters
  void setLineColor(QVector4D line_color);
  void setFloorColor(QVector4D floor_color);
  void setFloorLevel(float z);
  void setVisible(bool visible);

private:
  void compilePerspProgram();
  void compileOrthoProgram();
  void loadSquare();
  void unloadSquare();

  void drawOrtho(const QtCamera &camera, float z_floor);
  void drawPersp(const QtCamera &camera, float z_floor);

  static float normalizeAngle(float angle);
  float visibleDistance(float cell_size, float projected_cell_size,
                        const QtCamera &camera, float z_floor);
  bool computeHorizon(float &h_lo, float &h_hi, float cell_size,
                      const QtCamera &camera, float z_floor);
  void computeCellSize(float &cell_size, float &line_weight,
                       const QtCamera &camera, float z_floor);

  bool _visible;
  QOpenGLWidget *_parent;
  QVector4D _grid_line_color;
  QVector4D _grid_floor_color;
  float _grid_floor_z;
  float _cell_size;
  float _line_weight;

  QOpenGLShaderProgram _persp_program;
  QOpenGLShaderProgram _ortho_program;

  GLuint _vao;
  GLuint _buffer_square;
  GLuint _buffer_square_indices;
};

#endif
