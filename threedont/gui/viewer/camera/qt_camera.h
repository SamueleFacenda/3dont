#ifndef QMOUSECONTROLLEDCAMERA_H
#define QMOUSECONTROLLEDCAMERA_H
#include "../utils/box3.h"
#include "camera.h"
#include <QMatrix4x4>
#include <QVector2D>
#include <QVector3D>
#include <math.h>

class QtCamera : public Camera {
  // adapter class that utilizes Qt features
  // (Parent class Camera is Qt-independent)
public:
  enum ProjectionMode { PERSPECTIVE = 0,
                        ORTHOGRAPHIC = 1 };
  enum ViewAxis { ARBITRARY_AXIS,
                  X_AXIS,
                  Y_AXIS,
                  Z_AXIS };

  QtCamera();
  QtCamera(float xmin, float xmax, float ymin, float ymax, float zmin, float zmax);
  QtCamera(const QVector3D &lo, const QVector3D &hi);
  QtCamera(const vltools::Box3<float> &box);

  QVector3D getCameraPosition() const;
  QVector3D getLookAtPosition() const;
  QVector3D getRightVector() const;
  QVector3D getUpVector() const;
  QVector3D getViewVector() const;
  void setLookAtPosition(const QVector3D &p);

  using Camera::computeRightVector;
  using Camera::computeUpVector;
  using Camera::computeViewVector;
  using Camera::getCameraDistance;
  using Camera::getCameraPosition;
  using Camera::getLookAtPosition;
  using Camera::getPanRate;
  using Camera::getPhi;
  using Camera::getRightVector;
  using Camera::getRotateRate;
  using Camera::getTheta;
  using Camera::getUpVector;
  using Camera::getViewVector;
  using Camera::getZoomRate;
  using Camera::pan;
  using Camera::restore;
  using Camera::rotate;
  using Camera::save;
  using Camera::setCameraDistance;
  using Camera::setLookAtPosition;
  using Camera::setPanRate;
  using Camera::setPhi;
  using Camera::setRotateRate;
  using Camera::setTheta;
  using Camera::setZoomRate;
  using Camera::zoom;

  void pan(QVector2D delta);
  void rotate(QVector2D delta);
  float getTop() const;
  float getRight() const;
  float getAspectRatio() const;
  float getVerticalFOV() const;
  ProjectionMode getProjectionMode() const;
  ViewAxis getViewAxis() const;

  void setAspectRatio(float aspect_ratio);
  void setVerticalFOV(float vfov);
  void setProjectionMode(ProjectionMode mode);
  void setViewAxis(ViewAxis axis);

  QMatrix4x4 computeMVPMatrix(const vltools::Box3<float> &box) const;

private:
  void computeNearFar(float &d_near, float &d_far, const vltools::Box3<float> &box) const;
  static float pi();
  float _vfov;         // vertical fov in radians
  float _aspect_ratio; // width / height
  ProjectionMode _projection_mode;
  ViewAxis _view_axis;
};

#endif // QMOUSECONTROLLEDCAMERA_H
