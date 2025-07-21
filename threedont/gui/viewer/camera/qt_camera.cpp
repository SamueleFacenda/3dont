#include "qt_camera.h"
#include <limits>

QtCamera::QtCamera() : Camera(),
                       _vfov(pi() / 4.0f), _aspect_ratio(1.0f), _projection_mode(PERSPECTIVE), _view_axis(ARBITRARY_AXIS) {}

QtCamera::QtCamera(float xmin, float xmax, float ymin, float ymax, float zmin, float zmax) : Camera(xmin, xmax, ymin, ymax, zmin, zmax),
                                                                                             _vfov(pi() / 4.0f), _aspect_ratio(1.0f), _projection_mode(PERSPECTIVE), _view_axis(ARBITRARY_AXIS) {}

QtCamera::QtCamera(const QVector3D &lo, const QVector3D &hi) : Camera(lo.x(), hi.x(), lo.y(), hi.y(), lo.z(), hi.z()),
                                                               _vfov(pi() / 4.0f), _aspect_ratio(1.0f), _projection_mode(PERSPECTIVE), _view_axis(ARBITRARY_AXIS) {}

QtCamera::QtCamera(const vltools::Box3<float> &box) : Camera(box.x(0), box.x(1), box.y(0), box.y(1), box.z(0), box.z(1)),
                                                      _vfov(pi() / 4.0f), _aspect_ratio(1.0f), _projection_mode(PERSPECTIVE), _view_axis(ARBITRARY_AXIS) {}

QVector3D QtCamera::getCameraPosition() const {
  float p[3];
  Camera::getCameraPosition(p);
  return QVector3D(p[0], p[1], p[2]);
}

QVector3D QtCamera::getLookAtPosition() const {
  float p[3];
  Camera::getLookAtPosition(p);
  return QVector3D(p[0], p[1], p[2]);
}

QVector3D QtCamera::getRightVector() const {
  float v[3];
  Camera::getRightVector(v);
  return QVector3D(v[0], v[1], v[2]);
}

QVector3D QtCamera::getUpVector() const {
  float v[3];
  Camera::getUpVector(v);
  return QVector3D(v[0], v[1], v[2]);
}

QVector3D QtCamera::getViewVector() const {
  float v[3];
  Camera::getViewVector(v);
  return QVector3D(v[0], v[1], v[2]);
}

void QtCamera::setLookAtPosition(const QVector3D &p) {
  float buf[3] = {p.x(), p.y(), p.z()};
  Camera::setLookAtPosition(buf);
}

void QtCamera::pan(QVector2D delta) {
  // delta in ndc scale
  if (delta.x() == 0.0f && delta.y() == 0.0f)
    return;
  float h = getCameraDistance() * tan(0.5f * _vfov);
  float w = _aspect_ratio * h;
  delta *= QVector2D(w, h) / getPanRate();
  Camera::pan(delta.x(), delta.y());
}

void QtCamera::rotate(QVector2D delta) {
  // delta in screen space pixel scale
  if (delta.x() == 0.0f && delta.y() == 0.0f)
    return;
  if (_view_axis != ARBITRARY_AXIS)
    _view_axis = ARBITRARY_AXIS;
  Camera::rotate(delta.x(), delta.y());
}

float QtCamera::getTop() const {
  float t = tan(0.5f * _vfov);
  if (_projection_mode == ORTHOGRAPHIC)
    t *= getCameraDistance();
  return t;
}

float QtCamera::getRight() const {
  return _aspect_ratio * getTop();
}

float QtCamera::getAspectRatio() const {
  return _aspect_ratio;
}

float QtCamera::getVerticalFOV() const {
  return _vfov;
}

QtCamera::ProjectionMode QtCamera::getProjectionMode() const {
  return _projection_mode;
}

QtCamera::ViewAxis QtCamera::getViewAxis() const {
  return _view_axis;
}

void QtCamera::setAspectRatio(float aspect_ratio) {
  _aspect_ratio = aspect_ratio;
}

void QtCamera::setVerticalFOV(float vfov) {
  _vfov = vfov;
}

void QtCamera::setProjectionMode(ProjectionMode mode) {
  _projection_mode = mode;
}

void QtCamera::setViewAxis(ViewAxis axis) {
  if (axis == X_AXIS) {
    setPhi(0.0f);
    setTheta(0.0f);
  } else if (axis == Y_AXIS) {
    setPhi(-0.5f * pi());
    setTheta(0.0f);
  } else if (axis == Z_AXIS) {
    setPhi(-0.5f * pi());
    setTheta(0.5f * pi());
  }
  _view_axis = axis;
}

QMatrix4x4 QtCamera::computeMVPMatrix(const vltools::Box3<float> &box) const {
  QMatrix4x4 matrix;
  matrix.setToIdentity();
  float d_near, d_far;
  computeNearFar(d_near, d_far, box);
  if (_projection_mode == PERSPECTIVE) {
    matrix.perspective(_vfov / pi() * 180.0f, _aspect_ratio, std::max(0.1f, 0.8f * d_near), 1.2f * d_far);
  } else {
    float t = getCameraDistance() * tan(0.5f * _vfov);
    float r = _aspect_ratio * t;
    matrix.ortho(-r, r, -t, t, 0.8f * d_near, 1.2f * d_far);
  }
  matrix.lookAt(getCameraPosition(), getLookAtPosition(), getUpVector());
  return matrix;
}

void QtCamera::computeNearFar(float &d_near, float &d_far, const vltools::Box3<float> &box) const {
  d_near = std::numeric_limits<float>::max();
  d_far = -std::numeric_limits<float>::max();

  QVector3D view = getViewVector();
  QVector3D eye = getCameraPosition();
  for (std::size_t i = 0; i < 2; i++) {
    for (std::size_t j = 0; j < 2; j++) {
      for (std::size_t k = 0; k < 2; k++) {
        QVector3D corner(box.x(i), box.y(j), box.z(k));
        float t = QVector3D::dotProduct(corner - eye, -view);
        d_near = std::min(d_near, t);
        d_far = std::max(d_far, t);
      }
    }
  }
}

float QtCamera::pi() {
  return atan2(0.0, -1.0);
}
