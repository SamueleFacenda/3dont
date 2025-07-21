#include "camera.h"
#include <algorithm>
#include <math.h>

Camera::Camera()
    : _theta(0.0f),
      _phi(0.0f),
      _d(1.0f),
      _panRate(2.0f / 300),     // 2.0 per 300 pixels
      _zoomRate(0.8f),
      _rotateRate(PI / 2 / 256) // PI/2 per 256 pixels
{
  _lookAt[0] = _lookAt[1] = _lookAt[2] = 0.0f;
  save();
}

Camera::Camera(float xmin, float xmax, float ymin, float ymax, float zmin, float zmax)
    : _theta(PI / 4.0f),
      _phi(PI / 4.0f),
      _zoomRate(0.8f),
      _rotateRate(PI / 2 / 256) {
  // set _lookAt to center bottom
  _lookAt[0] = (xmin + xmax) / 2.0f;
  _lookAt[1] = (ymin + ymax) / 2.0f;
  _lookAt[2] = zmin;

  // set _d to length of widest span
  _d = (std::max) (xmax - xmin, (std::max) (ymax - ymin, zmax - zmin));

  _panRate = _d / 300;

  save();
}

void Camera::rotate(float dx, float dy) {
  _phi -= _rotateRate * dx;
  _theta += _rotateRate * dy;
}

void Camera::pan(float dx, float dy) {
  float x[3];
  float up[3];
  computeRightVector(x, _theta, _phi);
  computeUpVector(up, _theta, _phi);
  for (int i = 0; i < 3; i++)
    _lookAt[i] += _panRate * (-x[i] * dx + up[i] * dy);
}

void Camera::zoom(float dx) {
  _d = (std::max) (0.1f, _d * (float) pow(_zoomRate, dx));
}

void Camera::save() {
  std::copy(_lookAt, _lookAt + 3, _saved_lookAt);
  _saved_theta = _theta;
  _saved_phi = _phi;
  _saved_d = _d;
}

void Camera::restore() {
  std::copy(_saved_lookAt, _saved_lookAt + 3, _lookAt);
  _theta = _saved_theta;
  _phi = _saved_phi;
  _d = _saved_d;
}

void Camera::getCameraPosition(float (&p)[3]) const {
  computeCameraPosition(p, _lookAt, _theta, _phi, _d);
}

void Camera::getLookAtPosition(float (&p)[3]) const {
  std::copy(_lookAt, _lookAt + 3, p);
}

void Camera::getRightVector(float (&v)[3]) const {
  computeRightVector(v, _theta, _phi);
}

void Camera::getUpVector(float (&v)[3]) const {
  computeUpVector(v, _theta, _phi);
}

void Camera::getViewVector(float (&v)[3]) const {
  computeViewVector(v, _theta, _phi);
}

float Camera::getCameraDistance() const { return _d; }
float Camera::getTheta() const { return _theta; }
float Camera::getPhi() const { return _phi; }
float Camera::getPanRate() const { return _panRate; }
float Camera::getRotateRate() const { return _rotateRate; }
float Camera::getZoomRate() const { return _zoomRate; }

void Camera::setPanRate(float panRate) { _panRate = panRate; }
void Camera::setRotateRate(float rotateRate) { _rotateRate = rotateRate; }
void Camera::setZoomRate(float zoomRate) { _zoomRate = zoomRate; }

void Camera::setLookAtPosition(const float (&v)[3]) {
  _lookAt[0] = v[0];
  _lookAt[1] = v[1];
  _lookAt[2] = v[2];
}

void Camera::setPhi(const float phi) { _phi = _saved_phi = phi; }
void Camera::setTheta(const float theta) { _theta = _saved_theta = theta; }
void Camera::setCameraDistance(const float d) { _d = _saved_d = d; }

void Camera::computeRightVector(float (&v)[3], const float theta, const float phi) {
  v[0] = -sin(phi);
  v[1] = cos(phi);
  v[2] = 0.0f;
}

void Camera::computeUpVector(float (&v)[3], const float theta, const float phi) {
  v[0] = -sin(theta) * cos(phi);
  v[1] = -sin(theta) * sin(phi);
  v[2] = cos(theta);
}

void Camera::computeViewVector(float (&v)[3], const float theta, const float phi) {
  v[0] = cos(theta) * cos(phi);
  v[1] = cos(theta) * sin(phi);
  v[2] = sin(theta);
}

void Camera::computeCameraPosition(float (&p)[3], const float (&lookAt)[3],
                                   const float theta, const float phi,
                                   const float d) {
  float v[3];
  computeViewVector(v, theta, phi);
  for (int i = 0; i < 3; i++)
    p[i] = lookAt[i] + d * v[i];
}

void Camera::computeCameraFrame(float (&x)[3], float (&y)[3], float (&z)[3],
                                const float theta, const float phi) {
  computeRightVector(x, theta, phi);
  computeUpVector(y, theta, phi);
  computeViewVector(z, theta, phi);
}

void Camera::computeCameraMatrix(float (&m)[16], float (&lookAt)[3],
                                 float theta, float phi, float d) {
  float x[3];
  float y[3];
  float z[3];
  computeCameraFrame(x, y, z, theta, phi);

  // first column
  m[0] = x[0];
  m[1] = y[0];
  m[2] = z[0];
  m[3] = 0.0f;

  // second column
  m[4] = x[1];
  m[5] = y[1];
  m[6] = z[1];
  m[7] = 0.0f;

  // third column
  m[8] = x[2];
  m[9] = y[2];
  m[10] = z[2];
  m[11] = 0.0f;

  // fourth column
  m[12] = -(lookAt[0] + z[0] * d);
  m[13] = -(lookAt[1] + z[1] * d);
  m[14] = -(lookAt[2] + z[2] * d);
  m[15] = 1.0f;
}
