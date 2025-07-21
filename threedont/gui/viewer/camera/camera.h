#ifndef __CAMERA_H__
#define __CAMERA_H__

#define PI 3.14159265359f

class Camera {
public:
  Camera();
  Camera(float xmin, float xmax, float ymin, float ymax, float zmin, float zmax);

  /*! \brief rotate
   *  Camera is rotated based on mouse movement (dx,dy).
   *  _phi = _phi - _rotateRate * dx
   *  _theta = _theta + _rotateRate * dy
   */
  void rotate(float dx, float dy);

  /*! \brief pan
   *  Camera is panned based on mouse movement (dx,dy).
   *  _lookAt = _lookAt
   *            - _right * _panRate * dx
   *            + _up * _panRate * dy
   */
  void pan(float dx, float dy);

  /*! \brief zoom
   *  Camera is zoomed based on scroll by dx amount.
   *  _d = _d * _zoomRate ^ dx
   */
  void zoom(float dx);

  void save();
  void restore();

  /*! \brief getCameraPosition */
  void getCameraPosition(float (&p)[3]) const;

  /*! \brief getLookAtPosition */
  void getLookAtPosition(float (&p)[3]) const;

  /*! \brief getRightVector */
  void getRightVector(float (&v)[3]) const;

  /*! \brief getUpVector */
  void getUpVector(float (&v)[3]) const;

  /*! \brief getViewVector */
  void getViewVector(float (&v)[3]) const;

  float getCameraDistance() const;
  float getTheta() const;
  float getPhi() const;
  float getPanRate() const;
  float getRotateRate() const;
  float getZoomRate() const;
  void setPanRate(float panRate);
  void setRotateRate(float rotateRate);
  void setZoomRate(float zoomRate);
  void setLookAtPosition(const float (&v)[3]);
  void setPhi(const float phi);
  void setTheta(const float theta);
  void setCameraDistance(const float d);

  static void computeRightVector(float (&v)[3], const float theta,
                                 const float phi);
  static void computeUpVector(float (&v)[3], const float theta,
                              const float phi);
  static void computeViewVector(float (&v)[3], const float theta,
                                const float phi);

private:
  static void computeCameraPosition(float (&p)[3], const float (&lookAt)[3],
                                    const float theta, const float phi,
                                    const float d);
  static void computeCameraFrame(float (&x)[3], float (&y)[3], float (&z)[3],
                                 const float theta, const float phi);
  static void computeCameraMatrix(float (&m)[16], float (&lookAt)[3],
                                  float theta, float phi, float d);

  float _lookAt[3];
  float _theta; // angle of elevation
  float _phi;   // azimuthal angle
  float _d;     // camera distance from _lookAt

  float _saved_lookAt[3];
  float _saved_theta;
  float _saved_phi;
  float _saved_d;

  float _panRate;
  float _zoomRate;
  float _rotateRate;
};

#endif // __CAMERA_H__
