#ifndef __CAMERADOLLY_H__
#define __CAMERADOLLY_H__
#include <QVector3D>
#include <QtGlobal>
#include <vector>
#include "../utils/timer.h"
#include "splines.h"

class CameraPose {
public:
  CameraPose();
  CameraPose(const QVector3D &p, float phi, float theta, float d);

  // getter functions
  const QVector3D &lookAt() const;
  float phi() const;
  float theta() const;
  float d() const;

  // setter functions
  void setLookAt(const QVector3D &p);
  void setPhi(float phi);
  void setTheta(float theta);
  void setD(float d);

private:
  QVector3D _look_at;
  float _phi;
  float _theta;
  float _d;
};

struct CameraPosesSOA {
  CameraPosesSOA(std::vector<CameraPose> &poses);
  std::vector<float> x;
  std::vector<float> y;
  std::vector<float> z;
  std::vector<float> phi;
  std::vector<float> theta;
  std::vector<float> d;
};

class CameraDolly {
public:
  enum InterpolationType { CONSTANT,
                           LINEAR,
                           CUBIC_NATURAL,
                           CUBIC_PERIODIC };

  CameraDolly();
  CameraDolly(const std::vector<float> &ts,
              const std::vector<CameraPose> &poses,
              InterpolationType interp = LINEAR, bool repeat = false);
  ~CameraDolly();

  // actions
  void start();
  void stop();

  // dolly states
  void getTimeAndPose(float &t, CameraPose &p);
  float getTime();
  CameraPose getPose();
  CameraPose getPose(float t) const;
  bool done() const;

  // getters
  const std::vector<float> &ts() const;
  const std::vector<CameraPose> &poses() const;
  float startTime() const;
  float endTime() const;

  // setters
  void setStartTime(float t);
  void setEndTime(float t);
  void setInterpType(InterpolationType type);
  void setRepeat(bool b);

private:
  void check_and_init();
  void step();
  void compute_splines();
  void interpolate_const();
  void interpolate_linear();
  void interpolate_cubic(CubicSpline<float>::BoundaryBehavior b);

  std::vector<float> _ts;
  std::vector<CameraPose> _poses;

  Spline<float> *_look_at_x;
  Spline<float> *_look_at_y;
  Spline<float> *_look_at_z;
  Spline<float> *_phi;
  Spline<float> *_theta;
  Spline<float> *_d;

  float _start_time;
  float _end_time;
  float _current_time;
  double _timer;

  InterpolationType _interp_type;
  bool _repeat;
  bool _active;
};

#endif // __CAMERADOLLY_H__
