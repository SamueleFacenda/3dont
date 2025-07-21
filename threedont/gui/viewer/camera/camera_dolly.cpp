#include "camera_dolly.h"

CameraPose::CameraPose() : _look_at(), _phi(0.0f), _theta(0.0f), _d(1.0f) {}

CameraPose::CameraPose(const QVector3D &p, float phi, float theta, float d)
    : _look_at(p), _phi(phi), _theta(theta), _d(d) {}

const QVector3D &CameraPose::lookAt() const { return _look_at; }
float CameraPose::phi() const { return _phi; }
float CameraPose::theta() const { return _theta; }
float CameraPose::d() const { return _d; }

void CameraPose::setLookAt(const QVector3D &p) { _look_at = p; }
void CameraPose::setPhi(float phi) { _phi = phi; }
void CameraPose::setTheta(float theta) { _theta = theta; }
void CameraPose::setD(float d) { _d = d; }

CameraPosesSOA::CameraPosesSOA(std::vector<CameraPose> &poses) {
  x.resize(poses.size());
  y.resize(poses.size());
  z.resize(poses.size());
  phi.resize(poses.size());
  theta.resize(poses.size());
  d.resize(poses.size());
  for (int i = 0; i < (int) poses.size(); i++) {
    x[i] = poses[i].lookAt().x();
    y[i] = poses[i].lookAt().y();
    z[i] = poses[i].lookAt().z();
    phi[i] = poses[i].phi();
    theta[i] = poses[i].theta();
    d[i] = poses[i].d();
  }
}

CameraDolly::CameraDolly() : _interp_type(LINEAR), _repeat(false), _active(false) {
  check_and_init();
  compute_splines();
}

CameraDolly::CameraDolly(const std::vector<float> &ts,
                         const std::vector<CameraPose> &poses,
                         InterpolationType interp, bool repeat)
    : _ts(ts),
      _poses(poses),
      _interp_type(interp),
      _repeat(repeat),
      _active(false) {
  check_and_init();
  compute_splines();
}

CameraDolly::~CameraDolly() {
  delete _look_at_x;
  delete _look_at_y;
  delete _look_at_z;
  delete _phi;
  delete _theta;
  delete _d;
}

void CameraDolly::start() {
  _active = true;
  _timer = vltools::getTime();
  _current_time = _start_time;
}

void CameraDolly::stop() { _active = false; }

void CameraDolly::getTimeAndPose(float &t, CameraPose &p) {
  step();
  t = _current_time;
  p = getPose(t);
}

float CameraDolly::getTime() {
  step();
  return _current_time;
}

CameraPose CameraDolly::getPose() {
  step();
  return getPose(_current_time);
}

CameraPose CameraDolly::getPose(float t) const {
  CameraPose pose;
  QVector3D look_at(_look_at_x->eval(t), _look_at_y->eval(t),
                    _look_at_z->eval(t));
  pose.setLookAt(look_at);
  pose.setPhi(_phi->eval(t));
  pose.setTheta(_theta->eval(t));
  pose.setD(_d->eval(t));
  return pose;
}

bool CameraDolly::done() const { return !_active; }

const std::vector<float> &CameraDolly::ts() const { return _ts; }
const std::vector<CameraPose> &CameraDolly::poses() const { return _poses; }
float CameraDolly::startTime() const { return _start_time; }
float CameraDolly::endTime() const { return _end_time; }

void CameraDolly::setStartTime(float t) {
  _start_time = qMin(qMax(t, _ts.front()), _end_time);
}

void CameraDolly::setEndTime(float t) {
  _end_time = qMin(qMax(t, _start_time), _ts.back());
}

void CameraDolly::setInterpType(InterpolationType type) {
  if (type != _interp_type) {
    delete _look_at_x;
    delete _look_at_y;
    delete _look_at_z;
    delete _phi;
    delete _theta;
    delete _d;
    _interp_type = type;
    compute_splines();
  }
}

void CameraDolly::setRepeat(bool b) { _repeat = b; }

void CameraDolly::check_and_init() {
  if (_ts.size() == _poses.size() && !_ts.empty()) {
    _start_time = _ts.front();
    _end_time = _ts.back();
    _current_time = _start_time;
  } else {
    _ts.clear();
    _ts.push_back(0.0);
    _poses.clear();
    _poses.push_back(CameraPose());
    _start_time = 0.0;
    _end_time = 0.0;
    _current_time = _start_time;
  }
}

void CameraDolly::step() {
  if (_active) {
    float elapsed = (float) (vltools::getTime() - _timer);
    if (_repeat)
      _current_time = fmod(elapsed, _end_time - _start_time) + _start_time;
    else {
      _current_time = elapsed + _start_time;
      if (_current_time >= _end_time) {
        _current_time = _end_time;
        _active = false;
      }
    }
  }
}

void CameraDolly::compute_splines() {
  if (_interp_type == LINEAR)
    interpolate_linear();
  else if (_interp_type == CUBIC_NATURAL)
    interpolate_cubic(CubicSpline<float>::NATURAL);
  else if (_interp_type == CUBIC_PERIODIC)
    interpolate_cubic(CubicSpline<float>::PERIODIC);
  else
    interpolate_const();
}

void CameraDolly::interpolate_const() {
  CameraPosesSOA posesSOA(_poses);
  _look_at_x = new ConstantSpline<float>(_ts, posesSOA.x);
  _look_at_y = new ConstantSpline<float>(_ts, posesSOA.y);
  _look_at_z = new ConstantSpline<float>(_ts, posesSOA.z);
  _phi = new ConstantSpline<float>(_ts, posesSOA.phi);
  _theta = new ConstantSpline<float>(_ts, posesSOA.theta);
  _d = new ConstantSpline<float>(_ts, posesSOA.d);
}

void CameraDolly::interpolate_linear() {
  CameraPosesSOA posesSOA(_poses);
  _look_at_x = new LinearSpline<float>(_ts, posesSOA.x);
  _look_at_y = new LinearSpline<float>(_ts, posesSOA.y);
  _look_at_z = new LinearSpline<float>(_ts, posesSOA.z);
  _phi = new LinearSpline<float>(_ts, posesSOA.phi);
  _theta = new LinearSpline<float>(_ts, posesSOA.theta);
  _d = new LinearSpline<float>(_ts, posesSOA.d);
}

void CameraDolly::interpolate_cubic(CubicSpline<float>::BoundaryBehavior b) {
  CameraPosesSOA posesSOA(_poses);
  _look_at_x = new CubicSpline<float>(_ts, posesSOA.x, b);
  _look_at_y = new CubicSpline<float>(_ts, posesSOA.y, b);
  _look_at_z = new CubicSpline<float>(_ts, posesSOA.z, b);
  _phi = new CubicSpline<float>(_ts, posesSOA.phi, b);
  _theta = new CubicSpline<float>(_ts, posesSOA.theta, b);
  _d = new CubicSpline<float>(_ts, posesSOA.d, b);
}
