#include "splines.h"
#include <algorithm>
#include <cmath>

template<typename T>
Spline<T>::Spline(const std::vector<T> &ts, const std::vector<T> &vs) : _ts(ts), _vs(vs) {
  checkAndInit();
}

template<typename T>
Spline<T>::~Spline() {}

template<typename T>
bool Spline<T>::checkTs(std::vector<T> &ts) {
  for (std::size_t i = 1; i < ts.size(); i++)
    if (ts[i] <= ts[i - 1])
      return false;
  return !ts.empty();
}

template<typename T>
void Spline<T>::checkAndInit() {
  if (!checkTs(_ts) || _ts.size() != _vs.size()) {
    _ts.clear();
    _ts.push_back(0.0f);
    _vs.clear();
    _vs.push_back(0.0f);
  }
}

template<typename T>
ConstantSpline<T>::ConstantSpline(const std::vector<T> &ts, const std::vector<T> &vs)
    : Spline<T>(ts, vs) {}

template<typename T>
ConstantSpline<T>::~ConstantSpline() {}

template<typename T>
T ConstantSpline<T>::eval(T t) const {
  auto it = std::upper_bound(this->_ts.begin(), this->_ts.end(), t);
  int idx = it - this->_ts.begin();
  if (idx == 0)
    return this->_vs.front();
  else if (idx == (int) this->_ts.size())
    return this->_vs.back();
  else
    return this->_vs[idx - 1];
}

template<typename T>
LinearSpline<T>::LinearSpline(const std::vector<T> &ts, const std::vector<T> &vs)
    : Spline<T>(ts, vs) {}

template<typename T>
LinearSpline<T>::~LinearSpline() {}

template<typename T>
T LinearSpline<T>::eval(T t) const {
  auto it = std::upper_bound(this->_ts.begin(), this->_ts.end(), t);
  int idx = it - this->_ts.begin();
  if (idx == 0)
    return this->_vs.front();
  else if (idx == (int) this->_ts.size())
    return this->_vs.back();
  else {
    float dt = (t - this->_ts[idx - 1]) / (this->_ts[idx] - this->_ts[idx - 1]);
    return (1.0f - dt) * this->_vs[idx - 1] + dt * this->_vs[idx];
  }
}

template<typename T>
CubicSpline<T>::CubicSpline(const std::vector<float> &ts, const std::vector<float> &vs,
                            BoundaryBehavior boundaryBehavior)
    : Spline<T>(ts, vs), _boundary_behavior(boundaryBehavior) {
  calculateCoefficients();
}

template<typename T>
CubicSpline<T>::~CubicSpline() {}

template<typename T>
T CubicSpline<T>::eval(T t) const {
  auto it = std::upper_bound(this->_ts.begin(), this->_ts.end(), t);
  int idx = it - this->_ts.begin();
  if (idx == 0)
    return this->_vs.front();
  else if (idx == (int) this->_ts.size())
    return this->_vs.back();
  else {
    float dt = (t - this->_ts[idx - 1]) / (this->_ts[idx] - this->_ts[idx - 1]);
    float c[4] = {this->_vs[idx - 1], _coeffs_1[idx - 1], _coeffs_2[idx - 1], this->_vs[idx]};
    float a[4] = {1.0f, 3.0f, 3.0f, 1.0f};
    float v = 0.0f;
    for (int i = 0; i < 4; i++)
      v += c[i] * a[i] * powf(1.0f - dt, 3.0f - (float) i) * powf(dt, (float) i);
    return v;
  }
}

template<typename T>
void CubicSpline<T>::setupLinearSystem(SpMat &A, Eigen::VectorXf &b) {
  int num_intervals = (int) this->_ts.size() - 1;
  int num_knots = num_intervals - 1;
  int n = 2 * num_intervals;

  std::vector<float> delta_inv(num_intervals);
  std::vector<float> delta_inv_2(num_intervals);
  for (int i = 0; i < num_intervals; i++) {
    float temp = this->_ts[i + 1] - this->_ts[i];
    delta_inv[i] = 1.0f / temp;
    delta_inv_2[i] = delta_inv[i] * delta_inv[i];
  }

  b.resize(n);
  std::vector<Triplet> triplets;
  for (int i = 0; i < num_knots; i++) {
    triplets.push_back(Triplet(2 * i, 2 * i + 1, -3.0f * delta_inv[i]));
    triplets.push_back(Triplet(2 * i, 2 * (i + 1), -3.0f * delta_inv[i + 1]));
    b(2 * i) = -3.0f * this->_vs[i + 1] * delta_inv[i + 1] +
               -3.0f * this->_vs[i + 1] * delta_inv[i];

    triplets.push_back(Triplet(2 * i + 1, 2 * i, 6.0f * delta_inv_2[i]));
    triplets.push_back(Triplet(2 * i + 1, 2 * i + 1, -12.0f * delta_inv_2[i]));
    triplets.push_back(Triplet(2 * i + 1, 2 * (i + 1), 12.0f * delta_inv_2[i + 1]));
    triplets.push_back(Triplet(2 * i + 1, 2 * (i + 1) + 1, -6.0f * delta_inv_2[i + 1]));
    b(2 * i + 1) = 6.0f * this->_vs[i + 1] * delta_inv_2[i + 1] -
                   6.0f * this->_vs[i + 1] * delta_inv_2[i];
  }
  if (_boundary_behavior == PERIODIC) {
    triplets.push_back(Triplet(n - 2, 0, 3.0f * delta_inv.front()));
    triplets.push_back(Triplet(n - 2, n - 1, 3.0f * delta_inv.back()));
    b(n - 2) = 3.0f * this->_vs.back() * delta_inv.back() +
               3.0f * this->_vs.front() * delta_inv.front();

    triplets.push_back(Triplet(n - 1, 0, -12.0f * delta_inv_2.front()));
    triplets.push_back(Triplet(n - 1, 1, 6.0f * delta_inv_2.front()));
    triplets.push_back(Triplet(n - 1, n - 2, -6.0f * delta_inv_2.back()));
    triplets.push_back(Triplet(n - 1, n - 1, 12.0f * delta_inv_2.back()));
    b(n - 1) = 6.0f * this->_vs.back() * delta_inv_2.back() -
               6.0f * this->_vs.front() * delta_inv_2.front();
  } else if (_boundary_behavior == NATURAL) {
    triplets.push_back(Triplet(n - 2, 0, -12.0f * delta_inv_2.front()));
    triplets.push_back(Triplet(n - 2, 1, 6.0f * delta_inv_2.front()));
    b(n - 2) = -6.0f * this->_vs.front() * delta_inv_2.front();

    triplets.push_back(Triplet(n - 1, n - 2, 6.0f * delta_inv_2.back()));
    triplets.push_back(Triplet(n - 1, n - 1, -12.0f * delta_inv_2.back()));
    b(n - 1) = -6.0f * this->_vs.back() * delta_inv_2.back();
  }
  A.resize(n, n);
  A.setFromTriplets(triplets.begin(), triplets.end());
}

template<typename T>
void CubicSpline<T>::calculateCoefficients() {
  int num_intervals = (int) this->_ts.size() - 1;
  if (num_intervals > 0) {
    SpMat A;
    Eigen::VectorXf b, x;
    setupLinearSystem(A, b);
    x = Eigen::MatrixXf(A).colPivHouseholderQr().solve(b);
    _coeffs_1.resize(num_intervals);
    _coeffs_2.resize(num_intervals);
    for (int i = 0; i < num_intervals; i++) {
      _coeffs_1[i] = x(2 * i);
      _coeffs_2[i] = x(2 * i + 1);
    }
  }
}

template class Spline<float>;
template class ConstantSpline<float>;
template class LinearSpline<float>;
template class CubicSpline<float>;
