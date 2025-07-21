#ifndef __SPLINES_H__
#define __SPLINES_H__
#include "Eigen/Dense"
#include "Eigen/Sparse"
#include <vector>

template<typename T>
class Spline {
public:
  Spline(const std::vector<T> &ts, const std::vector<T> &vs);
  virtual ~Spline();

  virtual T eval(T t) const = 0;

  const std::vector<T> &ts() const;
  const std::vector<T> &vs() const;

protected:
  static bool checkTs(std::vector<T> &ts);
  void checkAndInit();

  std::vector<T> _ts;
  std::vector<T> _vs;
};

template<typename T>
class ConstantSpline : public Spline<T> {
public:
  ConstantSpline(const std::vector<T> &ts, const std::vector<T> &vs);
  ~ConstantSpline();
  T eval(T t) const override;
};

template<typename T>
class LinearSpline : public Spline<T> {
public:
  LinearSpline(const std::vector<T> &ts, const std::vector<T> &vs);
  ~LinearSpline();
  T eval(T t) const override;
};

template<typename T>
class CubicSpline : public Spline<T> {
public:
  typedef Eigen::Triplet<float> Triplet;
  typedef Eigen::SparseMatrix<float, Eigen::RowMajor> SpMat;
  enum BoundaryBehavior { NATURAL,
                          PERIODIC };
  CubicSpline(const std::vector<float> &ts, const std::vector<float> &vs,
              BoundaryBehavior boundaryBehavior = NATURAL);
  ~CubicSpline();
  T eval(T t) const override;

private:
  void setupLinearSystem(SpMat &A, Eigen::VectorXf &b);
  void calculateCoefficients();

  std::vector<float> _coeffs_1;
  std::vector<float> _coeffs_2;
  BoundaryBehavior _boundary_behavior;
};

#endif // __SPLINES_H__
