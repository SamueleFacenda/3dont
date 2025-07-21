#ifndef __POINTATTRIBUTES_H__
#define __POINTATTRIBUTES_H__
#include "octree.h"
#include <QVector3D>
#include <QVector4D>
#include <QtGlobal>
#include <algorithm>
#include <vector>

class PointAttributes {
private:
  std::vector<std::vector<float>> _attr;
  std::vector<quint64> _attr_size;
  std::vector<quint64> _attr_dim;
  std::size_t _curr_idx;

public:
  PointAttributes();

  bool set(const std::vector<float> &attr, quint64 attr_size, quint64 attr_dim);
  bool set(const std::vector<char> &data, const Octree &octree);
  void reset();

  const std::vector<float> &operator[](int i) const;
  float operator()(int i, int j) const;
  float operator()(int k, int i, int j) const;

  std::size_t currentIndex() const;
  quint64 size(int i) const;
  quint64 dim(int i) const;
  std::size_t numAttributes() const;
  void setCurrentIndex(std::size_t i);

private:
  bool _unpack(const std::vector<char> &data, unsigned int expected_size);
  void _reorder(std::size_t attr_idx, const Octree &octree);
  void _compute_LOD(std::size_t attr_idx, const Octree &octree);
  void _compute_rgba_LOD_helper(std::size_t attr_idx, const Octree::Node *node);
  void _compute_LOD_helper(std::size_t attr_idx, const Octree::Node *node);
  inline void _accumulate_rgba(QVector3D &x, float &w, const float *v);
  inline void _xw_to_rgba(float *dst, const QVector3D &x, const float &w, unsigned int n) const;

  template<typename T>
  bool _unpack_number(T &v, const char *&ptr, const char *const ptr_end) {
    // returns false if attempting to read beyond end of stream [ptr, ptr_end)
    if (ptr + sizeof(T) > ptr_end) {
      return false;
    } else {
      v = *(T *) ptr;
      ptr += sizeof(T);
      return true;
    }
  }

  template<typename T>
  bool _unpack_array(std::vector<T> &v, const char *&ptr, const char *const ptr_end) {
    if (ptr + sizeof(T) * v.size() > ptr_end) {
      return false;
    } else {
      std::copy((const T *) ptr, (const T *) (ptr + sizeof(T) * v.size()),
                v.begin());
      ptr += sizeof(T) * v.size();
      return true;
    }
  }
};

#endif // __POINTATTRIBUTES_H__
