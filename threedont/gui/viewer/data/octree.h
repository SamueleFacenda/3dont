#ifndef __OCTREE_H__
#define __OCTREE_H__

#include "../camera/camera.h"
#include "../utils/box3.h"
#include <vector>

class Octree {
public:
  struct Node {
    unsigned int centroid_index;
    unsigned int point_index;
    unsigned int point_count;
    bool is_leaf;
    Node *children[8];
  };

  struct CameraFrustum {
    float eye[3];
    float right[3];
    float up[3];
    float view[3];
    float image_r;
    float image_t;
    float z_near;

    void setImagePlane(float vfov, float aspect_ratio);
    void xyz2ruv(float (&ruv)[3], const float (&xyz)[3]) const;
    static void setupCameraFrustum(CameraFrustum &frustum, const Camera &camera,
                                   const float z_near, const float vfov,
                                   const float aspect_ratio);
    static void setupOrthoCamera(CameraFrustum &frustum, const Camera &camera,
                                 const float vfov, const float aspect_ratio);
  };

  Octree();
  ~Octree();

  const Node *getRoot() const;
  const std::vector<float> &getPointPos() const;
  const std::vector<unsigned int> &getIndices() const;
  const std::vector<unsigned int> &getIndicesR() const;

  unsigned int countNodes();
  unsigned int countInnerNodes();
  unsigned int getNumPoints() const;

  void buildTree(std::vector<float> &point_xyz, std::vector<float> &point_size,
                 unsigned int max_leaf_size = 64);

  enum ProjectionMode { PERSPECTIVE, ORTHOGRAPHIC };
  void getIndices(std::vector<unsigned int> &indices, const Camera &camera,
                  float vfov, float z_near, unsigned int width,
                  unsigned int height, float fudge_factor = 0.25f) const;

  void getIndicesOrtho(std::vector<unsigned int> &indices, const Camera &camera,
                       float image_r, float image_t, unsigned int height,
                       float fudge_factor = 0.25f) const;

  void getClickIndices(std::vector<unsigned int> &indices, const float screen_x,
                       const float screen_y, const float screen_radius,
                       const float screen_width, const float screen_height,
                       const float vfov,
                       const float near_clip, // positive, distance along -view
                       const Camera &camera,
                       const ProjectionMode projection_mode) const;

  void getClickIndicesBrute(
      std::vector<unsigned int> &indices, const float screen_x,
      const float screen_y, const float screen_radius, const float screen_width,
      const float screen_height, const float vfov,
      const float near_clip, // positive, distance along -view
      const Camera &camera, const ProjectionMode projection_mode) const;

private:
  static void traversalOrder(unsigned int (&nodeIndices)[8],
                             const float (&view)[3]);
  static float pointToPointDistanceSquared(const float (&x)[2],
                                           const float (&y)[2]);
  static float dotProduct(const float (&x)[3], const float (&y)[3]);
  static float boxToPointDistance(const float (&box_center)[2],
                                  const float (&box_size)[2],
                                  const float (&point)[2]);
  static void projectNode(float &v_min, float &v_max,
                          const float (&node_corner)[3], const float node_size,
                          const float (&view)[3], const float (&eye)[3]);
  static void getClickIndicesBruteHelper(
      int &min_idx, float &d_min, const float *points, const int count,
      const float (&click_pos)[2], const float click_radius,
      const CameraFrustum &frustum, const ProjectionMode &projection_mode);
  void getClickIndicesHelper(
      std::vector<unsigned int> &indices, float &d_min, const Node *node,
      const float (&node_corner)[3], const float node_size,
      const float (&click_pos)[2], const float click_radius,
      const CameraFrustum &frustum,
      const ProjectionMode projection_mode = PERSPECTIVE) const;

  Octree(const Octree &);
  Octree &operator=(Octree);

  void computeCentroid(float (&pos)[3], const unsigned int *indices,
                       const unsigned int count);
  void partitionXYZ(unsigned int *child_counts, unsigned int *indices,
                    unsigned char *labels, unsigned int count,
                    unsigned int bit = 2);
  unsigned int partition(unsigned int *indices, unsigned char *labels,
                         const unsigned int count, const unsigned int bit);
  unsigned int addCentroid(const float (&xyz)[3]);
  void computeCentroid(float (&centroid_xyz)[3], Node *children[]);
  bool pointsAreIdentical(const unsigned int *indices,
                          const unsigned int count);
  Node *buildTreeHelper(unsigned int *indices, unsigned char *labels,
                        const unsigned int count, const float (&cube_corner)[3],
                        const float cube_size);
  static void deleteTree(Node *root);
  static void
  boundProjectedAABB(float (&bound_center)[2], float (&bound_size)[2],
                     const float (&cube_center)[3], const float cube_size,
                     float z_near);
  static void
  boundOrthoProjectedAABB(float (&bound_center)[2], float (&bound_size)[2],
                          const float (&cube_center)[3],
                          const float cube_size);
  enum IntersectResult { OUTSIDE = 0, UNCERTAIN = 1, INSIDE = 2 };
  static IntersectResult intersectBoxes2D(const float (&box_1_center)[2],
                                          const float (&box_1_size)[2],
                                          const float (&box_2_center)[2],
                                          const float (&box_2_size)[2]);
  bool cubeInFrustum(const float (&cube_corner)[3], const float cube_size,
                     const CameraFrustum &frustum);
  void getIndicesHelper(
      std::vector<unsigned int> &indices, const Node *node,
      const CameraFrustum &frustum, const float (&cube_corner)[3],
      const float cube_size, const float eps,
      const ProjectionMode projection_mode = PERSPECTIVE,
      const float fudge_factor = 0.25f,
      const IntersectResult parent_intersect_result = UNCERTAIN) const;
  unsigned int countNodesHelper(const Node *node, bool count_leaf = true);

  unsigned int _max_leaf_size;
  unsigned int _num_points;
  Node *_root;
  std::vector<float> *_ptr_point_xyz;
  std::vector<float> *_ptr_point_size;

  float _lower_left_corner[3];
  float _cube_size;

  std::vector<unsigned int> _indices;
  std::vector<unsigned int> _indices_r;
  std::vector<unsigned char> _labels;
};

#endif // __OCTREE_H__
