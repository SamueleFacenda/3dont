#ifndef __VIEWER_H__
#define __VIEWER_H__

#include <QColor>
#include <QCoreApplication>
#include <QImage>
#include <QMatrix4x4>
#include <QMouseEvent>
#include <QOpenGLContext>
#include <QOpenGLDebugLogger>
#include <QOpenGLShaderProgram>
#include <QOpenGLWidget>
#include <QString>
#include <QTcpServer>
#include <QTcpSocket>
#include <QTimer>
#include <QVector3D>
#include <QWheelEvent>
#include <QtCore/qmath.h>
#include <fstream>
#include <iostream>
#include <limits>

#include "camera/camera_dolly.h"
#include "camera/qt_camera.h"
#include "components/background.h"
#include "components/floor_grid.h"
#include "components/look_at.h"
#include "components/point_cloud.h"
#include "components/selection_box.h"
#include "components/text.h"
#include "utils/alternative_frame_buffer.h"
#include "utils/comm_funcs.h"
#include "utils/opengl_funcs.h"
#include "utils/timer.h"

#define OPENGL_DEBUG

class Viewer : public QOpenGLWidget, protected OpenGLFuncs {
  Q_OBJECT
public:
  Viewer(QWidget *parent = nullptr);
  ~Viewer();

  void initializeGL() override;
  void resizeGL(int w, int h) override;
  void paintGL() override;
  int getServerPort();

signals:
  void singlePointSelected(unsigned int);

protected:
  void keyPressEvent(QKeyEvent *ev) override;
  void mouseDoubleClickEvent(QMouseEvent *ev) override;
  void mousePressEvent(QMouseEvent *ev) override;
  void mouseMoveEvent(QMouseEvent *ev) override;
  void mouseReleaseEvent(QMouseEvent *ev) override;
  void wheelEvent(QWheelEvent *ev) override;

private slots:
  void reply();
  void drawRefinedPointsDelayed();
  void renderPointsFine();
  void playCameraAnimation();

private:
  void updateFast();
  void scheduleFineRendering(int msec = 0);
  void updateSlow();
  void printScreen(std::string filename);
  void displayInfo();
  void renderPoints();
  QPointF win2ndc(QPointF p);

  QTcpServer *_server;
  QPointF _pressPos;

  QtCamera _camera;
  FloorGrid *_floor_grid;
  SelectionBox *_selection_box;
  PointCloud *_points;
  Background *_background;
  LookAt *_look_at;
  Text *_text;
  CameraDolly *_dolly;
  AlternativeFrameBuffer *_fine_render_fbo;

  enum FineRenderState { INACTIVE,
                         INITIALIZE,
                         CHUNK,
                         FINALIZE,
                         TERMINATE };
  FineRenderState _fine_render_state;
  QTimer *_timer_fine_render_delay;
  std::size_t _chunk_offset;
  std::size_t _max_chunk_size;
  std::vector<unsigned int> _refined_indices;

  QTcpSocket *_socket_waiting_on_enter_key;
  double _render_time;
  bool _show_text;
  bool _fine_rendering_available;
};

#endif // __VIEWER_H__
