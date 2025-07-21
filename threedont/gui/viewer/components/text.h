#ifndef __MI_OPENGL_TEXT_H__
#define __MI_OPENGL_TEXT_H__

#include "../utils/opengl_funcs.h"
#include <QOpenGLBuffer>
#include <QOpenGLContext>
#include <QOpenGLShaderProgram>
#include <QOpenGLVertexArrayObject>
#include <QOpenGLWidget>
#include <QRectF>
#include <QtCore/QHash>
#include <QtCore/QSysInfo>
#include <QtGlobal>
#include <QtGui/QPainter>
#include <QtGui/QPixmap>
#include <cmath>
#include <iostream>

/* following text rendering code adapted from libs/opengl/Text.h and
   libs/opengl/Text.cpp of mifit project: https://code.google.com/p/mifit/ */

class QChar;
class QFont;
class QFontMetrics;
class QString;

const int TEXTURE_SIZE = 256;

class Text : public OpenGLFuncs {
  struct CharData {
    GLuint textureId;
    uint width;
    uint height;
    GLfloat s[2];
    GLfloat t[2];
  };

public:
  Text(QOpenGLWidget *parent, const QFont &f);
  ~Text() override;

  void clearCache();
  const QFont &getFont() const;
  const QFontMetrics &getFontMetrics() const;
  QSizeF computeTextSize(const QString &text);
  QRectF renderText(float x, float y, const QString &text, const QVector4D &color = QVector4D(1, 1, 1, 1));

private:
  void initializeShaders();
  void initializeBuffers();
  void cleanup();
  void allocateTexture();
  CharData &createCharacter(QChar c);

  QOpenGLWidget *_parent;
  QFont font;
  QFontMetrics fontMetrics;
  QFont pixelFont;
  QFontMetrics pixelFontMetrics;
  QHash<ushort, CharData> characters;
  QList<GLuint> textures;
  GLint xOffset;
  GLint yOffset;
  QOpenGLShaderProgram *_shaderProgram;
  QOpenGLVertexArrayObject *_vao;
  QOpenGLBuffer *_vbo;
  QOpenGLBuffer *_ebo;
};

#endif // __MI_OPENGL_TEXT_H__

