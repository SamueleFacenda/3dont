#ifndef __MI_OPENGL_TEXT_H__
#define __MI_OPENGL_TEXT_H__

// #include <QOpenGLWidget>
#include "opengl_funcs.h"
#include <QOpenGLContext>
#include <QRectF>
#include <QWindow>
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
  Text(QOpenGLWidget* parent, const QFont &f)
      : _parent(parent),
        font(f),
        fontMetrics(f),
        pixelFont(f),
        pixelFontMetrics(f),
        xOffset(1),
        yOffset(1) {
    initializeOpenGLFunctions();

    // font sizes in units of pixels
    // (I don't really know how this works... this is a hack)
    if (_parent->devicePixelRatio() != 1.0)
      pixelFont.setPixelSize(
              qRound(_parent->devicePixelRatio() * font.pointSize()));
    pixelFontMetrics = QFontMetrics(pixelFont);
  }

  virtual ~Text() { clearCache(); }

  void clearCache() {
    foreach (GLuint texture, textures)
      glDeleteTextures(1, &texture);
    textures.clear();
    characters.clear();
  }

  const QFont &getFont() const { return font; }

  const QFontMetrics &getFontMetrics() const { return fontMetrics; }

  QSizeF computeTextSize(const QString &text) {
    QSizeF sz;
    for (int i = 0; i < text.length(); ++i) {
      CharData &c = createCharacter(text[i]);
      sz.setHeight(qMax(sz.height(), (qreal) c.height));
      sz.setWidth(sz.width() + c.width);
    }
    return sz;
  }

  QRectF renderText(float x, float y, const QString &text,
                    const QVector4D &color = QVector4D(1, 1, 1, 1)) {
    x = 2.0f * x / _parent->width() - 1.0f;
    y = 2.0f * y / _parent->height() - 1.0f;

    glPushAttrib(GL_CURRENT_BIT | GL_ENABLE_BIT | GL_TEXTURE_BIT);
    glPushMatrix();
    glEnable(GL_TEXTURE_2D); // TODO
    glEnable(GL_BLEND);
    glDisable(GL_DEPTH_TEST);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glUseProgram(0);

    GLuint texture = 0;
    glLoadIdentity();
    glTranslatef(x, y, 0);
    glColor4f(color.x(), color.y(), color.z(), color.w());
    QRectF rect(QPointF(x, y), QPointF(x, y));
    for (int i = 0; i < text.length(); ++i) {
      CharData &c = createCharacter(text[i]);

      if (texture != c.textureId) {
        texture = c.textureId;
        glBindTexture(GL_TEXTURE_2D, texture);
      }

      float w = c.width * 2.0f / _parent->width();
      float h = c.height * 2.0f / _parent->height();

      rect.setHeight(qMax(rect.height(), (qreal) c.height));
      rect.setWidth(rect.width() + c.width);

      glBegin(GL_QUADS);
      glTexCoord2f(c.s[0], c.t[0]);
      glVertex2f(0, 0);

      glTexCoord2f(c.s[1], c.t[0]);
      glVertex2f(w, 0);

      glTexCoord2f(c.s[1], c.t[1]);
      glVertex2f(w, h);

      glTexCoord2f(c.s[0], c.t[1]);
      glVertex2f(0, h);
      glEnd();

      glTranslatef(w, 0, 0);
    }

    glPopMatrix();
    glPopAttrib();

    return rect;
  }

private:
  void allocateTexture() {
    GLuint texture;
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);

    glTexImage2D(GL_TEXTURE_2D, 0, GL_ALPHA, TEXTURE_SIZE, TEXTURE_SIZE, 0,
                 GL_RGBA, GL_UNSIGNED_BYTE, 0);

    textures += texture;
  }

  CharData &createCharacter(QChar c) {
    ushort unicodeC = c.unicode();
    if (characters.contains(unicodeC)) return characters[unicodeC];

    if (textures.empty()) allocateTexture();

    GLuint texture = textures.last();

    GLsizei width = pixelFontMetrics.horizontalAdvance(c);
    GLsizei height = pixelFontMetrics.height();

    QPixmap pixmap(width, height);
    pixmap.fill(Qt::transparent);

    QPainter painter;
    painter.begin(&pixmap);
    painter.setRenderHints(QPainter::Antialiasing |
                           QPainter::TextAntialiasing);
    painter.setFont(pixelFont);
    painter.setPen(Qt::white);

    painter.drawText(0, pixelFontMetrics.ascent(), c);
    painter.end();
    QImage image = pixmap.toImage().mirrored();

    if (xOffset + width >= TEXTURE_SIZE) {
      xOffset = 1;
      yOffset += height;
    }
    if (yOffset + height >= TEXTURE_SIZE) {
      allocateTexture();
      yOffset = 1;
    }

    glBindTexture(GL_TEXTURE_2D, texture);
    glTexSubImage2D(GL_TEXTURE_2D, 0, xOffset, yOffset, width, height, GL_RGBA,
                    GL_UNSIGNED_BYTE, image.bits());

    CharData &character = characters[unicodeC];
    character.textureId = texture;
    character.width = fontMetrics.horizontalAdvance(c);
    character.height = fontMetrics.height();
    character.s[0] = static_cast<GLfloat>(xOffset) / TEXTURE_SIZE;
    character.t[0] = static_cast<GLfloat>(yOffset) / TEXTURE_SIZE;
    character.s[1] = static_cast<GLfloat>(xOffset + width) / TEXTURE_SIZE;
    character.t[1] = static_cast<GLfloat>(yOffset + height) / TEXTURE_SIZE;

    xOffset += width;
    return character;
  }
  
  QOpenGLWidget *_parent;

  QFont font;
  QFontMetrics fontMetrics;

  QFont pixelFont;
  QFontMetrics pixelFontMetrics;

  QHash<ushort, CharData> characters;
  QList<GLuint> textures;

  GLint xOffset;
  GLint yOffset;
};

#endif // __MI_OPENGL_TEXT_H__
