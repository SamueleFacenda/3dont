#ifndef __MI_OPENGL_TEXT_H__
#define __MI_OPENGL_TEXT_H__

#include <QOpenGLWidget>
#include <QOpenGLContext>
#include <QRectF>
#include <QtCore/QHash>
#include <QtCore/QSysInfo>
#include <QtGlobal>
#include <QtGui/QPainter>
#include <QtGui/QPixmap>
#include <QOpenGLBuffer>
#include <QOpenGLVertexArrayObject>
#include <QOpenGLShaderProgram>
#include <cmath>
#include <iostream>
#include "opengl_funcs.h"

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
  Text(QOpenGLWidget* parent, const QFont& f)
      : _parent(parent),
        font(f),
        fontMetrics(f),
        pixelFont(f),
        pixelFontMetrics(f),
        xOffset(1),
        yOffset(1),
        _shaderProgram(nullptr),
        _vao(nullptr),
        _vbo(nullptr) {
    initializeOpenGLFunctions();

    // Initialize modern OpenGL resources
    initializeShaders();
    initializeBuffers();

    // font sizes in units of pixels
    // (I don't really know how this works... this is a hack)
    if (_parent->devicePixelRatio() != 1.0)
      pixelFont.setPixelSize(
              qRound(_parent->devicePixelRatio() * font.pointSize()));
    pixelFontMetrics = QFontMetrics(pixelFont);
  }

  virtual ~Text() {
    clearCache();
    cleanup();
  }

  void clearCache() {
    foreach (GLuint texture, textures)
      glDeleteTextures(1, &texture);
    textures.clear();
    characters.clear();
  }

  const QFont& getFont() const { return font; }

  const QFontMetrics& getFontMetrics() const { return fontMetrics; }

  QSizeF computeTextSize(const QString& text) {
    QSizeF sz;
    for (int i = 0; i < text.length(); ++i) {
      CharData& c = createCharacter(text[i]);
      sz.setHeight(qMax(sz.height(), (qreal)c.height));
      sz.setWidth(sz.width() + c.width);
    }
    return sz;
  }

  QRectF renderText(float x, float y, const QString& text,
                    const QVector4D& color = QVector4D(1, 1, 1, 1)) {
    if (!_shaderProgram || !_vao) return QRectF();

    // Convert screen coordinates to normalized device coordinates
    float ndcX = 2.0f * x / _parent->width() - 1.0f;
    float ndcY = 2.0f * y / _parent->height() - 1.0f;

    // Save current OpenGL state
    GLboolean depthTestEnabled;
    GLboolean blendEnabled;
    GLint currentProgram;
    GLint currentVAO;
    GLint currentTexture;

    glGetBooleanv(GL_DEPTH_TEST, &depthTestEnabled);
    glGetBooleanv(GL_BLEND, &blendEnabled);
    glGetIntegerv(GL_CURRENT_PROGRAM, &currentProgram);
    glGetIntegerv(GL_VERTEX_ARRAY_BINDING, &currentVAO);
    glGetIntegerv(GL_TEXTURE_BINDING_2D, &currentTexture);

    // Set up rendering state
    glEnable(GL_BLEND);
    glDisable(GL_DEPTH_TEST);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    _shaderProgram->bind();
    _vao->bind();

    // Set uniforms
    _shaderProgram->setUniformValue("u_color", color);
    _shaderProgram->setUniformValue("u_texture", 0);

    float currentX = ndcX;
    GLuint currentTextureId = 0;
    QRectF rect(QPointF(x, y), QPointF(x, y));

    for (int i = 0; i < text.length(); ++i) {
      CharData& c = createCharacter(text[i]);

      if (currentTextureId != c.textureId) {
        currentTextureId = c.textureId;
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, currentTextureId);
      }

      float w = c.width * 2.0f / _parent->width();
      float h = c.height * 2.0f / _parent->height();

      rect.setHeight(qMax(rect.height(), (qreal)c.height));
      rect.setWidth(rect.width() + c.width);

      // Update vertex data for this character
      float vertices[] = {
              // Position (x, y)    // TexCoord (s, t)
              currentX,     ndcY,     c.s[0], c.t[0],  // Bottom-left
              currentX + w, ndcY,     c.s[1], c.t[0],  // Bottom-right
              currentX + w, ndcY + h, c.s[1], c.t[1],  // Top-right
              currentX,     ndcY + h, c.s[0], c.t[1]   // Top-left
      };

      _vbo->bind();
      _vbo->write(0, vertices, sizeof(vertices));
      _vbo->release();

      // Draw the character quad
      glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

      currentX += w;
    }

    _vao->release();
    _shaderProgram->release();

    // Restore OpenGL state
    if (!depthTestEnabled) glDisable(GL_DEPTH_TEST);
    else glEnable(GL_DEPTH_TEST);

    if (!blendEnabled) glDisable(GL_BLEND);
    else glEnable(GL_BLEND);

    glUseProgram(currentProgram);
    glBindVertexArray(currentVAO);
    glBindTexture(GL_TEXTURE_2D, currentTexture);

    return rect;
  }

private:
  void initializeShaders() {
    _shaderProgram = new QOpenGLShaderProgram();

    // Vertex shader
    QString vertexShaderSource = R"(
      #version 330 core
      layout (location = 0) in vec2 a_position;
      layout (location = 1) in vec2 a_texCoord;

      out vec2 v_texCoord;

      void main() {
        gl_Position = vec4(a_position, 0.0, 1.0);
        v_texCoord = a_texCoord;
      }
    )";

    // Fragment shader
    QString fragmentShaderSource = R"(
      #version 330 core
      in vec2 v_texCoord;

      uniform sampler2D u_texture;
      uniform vec4 u_color;

      out vec4 fragColor;

      void main() {
        float alpha = texture(u_texture, v_texCoord).a;
        fragColor = vec4(u_color.rgb, u_color.a * alpha);
      }
    )";

    _shaderProgram->addShaderFromSourceCode(QOpenGLShader::Vertex, vertexShaderSource);
    _shaderProgram->addShaderFromSourceCode(QOpenGLShader::Fragment, fragmentShaderSource);

    if (!_shaderProgram->link()) {
      std::cerr << "Failed to link shader program: "
                << _shaderProgram->log().toStdString() << std::endl;
    }
  }

  void initializeBuffers() {
    _vao = new QOpenGLVertexArrayObject();
    _vao->create();
    _vao->bind();

    _vbo = new QOpenGLBuffer(QOpenGLBuffer::VertexBuffer);
    _vbo->create();
    _vbo->bind();

    // Allocate space for 4 vertices * 4 floats per vertex
    _vbo->allocate(4 * 4 * sizeof(float));

    // Set up vertex attributes
    glEnableVertexAttribArray(0); // position
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);

    glEnableVertexAttribArray(1); // texture coordinates
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));

    // Create and bind element buffer for indices
    _ebo = new QOpenGLBuffer(QOpenGLBuffer::IndexBuffer);
    _ebo->create();
    _ebo->bind();

    // Quad indices (two triangles)
    unsigned int indices[] = {
            0, 1, 2,  // First triangle
            2, 3, 0   // Second triangle
    };

    _ebo->allocate(indices, sizeof(indices));

    _vao->release();
  }

  void cleanup() {
    delete _shaderProgram;
    _shaderProgram = nullptr;

    delete _vao;
    _vao = nullptr;

    delete _vbo;
    _vbo = nullptr;

    delete _ebo;
    _ebo = nullptr;
  }

  void allocateTexture() {
    GLuint texture;
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    // Use RGBA like the original code
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, TEXTURE_SIZE, TEXTURE_SIZE, 0,
                 GL_RGBA, GL_UNSIGNED_BYTE, 0);

    textures += texture;
  }

  CharData& createCharacter(QChar c) {
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
    QImage image = pixmap.toImage().flipped();

    if (xOffset + width >= TEXTURE_SIZE) {
      xOffset = 1;
      yOffset += height;
    }
    if (yOffset + height >= TEXTURE_SIZE) {
      allocateTexture();
      texture = textures.last();
      yOffset = 1;
    }

    glBindTexture(GL_TEXTURE_2D, texture);

    // Use RGBA format like the original code
    glTexSubImage2D(GL_TEXTURE_2D, 0, xOffset, yOffset, width, height, GL_RGBA,
                    GL_UNSIGNED_BYTE, image.bits());

    CharData& character = characters[unicodeC];
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

  QOpenGLWidget* _parent;

  QFont font;
  QFontMetrics fontMetrics;

  QFont pixelFont;
  QFontMetrics pixelFontMetrics;

  QHash<ushort, CharData> characters;
  QList<GLuint> textures;

  GLint xOffset;
  GLint yOffset;

  // Modern OpenGL resources
  QOpenGLShaderProgram* _shaderProgram;
  QOpenGLVertexArrayObject* _vao;
  QOpenGLBuffer* _vbo;
  QOpenGLBuffer* _ebo;
};

#endif  // __MI_OPENGL_TEXT_H__