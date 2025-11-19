#include <stddef.h>
typedef unsigned int GLenum;
typedef unsigned char GLubyte;
typedef unsigned int GLuint;
typedef int GLsizei;
typedef int GLint;
typedef float GLfloat;
typedef double GLdouble;
typedef unsigned char GLboolean;
typedef ptrdiff_t GLsizeiptr;

const GLubyte* glGetString(GLenum name) { static const GLubyte empty[] = ""; return empty; }
void glMatrixMode(GLenum mode) {}
void glLoadIdentity(void) {}
void glLoadMatrixf(const GLfloat *m) {}
void glMultMatrixf(const GLfloat *m) {}
void glOrtho(GLdouble l, GLdouble r, GLdouble b, GLdouble t, GLdouble n, GLdouble f) {}
void glFrustum(GLdouble l, GLdouble r, GLdouble b, GLdouble t, GLdouble n, GLdouble f) {}
void glBegin(GLenum mode) {}
void glEnd(void) {}
void glVertex2f(GLfloat x, GLfloat y) {}
void glVertex3f(GLfloat x, GLfloat y, GLfloat z) {}
void glColor4f(GLfloat r, GLfloat g, GLfloat b, GLfloat a) {}
void glColor3f(GLfloat r, GLfloat g, GLfloat b) {}
void glClear(GLenum mask) {}
void glClearColor(GLfloat r, GLfloat g, GLfloat b, GLfloat a) {}
void glFlush(void) {}
void glViewport(GLint x, GLint y, GLsizei w, GLsizei h) {}
void glEnable(GLenum cap) {}
void glDisable(GLenum cap) {}
void glBlendFunc(GLenum sfactor, GLenum dfactor) {}
void glHint(GLenum target, GLenum mode) {}
void glShadeModel(GLenum mode) {}
void glDepthMask(GLboolean flag) {}
void glClearDepth(GLdouble depth) {}
void glDrawBuffer(GLenum mode) {}
void glReadBuffer(GLenum mode) {}
void glTexEnvi(GLenum target, GLenum pname, GLint param) {}
void glTexParameteri(GLenum target, GLenum pname, GLint param) {}
void glBindTexture(GLenum target, GLuint texture) {}
void glGenTextures(GLsizei n, GLuint *textures) { if(textures) for(GLsizei i=0;i<n;++i) textures[i]=0; }
void glDeleteTextures(GLsizei n, const GLuint *textures) {}
void glTexImage2D(GLenum target, GLint level, GLint internalFormat, GLsizei width, GLsizei height, GLint border, GLenum format, GLenum type, const void *pixels) {}
void glPixelStorei(GLenum pname, GLint param) {}
void glScissor(GLint x, GLint y, GLsizei width, GLsizei height) {}
void glGetIntegerv(GLenum pname, GLint *data) { if(data) *data = 0; }
void glGetFloatv(GLenum pname, GLfloat *data) { if(data) *data = 0; }
GLenum glGetError(void) { return 0; }
void glRotatef(GLfloat angle, GLfloat x, GLfloat y, GLfloat z) {}
void glTranslatef(GLfloat x, GLfloat y, GLfloat z) {}
void glScalef(GLfloat x, GLfloat y, GLfloat z) {}
void glPushMatrix(void) {}
void glPopMatrix(void) {}
void glTexSubImage2D(GLenum target, GLint level, GLint xoffset, GLint yoffset, GLsizei width, GLsizei height, GLenum format, GLenum type, const void *pixels) {}
void glDisableClientState(GLenum array) {}
void glEnableClientState(GLenum array) {}
void glVertexPointer(GLint size, GLenum type, GLsizei stride, const void *pointer) {}
void glTexCoordPointer(GLint size, GLenum type, GLsizei stride, const void *pointer) {}
void glColorPointer(GLint size, GLenum type, GLsizei stride, const void *pointer) {}
void glDrawArrays(GLenum mode, GLint first, GLsizei count) {}
void glDrawElements(GLenum mode, GLsizei count, GLenum type, const void *indices) {}
void glBindFramebuffer(GLenum target, GLuint framebuffer) {}
void glDeleteFramebuffers(GLsizei n, const GLuint *framebuffers) {}
void glGenFramebuffers(GLsizei n, GLuint *framebuffers) { if(framebuffers) for(GLsizei i=0;i<n;++i) framebuffers[i]=0; }
void glFramebufferTexture2D(GLenum target, GLenum attachment, GLenum textarget, GLuint texture, GLint level) {}
void glCheckFramebufferStatus(GLenum target) {}
void glActiveTexture(GLenum texture) {}
void glBindBuffer(GLenum target, GLuint buffer) {}
void glGenBuffers(GLsizei n, GLuint *buffers) { if(buffers) for(GLsizei i=0;i<n;++i) buffers[i]=0; }
void glDeleteBuffers(GLsizei n, const GLuint *buffers) {}
void glBufferData(GLenum target, GLsizeiptr size, const void *data, GLenum usage) {}
void glBufferSubData(GLenum target, GLsizeiptr offset, GLsizeiptr size, const void *data) {}
void glUseProgram(GLuint program) {}
void glUniformMatrix4fv(GLint location, GLsizei count, GLboolean transpose, const GLfloat *value) {}
void glUniform1i(GLint location, GLint v0) {}
void glUniform1f(GLint location, GLfloat v0) {}
void glUniform2f(GLint location, GLfloat v0, GLfloat v1) {}
void glUniform3f(GLint location, GLfloat v0, GLfloat v1, GLfloat v2) {}
void glUniform4f(GLint location, GLfloat v0, GLfloat v1, GLfloat v2, GLfloat v3) {}
void glAttachShader(GLuint program, GLuint shader) {}
void glCompileShader(GLuint shader) {}
GLuint glCreateShader(GLenum type) { return 0; }
GLuint glCreateProgram(void) { return 0; }
void glLinkProgram(GLuint program) {}
void glShaderSource(GLuint shader, GLsizei count, const char **string, const GLint *length) {}
void glDeleteShader(GLuint shader) {}
void glDeleteProgram(GLuint program) {}
void glDetachShader(GLuint program, GLuint shader) {}
void glGetShaderiv(GLuint shader, GLenum pname, GLint *params) { if(params) *params = 0; }
void glGetProgramiv(GLuint program, GLenum pname, GLint *params) { if(params) *params = 0; }
void glGetShaderInfoLog(GLuint shader, GLsizei bufSize, GLsizei *length, char *infoLog) { if(length) *length = 0; if(infoLog && bufSize>0) infoLog[0]='\0'; }
void glGetProgramInfoLog(GLuint program, GLsizei bufSize, GLsizei *length, char *infoLog) { if(length) *length = 0; if(infoLog && bufSize>0) infoLog[0]='\0'; }
void glDeleteVertexArrays(GLsizei n, const GLuint *arrays) {}
void glGenVertexArrays(GLsizei n, GLuint *arrays) { if(arrays) for(GLsizei i=0;i<n;++i) arrays[i]=0; }
void glBindVertexArray(GLuint array) {}
void glDepthFunc(GLenum func) {}
void glPolygonMode(GLenum face, GLenum mode) {}
void glLineWidth(GLfloat width) {}
void glPointSize(GLfloat size) {}
void glNormal3f(GLfloat nx, GLfloat ny, GLfloat nz) {}
void glNormalPointer(GLenum type, GLsizei stride, const void *pointer) {}
void glClipPlane(GLenum plane, const GLdouble *equation) {}
void glLightfv(GLenum light, GLenum pname, const GLfloat *params) {}
void glLightModelfv(GLenum pname, const GLfloat *params) {}
void glMaterialfv(GLenum face, GLenum pname, const GLfloat *params) {}
void glMaterialf(GLenum face, GLenum pname, GLfloat param) {}
void glTexGeni(GLenum coord, GLenum pname, GLint param) {}
void glTexGenfv(GLenum coord, GLenum pname, const GLfloat *params) {}
void glFogf(GLenum pname, GLfloat param) {}
void glFogi(GLenum pname, GLint param) {}
void glFogfv(GLenum pname, const GLfloat *params) {}
void glAlphaFunc(GLenum func, GLfloat ref) {}
void glDrawPixels(GLsizei width, GLsizei height, GLenum format, GLenum type, const void *pixels) {}
void glRasterPos2f(GLfloat x, GLfloat y) {}
void glRasterPos3f(GLfloat x, GLfloat y, GLfloat z) {}
void glBitmap(GLsizei width, GLsizei height, GLfloat xorig, GLfloat yorig, GLfloat xmove, GLfloat ymove, const unsigned char *bitmap) {}
