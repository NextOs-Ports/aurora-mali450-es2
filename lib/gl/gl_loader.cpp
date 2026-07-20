#include "gl_core.hpp"

#include "../internal.hpp"

#include <dlfcn.h>

namespace aurora::gl {
namespace {
Module Log("aurora::gl");

bool g_loaded = false;
void* g_libGLESv2 = nullptr;
void* g_libEGL = nullptr;

void* dlsym_lib(void*& handle, const char* const* names, const char* sym) {
  if (handle == nullptr) {
    for (const char* const* n = names; *n != nullptr; ++n) {
      handle = dlopen(*n, RTLD_NOW | RTLD_GLOBAL);
      if (handle != nullptr) {
        break;
      }
    }
  }
  return handle != nullptr ? dlsym(handle, sym) : nullptr;
}

// Resolution chain, mirroring gpu.cpp's sdl2shim_egl_get_proc: the shim's own
// getProc first; then a direct dlsym against libGLESv2 / libEGL for the symbols
// some blobs' eglGetProcAddress refuses (notably core EGL entry points on
// PowerVR pre-EGL-1.5).
void* resolve(ProcAddressFn getProc, const char* name) {
  if (getProc != nullptr) {
    if (void* p = getProc(name)) {
      return p;
    }
  }
  static const char* const kGles[] = {"libGLESv2.so", "libGLESv2.so.2", nullptr};
  static const char* const kEgl[] = {"libEGL.so", "libEGL.so.1", nullptr};
  if (void* p = dlsym_lib(g_libGLESv2, kGles, name)) {
    return p;
  }
  return dlsym_lib(g_libEGL, kEgl, name);
}
} // namespace

GlProcTable gl;

bool load(ProcAddressFn getProc) {
  bool missing = false;

  // REQ: a core ES 3.0 entry point the render path cannot run without; a null
  // fails load(). OPT: device-only (EGL interop) or feature-gated (compressed
  // textures, MSAA, mapped buffers, KHR_debug) -- callers null-check these.
#define REQ(member, name)                                                                                    \
  gl.member = reinterpret_cast<decltype(gl.member)>(resolve(getProc, name));                                 \
  if (gl.member == nullptr) {                                                                                \
    Log.warn("[gl] missing required entry point {}", name);                                                  \
    missing = true;                                                                                          \
  }
#define OPT(member, name) gl.member = reinterpret_cast<decltype(gl.member)>(resolve(getProc, name));

  // Buffers
  REQ(GenBuffers, "glGenBuffers")
  REQ(DeleteBuffers, "glDeleteBuffers")
  REQ(BindBuffer, "glBindBuffer")
  REQ(BufferData, "glBufferData")
  REQ(BufferSubData, "glBufferSubData")
  OPT(BindBufferRange, "glBindBufferRange")
  OPT(BindBufferBase, "glBindBufferBase")
  OPT(MapBufferRange, "glMapBufferRange")
  OPT(UnmapBuffer, "glUnmapBuffer")
  OPT(FlushMappedBufferRange, "glFlushMappedBufferRange")
  OPT(CopyBufferSubData, "glCopyBufferSubData")
  // GL_EXT_buffer_storage on GLES (libmali/PowerVR); the ARB/core name is the desktop spelling.
  OPT(BufferStorage, "glBufferStorageEXT")
  if (gl.BufferStorage == nullptr) {
    OPT(BufferStorage, "glBufferStorage")
  }

  // Vertex arrays
  OPT(GenVertexArrays, "glGenVertexArrays")
  OPT(DeleteVertexArrays, "glDeleteVertexArrays")
  OPT(BindVertexArray, "glBindVertexArray")
#ifdef AURORA_GLES2
  if (gl.GenVertexArrays == nullptr) OPT(GenVertexArrays, "glGenVertexArraysOES")
  if (gl.DeleteVertexArrays == nullptr) OPT(DeleteVertexArrays, "glDeleteVertexArraysOES")
  if (gl.BindVertexArray == nullptr) OPT(BindVertexArray, "glBindVertexArrayOES")
#endif
  if (gl.GenVertexArrays == nullptr || gl.DeleteVertexArrays == nullptr || gl.BindVertexArray == nullptr) {
    Log.warn("[gl] missing required vertex-array entry points");
    missing = true;
  }
  REQ(EnableVertexAttribArray, "glEnableVertexAttribArray")
  REQ(DisableVertexAttribArray, "glDisableVertexAttribArray")
  REQ(VertexAttribPointer, "glVertexAttribPointer")
  OPT(VertexAttribIPointer, "glVertexAttribIPointer")

  // Textures
  REQ(GenTextures, "glGenTextures")
  REQ(DeleteTextures, "glDeleteTextures")
  REQ(BindTexture, "glBindTexture")
  REQ(ActiveTexture, "glActiveTexture")
  OPT(TexStorage2D, "glTexStorage2D")
  if (gl.TexStorage2D == nullptr) OPT(TexStorage2D, "glTexStorage2DEXT")
  REQ(TexImage2D, "glTexImage2D")
  REQ(TexSubImage2D, "glTexSubImage2D")
  OPT(CompressedTexImage2D, "glCompressedTexImage2D")
  OPT(CompressedTexSubImage2D, "glCompressedTexSubImage2D")
  REQ(CopyTexSubImage2D, "glCopyTexSubImage2D")
  REQ(TexParameteri, "glTexParameteri")
  REQ(TexParameterf, "glTexParameterf")
  REQ(GenerateMipmap, "glGenerateMipmap")
  REQ(PixelStorei, "glPixelStorei")

  // Sampler objects
  OPT(GenSamplers, "glGenSamplers")
  OPT(DeleteSamplers, "glDeleteSamplers")
  OPT(BindSampler, "glBindSampler")
  OPT(SamplerParameteri, "glSamplerParameteri")
  OPT(SamplerParameterf, "glSamplerParameterf")

  // Framebuffers / renderbuffers
  REQ(GenFramebuffers, "glGenFramebuffers")
  REQ(DeleteFramebuffers, "glDeleteFramebuffers")
  REQ(BindFramebuffer, "glBindFramebuffer")
  REQ(FramebufferTexture2D, "glFramebufferTexture2D")
  REQ(FramebufferRenderbuffer, "glFramebufferRenderbuffer")
  REQ(CheckFramebufferStatus, "glCheckFramebufferStatus")
  OPT(BlitFramebuffer, "glBlitFramebuffer")
  OPT(InvalidateFramebuffer, "glInvalidateFramebuffer")
  if (gl.InvalidateFramebuffer == nullptr) OPT(InvalidateFramebuffer, "glDiscardFramebufferEXT")
  OPT(DrawBuffers, "glDrawBuffers")
  OPT(ReadBuffer, "glReadBuffer")
  REQ(ReadPixels, "glReadPixels")
  REQ(GenRenderbuffers, "glGenRenderbuffers")
  REQ(DeleteRenderbuffers, "glDeleteRenderbuffers")
  REQ(BindRenderbuffer, "glBindRenderbuffer")
  REQ(RenderbufferStorage, "glRenderbufferStorage")
  OPT(RenderbufferStorageMultisample, "glRenderbufferStorageMultisample")

  // Clears
  REQ(ClearColor, "glClearColor")
  REQ(ClearDepthf, "glClearDepthf")
  REQ(ClearStencil, "glClearStencil")
  REQ(Clear, "glClear")
  OPT(ClearBufferfv, "glClearBufferfv")
  OPT(ClearBufferfi, "glClearBufferfi")
  OPT(ClearBufferiv, "glClearBufferiv")
  OPT(ClearBufferuiv, "glClearBufferuiv")

  // Shaders / programs
  REQ(CreateShader, "glCreateShader")
  REQ(ShaderSource, "glShaderSource")
  REQ(CompileShader, "glCompileShader")
  REQ(GetShaderiv, "glGetShaderiv")
  REQ(GetShaderInfoLog, "glGetShaderInfoLog")
  REQ(DeleteShader, "glDeleteShader")
  REQ(CreateProgram, "glCreateProgram")
  REQ(AttachShader, "glAttachShader")
  REQ(BindAttribLocation, "glBindAttribLocation")
  REQ(LinkProgram, "glLinkProgram")
  REQ(GetProgramiv, "glGetProgramiv")
  REQ(GetProgramInfoLog, "glGetProgramInfoLog")
  REQ(UseProgram, "glUseProgram")
  REQ(DeleteProgram, "glDeleteProgram")
  REQ(GetUniformLocation, "glGetUniformLocation")
  REQ(Uniform1i, "glUniform1i")
  REQ(Uniform4fv, "glUniform4fv")
  OPT(GetUniformBlockIndex, "glGetUniformBlockIndex")
  OPT(UniformBlockBinding, "glUniformBlockBinding")
  OPT(GetActiveUniformBlockiv, "glGetActiveUniformBlockiv")
  // Program binary (ES 3.0 core, but OPT: the binary cache degrades to source
  // compile when absent, and the ProgramParameteri hint is inert on drivers that
  // do not need it). glProgramBinary is the base spelling; some GLES stacks only
  // export the OES alias.
  OPT(GetProgramBinary, "glGetProgramBinary")
  if (gl.GetProgramBinary == nullptr) {
    OPT(GetProgramBinary, "glGetProgramBinaryOES")
  }
  OPT(ProgramBinary, "glProgramBinary")
  if (gl.ProgramBinary == nullptr) {
    OPT(ProgramBinary, "glProgramBinaryOES")
  }
  OPT(ProgramParameteri, "glProgramParameteri")

  // Fixed-function state
  REQ(Enable, "glEnable")
  REQ(Disable, "glDisable")
  REQ(BlendFuncSeparate, "glBlendFuncSeparate")
  REQ(BlendEquationSeparate, "glBlendEquationSeparate")
  REQ(BlendColor, "glBlendColor")
  REQ(ColorMask, "glColorMask")
  REQ(DepthMask, "glDepthMask")
  REQ(DepthFunc, "glDepthFunc")
  REQ(DepthRangef, "glDepthRangef")
  REQ(CullFace, "glCullFace")
  REQ(FrontFace, "glFrontFace")
  REQ(PolygonOffset, "glPolygonOffset")
  REQ(Viewport, "glViewport")
  REQ(Scissor, "glScissor")
  REQ(StencilFuncSeparate, "glStencilFuncSeparate")
  REQ(StencilOpSeparate, "glStencilOpSeparate")
  REQ(StencilMaskSeparate, "glStencilMaskSeparate")

  // Draw
  REQ(DrawArrays, "glDrawArrays")
  REQ(DrawElements, "glDrawElements")
  OPT(DrawArraysInstanced, "glDrawArraysInstanced")
  OPT(DrawElementsInstanced, "glDrawElementsInstanced")

  // Sync / query
  OPT(FenceSync, "glFenceSync")
  OPT(ClientWaitSync, "glClientWaitSync")
  OPT(WaitSync, "glWaitSync")
  OPT(DeleteSync, "glDeleteSync")
  REQ(Finish, "glFinish")
  REQ(Flush, "glFlush")
  REQ(GetError, "glGetError")
  REQ(GetString, "glGetString")
  OPT(GetStringi, "glGetStringi")
  REQ(GetIntegerv, "glGetIntegerv")
  REQ(GetFloatv, "glGetFloatv")

  // Debug (KHR_debug; optional)
  OPT(DebugMessageCallback, "glDebugMessageCallback")
  OPT(DebugMessageControl, "glDebugMessageControl")
  OPT(PushDebugGroup, "glPushDebugGroup")
  OPT(PopDebugGroup, "glPopDebugGroup")
  OPT(ObjectLabel, "glObjectLabel")

  // EGL (device path only -- absent on the desktop SDL_GL path, so all OPT)
  OPT(eglGetCurrentContext, "eglGetCurrentContext")
  OPT(eglGetCurrentDisplay, "eglGetCurrentDisplay")
  OPT(eglGetCurrentSurface, "eglGetCurrentSurface")
  OPT(eglMakeCurrent, "eglMakeCurrent")
  OPT(eglCreateContext, "eglCreateContext")
  OPT(eglDestroyContext, "eglDestroyContext")
  OPT(eglCreatePbufferSurface, "eglCreatePbufferSurface")
  OPT(eglDestroySurface, "eglDestroySurface")
  OPT(eglChooseConfig, "eglChooseConfig")
  OPT(eglGetConfigAttrib, "eglGetConfigAttrib")
  OPT(eglGetError, "eglGetError")
  OPT(eglQueryString, "eglQueryString")
  OPT(eglCreateImageKHR, "eglCreateImageKHR")
  OPT(eglDestroyImageKHR, "eglDestroyImageKHR")
  OPT(eglCreateSyncKHR, "eglCreateSyncKHR")
  OPT(eglDestroySyncKHR, "eglDestroySyncKHR")
  OPT(eglClientWaitSyncKHR, "eglClientWaitSyncKHR")
  OPT(eglWaitSyncKHR, "eglWaitSyncKHR")
  OPT(glEGLImageTargetTexture2DOES, "glEGLImageTargetTexture2DOES")

#undef REQ
#undef OPT

  g_loaded = !missing;
  if (missing) {
    Log.error("[gl] entry-point table incomplete; GL backend cannot initialize");
  }
  return g_loaded;
}

bool loaded() { return g_loaded; }

} // namespace aurora::gl
