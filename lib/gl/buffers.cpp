#include "buffers.hpp"

#include "census.hpp"
#include "../internal.hpp"
#include "../webgpu/gpu.hpp"

#include <cstdint>
#include <cstring>
#ifdef AURORA_GLES2
#include <memory>
#include <unordered_map>
#endif

namespace aurora::gl {
namespace {
Module Log("aurora::gl");
#ifdef AURORA_GLES2
struct CpuUniformBuffer {
  std::unique_ptr<uint8_t[]> bytes;
  uint64_t size = 0;
};
std::unordered_map<GLuint, CpuUniformBuffer> g_cpuUniformBuffers;
GLuint g_nextCpuUniformId = 0x80000000u;
#endif
} // namespace

Buffer create_buffer(GLenum target, uint64_t size, bool dynamic, bool persistent) {
#ifdef AURORA_GLES2
  if (target == GL_UNIFORM_BUFFER) {
    const GLuint id = g_nextCpuUniformId++;
    CpuUniformBuffer storage{.bytes = std::make_unique<uint8_t[]>(static_cast<size_t>(size)), .size = size};
    std::memset(storage.bytes.get(), 0, static_cast<size_t>(size));
    void* mapped = storage.bytes.get();
    g_cpuUniformBuffers.emplace(id, std::move(storage));
    census::buffers.add(static_cast<int64_t>(size));
    return Buffer{.id = id, .target = target, .size = size, .mapped = mapped};
  }
#endif
  GLuint id = 0;
  gl.GenBuffers(1, &id);
  gl.BindBuffer(target, id);

  void* mapped = nullptr;
  if (persistent && webgpu::g_bufferStorageSupported && gl.BufferStorage != nullptr && gl.MapBufferRange != nullptr) {
    // Immutable persistent-coherent write-combine storage. GL_DYNAMIC_STORAGE_BIT is deliberately
    // kept so that, if the mapping below fails, glBufferSubData is still legal on this buffer (the
    // upload_buffer fallback path). GL_MAP_COHERENT_BIT means CPU writes reach the GPU with no
    // glFlushMappedBufferRange.
    constexpr GLbitfield kStorageFlags =
        GL_DYNAMIC_STORAGE_BIT | GL_MAP_WRITE_BIT | GL_MAP_PERSISTENT_BIT | GL_MAP_COHERENT_BIT;
    constexpr GLbitfield kMapFlags = GL_MAP_WRITE_BIT | GL_MAP_PERSISTENT_BIT | GL_MAP_COHERENT_BIT;
    gl.BufferStorage(target, static_cast<GLsizeiptr>(size), nullptr, kStorageFlags);
    mapped = gl.MapBufferRange(target, 0, static_cast<GLsizeiptr>(size), kMapFlags);
    if (mapped == nullptr) {
      // Storage is immutable now, but DYNAMIC_STORAGE keeps glBufferSubData valid -- leave mapped
      // null and let upload_buffer take the glBufferSubData branch.
      Log.warn("[gl] persistent map failed for {}-byte buffer; using glBufferSubData on immutable storage", size);
    }
  } else {
    gl.BufferData(target, static_cast<GLsizeiptr>(size), nullptr, dynamic ? GL_DYNAMIC_DRAW : GL_STATIC_DRAW);
  }
  census::buffers.add(static_cast<int64_t>(size));
  return Buffer{.id = id, .target = target, .size = size, .mapped = mapped};
}

void upload_buffer(const Buffer& buffer, uint64_t offset, const void* data, uint64_t size) {
  if (buffer.id == 0 || size == 0) {
    return;
  }
  if (buffer.mapped != nullptr) {
    // Coherent persistent mapping: the memcpy IS the upload -- no bind, no glBufferSubData, no
    // driver-side copy, no flush. The caller guarantees the GPU is not reading [offset, offset+size).
    std::memcpy(static_cast<uint8_t*>(buffer.mapped) + offset, data, size);
    return;
  }
  gl.BindBuffer(buffer.target, buffer.id);
  gl.BufferSubData(buffer.target, static_cast<GLintptr>(offset), static_cast<GLsizeiptr>(size), data);
}

const uint8_t* uniform_buffer_data(GLuint id, uint64_t offset, uint64_t size) noexcept {
#ifdef AURORA_GLES2
  const auto it = g_cpuUniformBuffers.find(id);
  if (it == g_cpuUniformBuffers.end() || offset > it->second.size || size > it->second.size - offset) {
    return nullptr;
  }
  return it->second.bytes.get() + offset;
#else
  (void)id;
  (void)offset;
  (void)size;
  return nullptr;
#endif
}

void destroy_buffer(Buffer& buffer) noexcept {
  if (buffer.id != 0) {
    census::buffers.sub(static_cast<int64_t>(buffer.size));
#ifdef AURORA_GLES2
    if (buffer.target == GL_UNIFORM_BUFFER) {
      g_cpuUniformBuffers.erase(buffer.id);
      buffer.id = 0;
      buffer.mapped = nullptr;
      return;
    }
#endif
    if (buffer.mapped != nullptr && gl.UnmapBuffer != nullptr) {
      gl.BindBuffer(buffer.target, buffer.id);
      gl.UnmapBuffer(buffer.target);
    }
    gl.DeleteBuffers(1, &buffer.id);
    buffer.id = 0;
    buffer.mapped = nullptr;
  }
}

} // namespace aurora::gl
