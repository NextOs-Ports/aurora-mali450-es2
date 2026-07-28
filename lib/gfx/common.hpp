#pragma once

#include "../internal.hpp"
#include "../webgpu/gpu.hpp"

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <type_traits>
#include <utility>

#include <aurora/gfx.h>
#include <aurora/gfx.hpp>
#include <aurora/math.hpp>
#include <dolphin/gx/GXEnum.h>
#define XXH_STATIC_LINKING_ONLY
#include <xxhash.h>

namespace aurora {
#if INTPTR_MAX == INT32_MAX
using HashType = XXH32_hash_t;
#else
using HashType = XXH64_hash_t;
#endif
static inline HashType xxh3_hash_s(const void* input, size_t len, HashType seed = 0) {
  return static_cast<HashType>(XXH3_64bits_withSeed(input, len, seed));
}
template <typename T>
static inline HashType xxh3_hash(const T& input, HashType seed = 0) {
  // Validate that the type has no padding bytes, which can easily cause
  // hash mismatches. This also disallows floats, but that's okay for us.
  static_assert(std::has_unique_object_representations_v<T>);
  return xxh3_hash_s(&input, sizeof(T), seed);
}

class Hasher {
public:
  explicit Hasher(const XXH64_hash_t seed = 0) {
    XXH3_INITSTATE(&state);
    XXH3_64bits_reset_withSeed(&state, seed);
  }

  void update(const void* data, const size_t size) { XXH3_64bits_update(&state, data, size); }

  template <typename T>
  void update(const T& data) {
    static_assert(std::has_unique_object_representations_v<T>);
    update(&data, sizeof(T));
  }

  [[nodiscard]] XXH64_hash_t digest() const { return XXH3_64bits_digest(&state); }

private:
  XXH3_state_t state;
};

class ByteBuffer {
public:
  ByteBuffer() noexcept = default;
  explicit ByteBuffer(size_t size) noexcept
  : m_data(static_cast<uint8_t*>(calloc(1, size))), m_length(size), m_capacity(size) {}
  explicit ByteBuffer(uint8_t* data, size_t size) noexcept : m_data(data), m_capacity(size), m_owned(false) {}
  ~ByteBuffer() noexcept {
    if (m_data != nullptr && m_owned) {
      free(m_data);
    }
  }
  ByteBuffer(ByteBuffer&& rhs) noexcept
  : m_data(rhs.m_data), m_length(rhs.m_length), m_capacity(rhs.m_capacity), m_owned(rhs.m_owned) {
    rhs.m_data = nullptr;
    rhs.m_length = 0;
    rhs.m_capacity = 0;
    rhs.m_owned = true;
  }
  ByteBuffer& operator=(ByteBuffer&& rhs) noexcept {
    if (m_data != nullptr && m_owned) {
      free(m_data);
    }
    m_data = rhs.m_data;
    m_length = rhs.m_length;
    m_capacity = rhs.m_capacity;
    m_owned = rhs.m_owned;
    rhs.m_data = nullptr;
    rhs.m_length = 0;
    rhs.m_capacity = 0;
    rhs.m_owned = true;
    return *this;
  }
  ByteBuffer(ByteBuffer const&) = delete;
  ByteBuffer& operator=(ByteBuffer const&) = delete;
  operator ArrayRef<uint8_t>() const noexcept { return {m_data, m_length}; }

  [[nodiscard]] uint8_t* data() noexcept { return m_data; }
  [[nodiscard]] const uint8_t* data() const noexcept { return m_data; }
  [[nodiscard]] size_t size() const noexcept { return m_length; }
  [[nodiscard]] bool empty() const noexcept { return m_length == 0; }

  void append(const void* data, size_t size) {
    resize(m_length + size, false);
    memcpy(m_data + m_length, data, size);
    m_length += size;
  }

  template <typename T>
  void append(const T& obj) {
    append(&obj, sizeof(T));
  }

  void append_zeroes(size_t size) {
    resize(m_length + size, true);
    m_length += size;
  }

  void release() {
    if (m_data != nullptr && m_owned) {
      free(m_data);
    }
    m_data = nullptr;
    m_length = 0;
    m_capacity = 0;
    m_owned = true;
  }

  void clear() {
    m_length = 0;
  }

  void reserve_extra(size_t size) { resize(m_length + size, true); }

  ByteBuffer clone() const {
    ByteBuffer clone{m_length};
    std::memcpy(clone.data(), m_data, m_length);
    return clone;
  }

private:
  uint8_t* m_data = nullptr;
  size_t m_length = 0;
  size_t m_capacity = 0;
  bool m_owned = true;

  void resize(size_t size, bool zeroed) {
    if (size == 0) {
      clear();
    } else if (m_data == nullptr) {
      if (zeroed) {
        m_data = static_cast<uint8_t*>(calloc(1, size));
      } else {
        m_data = static_cast<uint8_t*>(malloc(size));
      }
      m_owned = true;
    } else if (size > m_capacity) {
      if (!m_owned) {
        abort();
      }
      // Exponential expansion to avoid O(n^2) time complexity.
      if (size < m_capacity * 2) {
        size = m_capacity * 2;
      }
      m_data = static_cast<uint8_t*>(realloc(m_data, size));
      if (zeroed) {
        memset(m_data + m_capacity, 0, size - m_capacity);
      }
    } else {
      return;
    }
    m_capacity = size;
  }
};
} // namespace aurora

namespace aurora::gfx {
inline constexpr bool UseTextureBuffer = true;
inline constexpr uint64_t UniformBufferSize = 25165824;  // 24mb
inline constexpr uint64_t VertexBufferSize = 5242880;    // 5mb
inline constexpr uint64_t IndexBufferSize = 1048576;     // 1mb
inline constexpr uint64_t StorageBufferSize = 8388608;   // 8mb
inline constexpr uint64_t TextureUploadSize = 25165824;  // 24mb

extern AuroraStats g_stats;
extern uint32_t g_drawCallCount;
extern uint32_t g_mergedDrawCallCount;
// Recording-thread stall accounting for the periodic [fps] line. Each accumulator counts
// nanoseconds the recording thread spent blocked this logging interval: worker-queue
// backpressure (push into a full BoundedQueue), waiting for the worker to publish a
// presentable frame, and the main-thread blit+swap itself. Written by their respective
// modules, read+reset by end_frame's stats block (all on the recording thread except the
// queue counter, which is also recording-thread-only — plain atomics keep it simple).
namespace perfstall {
extern std::atomic<uint64_t> queueWaitNs;
extern std::atomic<uint64_t> presentWaitNs;
extern std::atomic<uint64_t> blitSwapNs;
// Decode-side split of the GX drain (all recording-thread): geometry content hashing +
// cache lookups, cache-miss vertex expansion (+ index build + ring/cache push), per-draw
// pipeline/shader-info/bind-group construction, and per-draw uniform building.
extern std::atomic<uint64_t> geomHashNs;
extern std::atomic<uint64_t> geomExpandNs;
extern std::atomic<uint64_t> drawCfgNs;
extern std::atomic<uint64_t> drawUniNs;
} // namespace perfstall
extern gl::Buffer g_vertexBuffer;
extern gl::Buffer g_uniformBuffer;
extern gl::Buffer g_indexBuffer;
extern gl::Buffer g_storageBuffer;
// Persistent cache for CPU-expanded native-fetch geometry; survives across frames
// (unlike the per-frame staging rings above). Created lazily on first use.
extern gl::Buffer g_nativeVertexCacheBuffer;
extern gl::Buffer g_nativeIndexCacheBuffer;

using BindGroupRef = HashType;
using PipelineRef = HashType;
using SamplerRef = HashType;
using ShaderRef = HashType;

struct ClipRect {
  int32_t x;
  int32_t y;
  int32_t width;
  int32_t height;

  bool operator==(const ClipRect& rhs) const { return memcmp(this, &rhs, sizeof(*this)) == 0; }
  bool operator!=(const ClipRect& rhs) const { return !(*this == rhs); }
};

using webgpu::Viewport;

struct TextureRef;
using TextureHandle = std::shared_ptr<TextureRef>;
// No WebGPU CommandEncoder on GL: aurora's own command list + the render worker
// already sequence everything, so the present callback just issues GL directly.
using EndFrameCallback = std::function<void()>;

enum class ShaderType : uint8_t {
  Clear = 0,
  GX = 1,
  Rml = 2,
};

void initialize();
void shutdown();

bool begin_frame();
void finish();
void end_frame(EndFrameCallback callback);
uint32_t current_frame() noexcept;
void render_pass(gl::PassEncoder& pass, uint32_t idx);
void after_submit() noexcept;
void gpu_synchronize();
void after_present() noexcept;
float calculate_fps() noexcept;
void resolve_pass_into(TextureHandle texture, ClipRect rect, bool clearColor, bool clearAlpha, bool clearDepth,
                       Vec4<float> clearColorValue, float clearDepthValue, GXTexFmt resolveFormat = GX_TF_RGBA8);

struct ColorPassDescriptor {
  const char* label = nullptr;
  gl::Texture colorView;
  gl::Texture resolveView;
  gl::Texture depthStencilView;
  gl::Extent3D targetSize;
  uint32_t sampleCount = 1;
  gl::LoadOp colorLoadOp = gl::LoadOp::Clear;
  gl::StoreOp colorStoreOp = gl::StoreOp::Store;
  gl::Color clearColor{0.f, 0.f, 0.f, 0.f};
  bool hasDepth = false;
  gl::LoadOp depthLoadOp = gl::LoadOp::Undefined;
  gl::StoreOp depthStoreOp = gl::StoreOp::Undefined;
  float depthClearValue = 0.f;
  // Combined depth-stencil attachment whose depth aspect is present but unused (stencil-only passes
  // on a packed format). Marks the depth aspect read-only so no load/store op is required.
  bool depthReadOnly = false;
  bool hasStencil = false;
  gl::LoadOp stencilLoadOp = gl::LoadOp::Undefined;
  gl::StoreOp stencilStoreOp = gl::StoreOp::Undefined;
  uint32_t stencilClearValue = 0;
};

// A texture + subregion origin for a texture-to-texture copy (EFB copies). The
// WebGPU TexelCopyTextureInfo/Extent3D pair collapses to gl handles.
struct TextureCopyView {
  gl::Texture texture;
  gl::Origin3D origin;
};

void begin_color_pass(const ColorPassDescriptor& desc);
void end_color_pass();
void queue_texture_copy(TextureCopyView src, TextureCopyView dst, gl::Extent3D size, bool flipY = false);

void begin_offscreen(uint32_t width, uint32_t height);
void end_offscreen();
bool is_offscreen() noexcept;
uint32_t get_sample_count() noexcept;
void clear_caches() noexcept;

namespace tex_palette_conv {
struct ConvRequest;
} // namespace tex_palette_conv
void queue_palette_conv(tex_palette_conv::ConvRequest req);

Range push_verts(const uint8_t* data, size_t length, size_t alignment);
template <typename T>
static Range push_verts(ArrayRef<T> data, size_t alignment) {
  return push_verts(reinterpret_cast<const uint8_t*>(data.data()), data.size() * sizeof(T), alignment);
}
Range push_indices(const uint8_t* data, size_t length, size_t alignment);
template <typename T>
static Range push_indices(ArrayRef<T> data, size_t alignment) {
  return push_indices(reinterpret_cast<const uint8_t*>(data.data()), data.size() * sizeof(T), alignment);
}
// Upload into the active bank of the persistent native geometry cache buffers. Returns
// {range, true} on success; {_, false} when that bank is exhausted (caller should fall
// back to the per-frame ring and request a bank rotation). Ranges are consumed by
// native-fetch draws with DrawData::cachedGeometry set, which bind these buffers instead
// of the per-frame rings.
std::pair<Range, bool> push_native_cached_verts(const uint8_t* data, size_t length);
std::pair<Range, bool> push_native_cached_indices(const uint8_t* data, size_t length);
// Deferred bank rotation of the native geometry cache; takes effect at the next
// begin_frame. Only the bank selected for reuse becomes stale, while the other bank
// remains hot. The generation counter lets CPU-side maps discard entries from the
// recycled bank.
void request_native_geometry_cache_reset() noexcept;
uint32_t native_geom_cache_generation() noexcept;
uint32_t native_geom_cache_bank() noexcept;
Range push_uniform(const uint8_t* data, size_t length);
template <typename T>
static Range push_uniform(const T& data) {
  return push_uniform(reinterpret_cast<const uint8_t*>(&data), sizeof(T));
}
Range push_storage(const uint8_t* data, size_t length);
template <typename T>
static Range push_storage(ArrayRef<T> data) {
  return push_storage(reinterpret_cast<const uint8_t*>(data.data()), data.size() * sizeof(T));
}
template <typename T>
static Range push_storage(const T& data) {
  return push_storage(reinterpret_cast<const uint8_t*>(&data), sizeof(T));
}
Range push_texture_data(const uint8_t* data, uint32_t bytesPerRow, uint32_t rowsPerImage);

template <typename State>
const State& get_state();
template <typename DrawData>
void push_draw_command(DrawData data);
template <typename DrawData>
DrawData* get_last_draw_command();

template <typename PipelineConfig>
PipelineRef pipeline_ref(const PipelineConfig& config);
bool bind_pipeline(PipelineRef ref, gl::PassEncoder& pass);

// The cached bind group is a POD gl::BindingSet (up to 8 texture+sampler pairs, or
// a uniform buffer bound with a dynamic offset). Callers build the set and hand it
// in; the cache keys on its content hash.
BindGroupRef bind_group_ref(const gl::BindingSet& bindingSet);
const gl::BindingSet& find_bind_group(BindGroupRef id);

gl::Sampler sampler_ref(const gl::SamplerDescriptor& descriptor);

// GL resource objects must be created with the render context current (the worker
// owns it), but callers (texture resolve, bind-group build) run on the recording
// thread. These marshal gl::create_texture/create_sampler to the worker and block
// for the result, so the returned handle's GL name is valid immediately. Cheap in
// steady state (creation happens on cache miss, not per frame).
gl::Texture create_gl_texture(gl::TextureFormat format, gl::Extent3D size, uint32_t mips, bool renderable);
gl::Sampler create_gl_sampler(const gl::SamplerDescriptor& descriptor);

uint32_t align_uniform(uint32_t value);

Vec2<uint32_t> get_render_target_size() noexcept;
void set_viewport(const Viewport& viewport) noexcept;
void set_scissor(const ClipRect& scissor) noexcept;

void push_debug_group(std::string label);
void insert_debug_marker(std::string label);
} // namespace aurora::gfx
