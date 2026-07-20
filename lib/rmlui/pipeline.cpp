#include "pipeline.hpp"

#include "../internal.hpp"
#include "../webgpu/gpu.hpp"

#include "../gl/gl_core.hpp"
#include "../gl/program.hpp"
#include "../gl/state.hpp"

#include <algorithm>
#include <array>
#include <string>
#include <string_view>

#include <RmlUi/Core/Vertex.h>
#include <tracy/Tracy.hpp>

namespace aurora::rmlui {
namespace {
using namespace std::string_view_literals;

// WebGPU bind-group *layouts* have no GL equivalent (GL binds textures/uniform
// blocks by unit/index at draw time), so the three layout globals are gone. Only
// the shared filtering sampler survives.
gl::Sampler g_sampler;

constexpr uint32_t DynamicGroup1 = 1u << 1u;
constexpr uint32_t DynamicGroup2 = 1u << 2u;

// GL uniform-block binding points (S4). The common vertex/gamma block sits at 0
// (common_bind_group_ref); the per-effect block (gradient/blur/seed/filter/shadow) at
// 1 (uniform_bind_group_ref). GL_INVALID_INDEX (0xFFFFFFFF) means the program does not
// declare that block, which is fine -- the inert bind is ignored.
constexpr gl::GLuint kCommonBlockBinding = 0;
constexpr gl::GLuint kExtraBlockBinding = 1;
constexpr gl::GLuint kInvalidBlockIndex = 0xFFFFFFFFu;
// Texture units: the image sampler `t` is unit 0, the mask sampler `mask_t` unit 1.
constexpr gl::GLint kImageTextureUnit = 0;
constexpr gl::GLint kMaskTextureUnit = 1;

constexpr uint64_t CommonUniformBindingSize = AURORA_ALIGN(sizeof(UniformBlock), 16);
constexpr uint64_t ExtraUniformBindingSize =
    AURORA_ALIGN(std::max({sizeof(BlurUniformBlock), sizeof(DropShadowUniformBlock), sizeof(SimpleFilterUniformBlock),
                           sizeof(GradientUniformBlock), sizeof(SeedResampleUniformBlock)}),
                 16);

// ---------------------------------------------------------------------------
// GLSL (#version 300 es) ports of the RmlUi WGSL shaders. Translation rules
// (mirrors the GX emitter): textureSample(t,s,uv) -> texture(t,uv) (combined
// samplers, S3); textureLoad -> texelFetch; textureDimensions -> textureSize;
// @builtin(position) fragment input -> gl_FragCoord; @builtin(vertex_index) ->
// gl_VertexID; uniform blocks -> layout(std140) uniform Name {...} inst; (S4);
// atan2(y,x) -> atan(y,x). The `Uniforms` block is declared identically in the
// vertex and fragment stages (GLSL ES requires matching block declarations).
// No projection Y-flip here: the MVP flip lives in SetupRenderState (S1d).
// ---------------------------------------------------------------------------

// Geometry vertex shader (also used by the gradient pipeline). The `Uniforms` block is
// repeated verbatim in every stage that references it (GLSL ES requires the vertex and
// fragment declarations of a shared block to match exactly).

#ifdef AURORA_GLES2
constexpr std::string_view vertexSource = R"(#version 100
precision highp float;
uniform vec4 u_common[6];
attribute vec2 in_position;
attribute vec2 in_uv;
attribute vec4 in_color;
varying vec4 v_color;
varying vec2 v_uv;
void main() {
  mat4 mvp = mat4(u_common[0], u_common[1], u_common[2], u_common[3]);
  vec2 translated_pos = u_common[4].xy + in_position;
  gl_Position = mvp * vec4(translated_pos, 0.0, 1.0);
  v_color = in_color;
  v_uv = in_uv;
}
)"sv;

constexpr std::string_view fragmentSource = R"(#version 100
precision highp float;
uniform vec4 u_common[6];
uniform sampler2D t;
varying vec4 v_color;
varying vec2 v_uv;
void main() {
  vec4 color = v_color * texture2D(t, v_uv);
  float gamma_value = u_common[5].x;
  gl_FragColor = gamma_value == 1.0 ? color : vec4(pow(color.rgb, vec3(gamma_value)), color.a);
}
)"sv;

constexpr std::string_view gradientFragmentSource = R"(#version 100
precision highp float;
precision highp int;
uniform vec4 u_common[6];
uniform vec4 u_extra[22];
varying vec4 v_color;
varying vec2 v_uv;
const float PI = 3.14159265;

float stop_position(int i) {
  if (i == 0) return u_extra[18].x; if (i == 1) return u_extra[18].y;
  if (i == 2) return u_extra[18].z; if (i == 3) return u_extra[18].w;
  if (i == 4) return u_extra[19].x; if (i == 5) return u_extra[19].y;
  if (i == 6) return u_extra[19].z; if (i == 7) return u_extra[19].w;
  if (i == 8) return u_extra[20].x; if (i == 9) return u_extra[20].y;
  if (i == 10) return u_extra[20].z; if (i == 11) return u_extra[20].w;
  if (i == 12) return u_extra[21].x; if (i == 13) return u_extra[21].y;
  if (i == 14) return u_extra[21].z; return u_extra[21].w;
}
vec4 stop_color(int i) {
  if (i == 0) return u_extra[2]; if (i == 1) return u_extra[3];
  if (i == 2) return u_extra[4]; if (i == 3) return u_extra[5];
  if (i == 4) return u_extra[6]; if (i == 5) return u_extra[7];
  if (i == 6) return u_extra[8]; if (i == 7) return u_extra[9];
  if (i == 8) return u_extra[10]; if (i == 9) return u_extra[11];
  if (i == 10) return u_extra[12]; if (i == 11) return u_extra[13];
  if (i == 12) return u_extra[14]; if (i == 13) return u_extra[15];
  if (i == 14) return u_extra[16]; return u_extra[17];
}
vec4 stop_color_mix(float value, float num_stops) {
  vec4 color = stop_color(0);
  for (int i = 1; i < 16; ++i) {
    if (float(i) < num_stops) {
      color = mix(color, stop_color(i), smoothstep(stop_position(i - 1), stop_position(i), value));
    }
  }
  return color;
}
float bayer_dither(vec2 position) {
  return (fract(sin(dot(floor(position), vec2(12.9898, 78.233))) * 43758.5453) - 0.5) / 255.0;
}
void main() {
  float function_id = u_extra[0].x;
  float num_stops = u_extra[0].y;
  vec2 p = u_extra[0].zw;
  vec2 v = u_extra[1].xy;
  float value = 0.0;
  if (function_id == 0.0 || function_id == 3.0) {
    value = dot(v, v_uv - p) / dot(v, v);
  } else if (function_id == 1.0 || function_id == 4.0) {
    value = length(v * (v_uv - p));
  } else {
    vec2 d = v_uv - p;
    vec2 rotated = vec2(v.x * d.x + v.y * d.y, -v.y * d.x + v.x * d.y);
    value = 0.5 + atan(-rotated.x, rotated.y) / (2.0 * PI);
  }
  if (function_id >= 3.0) {
    float p0 = stop_position(0);
    float p1 = stop_position(int(num_stops) - 1);
    float span = p1 - p0;
    value = p0 + (value - p0) - span * floor((value - p0) / span);
  }
  vec4 color = v_color * stop_color_mix(value, num_stops);
  float gamma_value = u_common[5].x;
  if (gamma_value != 1.0) {
    color.rgb = clamp(pow(color.rgb, vec3(gamma_value)) + vec3(bayer_dither(gl_FragCoord.xy)), 0.0, 1.0);
  }
  gl_FragColor = color;
}
)"sv;

constexpr std::string_view fullscreenVertexSource = R"(#version 100
precision highp float;
attribute vec2 a_position;
attribute vec2 a_uv;
varying vec2 v_uv;
void main() {
  gl_Position = vec4(a_position, 0.0, 1.0);
  v_uv = vec2(a_uv.x, 1.0 - a_uv.y);
}
)"sv;

constexpr std::string_view blurVertexSource = R"(#version 100
precision highp float;
uniform vec4 u_extra[22];
attribute vec2 a_position;
attribute vec2 a_uv;
varying vec2 v_uv0; varying vec2 v_uv1; varying vec2 v_uv2; varying vec2 v_uv3;
varying vec2 v_uv4; varying vec2 v_uv5; varying vec2 v_uv6;
vec2 blur_uv(vec2 uv, float index) { return uv - (index - 3.0) * u_extra[0].xy; }
void main() {
  vec2 uv = vec2(a_uv.x, 1.0 - a_uv.y);
  gl_Position = vec4(a_position, 0.0, 1.0);
  v_uv0 = blur_uv(uv, 0.0); v_uv1 = blur_uv(uv, 1.0); v_uv2 = blur_uv(uv, 2.0);
  v_uv3 = blur_uv(uv, 3.0); v_uv4 = blur_uv(uv, 4.0); v_uv5 = blur_uv(uv, 5.0);
  v_uv6 = blur_uv(uv, 6.0);
}
)"sv;

constexpr std::string_view blitFragmentSource = R"(#version 100
precision highp float; uniform sampler2D t; varying vec2 v_uv;
void main() { gl_FragColor = texture2D(t, v_uv); }
)"sv;
constexpr std::string_view opaqueBlitFragmentSource = R"(#version 100
precision highp float; uniform sampler2D t; varying vec2 v_uv;
void main() { gl_FragColor = vec4(texture2D(t, v_uv).rgb, 1.0); }
)"sv;
constexpr std::string_view seedResampleFragmentSource = R"(#version 100
precision highp float; uniform sampler2D t; uniform vec4 u_extra[22]; varying vec2 v_uv;
void main() { gl_FragColor = vec4(texture2D(t, v_uv).rgb, 1.0); }
)"sv;
constexpr std::string_view simpleFilterFragmentSource = R"(#version 100
precision highp float; uniform sampler2D t; uniform vec4 u_extra[22]; varying vec2 v_uv;
void main() {
  vec4 c = texture2D(t, v_uv);
  mat4 matrix = mat4(u_extra[0], u_extra[1], u_extra[2], u_extra[3]);
  vec4 transformed = matrix * c;
  gl_FragColor = vec4(transformed.rgb, c.a) * u_extra[4].x;
}
)"sv;
constexpr std::string_view maskImageFragmentSource = R"(#version 100
precision highp float; uniform sampler2D t; uniform sampler2D mask_t; varying vec2 v_uv;
void main() { gl_FragColor = texture2D(t, v_uv) * texture2D(mask_t, v_uv).a; }
)"sv;
constexpr std::string_view blurFragmentSource = R"(#version 100
precision highp float; precision highp int;
uniform sampler2D t; uniform vec4 u_extra[22];
varying vec2 v_uv0; varying vec2 v_uv1; varying vec2 v_uv2; varying vec2 v_uv3;
varying vec2 v_uv4; varying vec2 v_uv5; varying vec2 v_uv6;
float weight(int i) { int a = i < 0 ? -i : i; if (a == 0) return u_extra[2].x; if (a == 1) return u_extra[2].y; if (a == 2) return u_extra[2].z; return u_extra[2].w; }
vec4 tap(vec2 uv, int i) {
  vec2 region = step(u_extra[1].xy, uv) * step(uv, u_extra[1].zw);
  return texture2D(t, uv) * weight(i) * region.x * region.y;
}
void main() {
  gl_FragColor = tap(v_uv0, -3) + tap(v_uv1, -2) + tap(v_uv2, -1) + tap(v_uv3, 0) +
                 tap(v_uv4, 1) + tap(v_uv5, 2) + tap(v_uv6, 3);
}
)"sv;
constexpr std::string_view regionBlitFragmentSource = R"(#version 100
precision highp float; uniform sampler2D t; uniform vec4 u_extra[22]; varying vec2 v_uv;
void main() { gl_FragColor = texture2D(t, mix(u_extra[1].xy, u_extra[1].zw, v_uv)); }
)"sv;
constexpr std::string_view dropShadowFragmentSource = R"(#version 100
precision highp float; uniform sampler2D t; uniform vec4 u_extra[22]; varying vec2 v_uv;
void main() {
  vec2 uv = v_uv - u_extra[1].xy;
  vec2 region = step(u_extra[1].zw, uv) * step(uv, u_extra[2].xy);
  gl_FragColor = u_extra[0] * texture2D(t, uv).a * region.x * region.y;
}
)"sv;
#else
constexpr std::string_view vertexSource = R"(#version 300 es
layout(std140) uniform Uniforms {
  mat4 mvp;
  vec4 translation;
  float gamma;
} uniforms;
layout(location = 0) in vec2 in_position;
layout(location = 1) in vec2 in_uv;
layout(location = 2) in vec4 in_color;
out vec4 v_color;
out vec2 v_uv;
void main() {
  vec2 translatedPos = uniforms.translation.xy + in_position;
  gl_Position = uniforms.mvp * vec4(translatedPos, 0.0, 1.0);
  v_color = in_color;
  v_uv = in_uv;
}
)"sv;

constexpr std::string_view fragmentSource = R"(#version 300 es
precision highp float;
layout(std140) uniform Uniforms {
  mat4 mvp;
  vec4 translation;
  float gamma;
} uniforms;
uniform sampler2D t;
in vec4 v_color;
in vec2 v_uv;
out vec4 out_color;
void main() {
  vec4 color = v_color * texture(t, v_uv);
  if (uniforms.gamma == 1.0) {
    out_color = color;
  } else {
    vec3 corrected = pow(color.rgb, vec3(uniforms.gamma));
    out_color = vec4(corrected, color.a);
  }
}
)"sv;

constexpr std::string_view gradientFragmentSource = R"(#version 300 es
precision highp float;
precision highp int;
layout(std140) uniform Uniforms {
  mat4 mvp;
  vec4 translation;
  float gamma;
} uniforms;
layout(std140) uniform GradientUniforms {
  int function;
  int num_stops;
  vec2 p;
  vec2 v;
  vec2 padding;
  vec4 stop_colors[16];
  vec4 stop_positions[4];
} gradient;
in vec4 v_color;
in vec2 v_uv;
out vec4 out_color;

const int LINEAR = 0;
const int RADIAL = 1;
const int CONIC = 2;
const int REPEATING_LINEAR = 3;
const int REPEATING_RADIAL = 4;
const int REPEATING_CONIC = 5;
const float PI = 3.14159265;

float bayer_dither(vec2 position) {
  int bayer[64] = int[64](
    0, 32, 8, 40, 2, 34, 10, 42,
    48, 16, 56, 24, 50, 18, 58, 26,
    12, 44, 4, 36, 14, 46, 6, 38,
    60, 28, 52, 20, 62, 30, 54, 22,
    3, 35, 11, 43, 1, 33, 9, 41,
    51, 19, 59, 27, 49, 17, 57, 25,
    15, 47, 7, 39, 13, 45, 5, 37,
    63, 31, 55, 23, 61, 29, 53, 21);
  int x = int(position.x) % 8;
  int y = int(position.y) % 8;
  return (float(bayer[x + y * 8]) / 64.0 - 0.5) / 255.0;
}

float stop_position(int index) {
  int group_index = index / 4;
  int component_index = index % 4;
  return gradient.stop_positions[group_index][component_index];
}

vec4 stop_color_mix(float t) {
  vec4 color = gradient.stop_colors[0];
  for (int i = 1; i < 16; i++) {
    if (i < gradient.num_stops) {
      color = mix(color, gradient.stop_colors[i], smoothstep(stop_position(i - 1), stop_position(i), t));
    }
  }
  return color;
}

void main() {
  float t = 0.0;
  if (gradient.function == LINEAR || gradient.function == REPEATING_LINEAR) {
    float dist_square = dot(gradient.v, gradient.v);
    vec2 vv = v_uv - gradient.p;
    t = dot(gradient.v, vv) / dist_square;
  } else if (gradient.function == RADIAL || gradient.function == REPEATING_RADIAL) {
    vec2 vv = v_uv - gradient.p;
    t = length(gradient.v * vv);
  } else if (gradient.function == CONIC || gradient.function == REPEATING_CONIC) {
    vec2 vv = v_uv - gradient.p;
    vec2 rotated = vec2(gradient.v.x * vv.x + gradient.v.y * vv.y,
                        -gradient.v.y * vv.x + gradient.v.x * vv.y);
    t = 0.5 + atan(-rotated.x, rotated.y) / (2.0 * PI);
  }
  if (gradient.function == REPEATING_LINEAR ||
      gradient.function == REPEATING_RADIAL ||
      gradient.function == REPEATING_CONIC) {
    float t0 = stop_position(0);
    float t1 = stop_position(gradient.num_stops - 1);
    float span = t1 - t0;
    t = t0 + (t - t0) - span * floor((t - t0) / span);
  }
  vec4 color = v_color * stop_color_mix(t);
  if (uniforms.gamma == 1.0) {
    out_color = color;
  } else {
    vec3 corrected = pow(color.rgb, vec3(uniforms.gamma));
    vec3 dithered = clamp(corrected + vec3(bayer_dither(gl_FragCoord.xy)), vec3(0.0), vec3(1.0));
    out_color = vec4(dithered, color.a);
  }
}
)"sv;

// Fullscreen-triangle vertex shader (gl_VertexID; no vertex buffer). Orientation-preserving in GL:
// v=0 at NDC.y=-1 (dest row 0), so a fullscreen copy maps source row R -> dest row R. The whole GL
// pipeline is bottom-left origin (GX scene target, every RmlUi render target, the window), so
// preserving here keeps the entire chain -- scene seed, layer copies, blur/filter passes and the
// final present composite -- consistently GL-native with no explicit V-flip anywhere.
constexpr std::string_view fullscreenVertexSource = R"(#version 300 es
out vec2 v_uv;
void main() {
  const vec2 positions[3] = vec2[3](vec2(-1.0, 1.0), vec2(-1.0, -3.0), vec2(3.0, 1.0));
  const vec2 uvs[3] = vec2[3](vec2(0.0, 1.0), vec2(0.0, -1.0), vec2(2.0, 1.0));
  gl_Position = vec4(positions[gl_VertexID], 0.0, 1.0);
  v_uv = uvs[gl_VertexID];
}
)"sv;

// Blur/region fullscreen vertex shader: pre-computes seven tap UVs from the texel offset.
constexpr std::string_view blurVertexSource = R"(#version 300 es
layout(std140) uniform BlurUniforms {
  vec2 texel_offset;
  float radius;
  float padding;
  vec2 tex_coord_min;
  vec2 tex_coord_max;
  vec4 weights;
} blur;
out vec2 v_uv0;
out vec2 v_uv1;
out vec2 v_uv2;
out vec2 v_uv3;
out vec2 v_uv4;
out vec2 v_uv5;
out vec2 v_uv6;
const int BLUR_NUM_WEIGHTS = 4;
vec2 blur_uv(vec2 uv, int index) {
  return uv - float(index - BLUR_NUM_WEIGHTS + 1) * blur.texel_offset;
}
void main() {
  const vec2 positions[3] = vec2[3](vec2(-1.0, 1.0), vec2(-1.0, -3.0), vec2(3.0, 1.0));
  const vec2 uvs[3] = vec2[3](vec2(0.0, 1.0), vec2(0.0, -1.0), vec2(2.0, 1.0));
  vec2 uv = uvs[gl_VertexID];
  gl_Position = vec4(positions[gl_VertexID], 0.0, 1.0);
  v_uv0 = blur_uv(uv, 0);
  v_uv1 = blur_uv(uv, 1);
  v_uv2 = blur_uv(uv, 2);
  v_uv3 = blur_uv(uv, 3);
  v_uv4 = blur_uv(uv, 4);
  v_uv5 = blur_uv(uv, 5);
  v_uv6 = blur_uv(uv, 6);
}
)"sv;

constexpr std::string_view blitFragmentSource = R"(#version 300 es
precision highp float;
uniform sampler2D t;
in vec2 v_uv;
out vec4 out_color;
void main() { out_color = texture(t, v_uv); }
)"sv;

constexpr std::string_view opaqueBlitFragmentSource = R"(#version 300 es
precision highp float;
uniform sampler2D t;
in vec2 v_uv;
out vec4 out_color;
void main() { out_color = vec4(texture(t, v_uv).rgb, 1.0); }
)"sv;

constexpr std::string_view seedResampleFragmentSource = R"(#version 300 es
precision highp float;
precision highp int;
uniform sampler2D t;
layout(std140) uniform SeedUniforms {
  uint sampler_mode;
  float frame_width;
  float frame_height;
  uint pad;
} seed;
in vec2 v_uv;
out vec4 out_color;

vec4 sample_by_pixel(ivec2 pixel) {
  ivec2 source_dims = textureSize(t, 0);
  ivec2 max_coord = source_dims - ivec2(1, 1);
  ivec2 coord = clamp(pixel, ivec2(0, 0), max_coord);
  return texelFetch(t, coord, 0);
}

vec4 sample_area(vec2 frag_position) {
  vec2 source_size = vec2(textureSize(t, 0));
  vec2 target_size = max(vec2(seed.frame_width, seed.frame_height), vec2(1.0, 1.0));
  vec2 source_min = clamp((frag_position - vec2(0.5, 0.5)) / target_size, vec2(0.0), vec2(1.0)) * source_size;
  vec2 source_max = clamp((frag_position + vec2(0.5, 0.5)) / target_size, vec2(0.0), vec2(1.0)) * source_size;
  ivec2 first_pixel = ivec2(floor(source_min));
  ivec2 last_pixel = ivec2(ceil(source_max));
  const int max_iterations = 16;
  vec4 avg_color = vec4(0.0);
  float total_weight = 0.0;
  for (int iy = 0; iy < max_iterations; iy++) {
    int source_y = first_pixel.y + iy;
    if (source_y < last_pixel.y) {
      float y0 = float(source_y);
      float weight_y = max(min(source_max.y, y0 + 1.0) - max(source_min.y, y0), 0.0);
      for (int ix = 0; ix < max_iterations; ix++) {
        int source_x = first_pixel.x + ix;
        if (source_x < last_pixel.x) {
          float x0 = float(source_x);
          float weight_x = max(min(source_max.x, x0 + 1.0) - max(source_min.x, x0), 0.0);
          float weight = weight_x * weight_y;
          avg_color += weight * sample_by_pixel(ivec2(source_x, source_y));
          total_weight += weight;
        }
      }
    }
  }
  return avg_color / max(total_weight, 0.000001);
}

void main() {
  // Scene, RmlUi targets and window are all GL-native (bottom-left origin), so this seed just
  // preserves orientation like every other fullscreen pass -- no V-flip needed.
  vec4 color = texture(t, v_uv);
  if (seed.sampler_mode == 1u) {
    color = sample_area(gl_FragCoord.xy);
  }
  out_color = vec4(color.rgb, 1.0);
}
)"sv;

constexpr std::string_view simpleFilterFragmentSource = R"(#version 300 es
precision highp float;
uniform sampler2D t;
layout(std140) uniform SimpleFilterUniforms {
  mat4 matrix;
  vec4 opacity;
} simple_filter;
in vec2 v_uv;
out vec4 out_color;
void main() {
  vec4 tex_color = texture(t, v_uv);
  vec4 transformed = simple_filter.matrix * tex_color;
  out_color = vec4(transformed.rgb, tex_color.a) * simple_filter.opacity.x;
}
)"sv;

constexpr std::string_view maskImageFragmentSource = R"(#version 300 es
precision highp float;
uniform sampler2D t;
uniform sampler2D mask_t;
in vec2 v_uv;
out vec4 out_color;
void main() {
  vec4 tex_color = texture(t, v_uv);
  float mask_alpha = texture(mask_t, v_uv).a;
  out_color = tex_color * mask_alpha;
}
)"sv;

constexpr std::string_view blurFragmentSource = R"(#version 300 es
precision highp float;
precision highp int;
uniform sampler2D t;
layout(std140) uniform BlurUniforms {
  vec2 texel_offset;
  float radius;
  float padding;
  vec2 tex_coord_min;
  vec2 tex_coord_max;
  vec4 weights;
} blur;
in vec2 v_uv0;
in vec2 v_uv1;
in vec2 v_uv2;
in vec2 v_uv3;
in vec2 v_uv4;
in vec2 v_uv5;
in vec2 v_uv6;
out vec4 out_color;
float get_weight(int index) { return blur.weights[abs(index)]; }
vec4 sample_blur(vec2 sample_uv, int offset_index) {
  vec2 in_region = step(blur.tex_coord_min, sample_uv) * step(sample_uv, blur.tex_coord_max);
  return texture(t, sample_uv) * get_weight(offset_index) * in_region.x * in_region.y;
}
void main() {
  vec4 color = sample_blur(v_uv0, -3);
  color += sample_blur(v_uv1, -2);
  color += sample_blur(v_uv2, -1);
  color += sample_blur(v_uv3, 0);
  color += sample_blur(v_uv4, 1);
  color += sample_blur(v_uv5, 2);
  color += sample_blur(v_uv6, 3);
  out_color = color;
}
)"sv;

constexpr std::string_view regionBlitFragmentSource = R"(#version 300 es
precision highp float;
uniform sampler2D t;
layout(std140) uniform BlurUniforms {
  vec2 texel_offset;
  float radius;
  float padding;
  vec2 tex_coord_min;
  vec2 tex_coord_max;
  vec4 weights;
} blur;
in vec2 v_uv;
out vec4 out_color;
void main() {
  vec2 sample_uv = mix(blur.tex_coord_min, blur.tex_coord_max, v_uv);
  out_color = texture(t, sample_uv);
}
)"sv;

constexpr std::string_view dropShadowFragmentSource = R"(#version 300 es
precision highp float;
uniform sampler2D t;
layout(std140) uniform DropShadowUniforms {
  vec4 color;
  vec2 uv_offset;
  vec2 tex_coord_min;
  vec2 tex_coord_max;
} shadow;
in vec2 v_uv;
out vec4 out_color;
void main() {
  vec2 sample_uv = v_uv - shadow.uv_offset;
  vec2 in_region = step(shadow.tex_coord_min, sample_uv) * step(sample_uv, shadow.tex_coord_max);
  float alpha = texture(t, sample_uv).a * in_region.x * in_region.y;
  out_color = shadow.color * alpha;
}
)"sv;
#endif

std::string_view fragment_source(PipelineKind kind) {
  switch (kind) {
  case PipelineKind::Geometry:
    return fragmentSource;
  case PipelineKind::Gradient:
    return gradientFragmentSource;
  case PipelineKind::OpaqueBlit:
    return opaqueBlitFragmentSource;
  case PipelineKind::SeedResample:
    return seedResampleFragmentSource;
  case PipelineKind::Blur:
    return blurFragmentSource;
  case PipelineKind::RegionBlit:
    return regionBlitFragmentSource;
  case PipelineKind::DropShadow:
    return dropShadowFragmentSource;
  case PipelineKind::SimpleFilter:
    return simpleFilterFragmentSource;
  case PipelineKind::MaskImage:
    return maskImageFragmentSource;
  case PipelineKind::Blit:
  default:
    return blitFragmentSource;
  }
}

std::string_view vertex_source(VertexLayoutKind kind) {
  switch (kind) {
  case VertexLayoutKind::Geometry:
    return vertexSource;
  case VertexLayoutKind::BlurFullscreen:
    return blurVertexSource;
  case VertexLayoutKind::Fullscreen:
  default:
    return fullscreenVertexSource;
  }
}

// Compile + link the GLSL pair and wire fixed bindings: the `Uniforms` block -> point 0,
// any per-effect block -> point 1, the `t`/`mask_t` samplers -> units 0/1. Runs on a
// GL-context thread (the pipeline compiler share context or the render worker), so it
// issues GL directly (no marshal), matching gx::create_pipeline and tex_copy_conv.
gl::GLuint compile_rml_program(VertexLayoutKind vertexLayout, PipelineKind kind, const char* label) {
  const std::string vs{vertex_source(vertexLayout)};
  const std::string fs{fragment_source(kind)};
  const gl::GLuint program = gl::compile_program(vs.c_str(), fs.c_str(), label);
  if (program == 0) {
    return 0;
  }
#ifdef AURORA_GLES2
  gl::register_uniform_binding(program, kCommonBlockBinding, "u_common", static_cast<uint32_t>(CommonUniformBindingSize));
  gl::register_uniform_binding(program, kExtraBlockBinding, "u_extra", static_cast<uint32_t>(ExtraUniformBindingSize));
#else
  const gl::GLuint commonIndex = gl::gl.GetUniformBlockIndex(program, "Uniforms");
  if (commonIndex != kInvalidBlockIndex) {
    gl::gl.UniformBlockBinding(program, commonIndex, kCommonBlockBinding);
  }
  for (const char* name :
       {"GradientUniforms", "BlurUniforms", "SeedUniforms", "SimpleFilterUniforms", "DropShadowUniforms"}) {
    const gl::GLuint index = gl::gl.GetUniformBlockIndex(program, name);
    if (index != kInvalidBlockIndex) {
      gl::gl.UniformBlockBinding(program, index, kExtraBlockBinding);
    }
  }
#endif
  gl::gl.UseProgram(program);
  const gl::GLint imageLoc = gl::gl.GetUniformLocation(program, "t");
  if (imageLoc >= 0) {
    gl::gl.Uniform1i(imageLoc, kImageTextureUnit);
  }
  const gl::GLint maskLoc = gl::gl.GetUniformLocation(program, "mask_t");
  if (maskLoc >= 0) {
    gl::gl.Uniform1i(maskLoc, kMaskTextureUnit);
  }
  gl::gl.UseProgram(0);
  // A program linked on the compiler share context must be flushed to be visible on the
  // render worker's context.
  gl::gl.Flush();
  return program;
}

} // namespace

void initialize_pipeline() {
  // The three WebGPU bind-group layouts are gone (no GL equivalent). Only the shared
  // filtering sampler remains; gfx::sampler_ref is the GL-native replacement for the
  // old g_device.CreateSampler.
  constexpr gl::SamplerDescriptor samplerDesc{
      .addressU = gl::AddressMode::Repeat,
      .addressV = gl::AddressMode::Repeat,
      .addressW = gl::AddressMode::Repeat,
      .magFilter = gl::FilterMode::Linear,
      .minFilter = gl::FilterMode::Linear,
      .mipmapFilter =
#ifdef AURORA_GLES2
          gl::MipmapFilterMode::Undefined,
#else
          gl::MipmapFilterMode::Linear,
#endif
      .maxAnisotropy = 1,
  };
  g_sampler = gfx::sampler_ref(samplerDesc);
}

void shutdown_pipeline() {
  g_sampler = {};
}

gfx::BindGroupRef texture_bind_group_ref(const gl::Texture& view) {
  gl::BindingSet set{};
  set.textures[kImageTextureUnit].texture = view.id;
  // The common filtering sampler pairs with the image texture on GL (WebGPU kept the
  // sampler in group 0; GL binds texture+sampler together at the unit).
  set.textures[kImageTextureUnit].sampler = g_sampler.id;
  return gfx::bind_group_ref(set);
}

gfx::BindGroupRef mask_bind_group_ref(const gl::Texture& imageView, const gl::Texture& maskView) {
  // maskImage samples two textures in one draw: `t` at unit 0 and `mask_t` at unit 1.
  // A single bind group carries both so the second SetBindGroup does not clobber unit 0.
  gl::BindingSet set{};
  set.textures[kImageTextureUnit].texture = imageView.id;
  set.textures[kImageTextureUnit].sampler = g_sampler.id;
  set.textures[kMaskTextureUnit].texture = maskView.id;
  set.textures[kMaskTextureUnit].sampler = g_sampler.id;
  return gfx::bind_group_ref(set);
}

gfx::BindGroupRef common_bind_group_ref() {
  gl::BindingSet set{};
  set.buffers[0] = gl::BindingSet::BufferBinding{
      .buffer = gfx::g_uniformBuffer.id,
      .binding = kCommonBlockBinding,
      .offset = 0,
      .size = static_cast<uint32_t>(CommonUniformBindingSize),
      .dynamic = true,
  };
  set.bufferCount = 1;
  return gfx::bind_group_ref(set);
}

gfx::BindGroupRef uniform_bind_group_ref() {
  gl::BindingSet set{};
  set.buffers[0] = gl::BindingSet::BufferBinding{
      .buffer = gfx::g_uniformBuffer.id,
      .binding = kExtraBlockBinding,
      .offset = 0,
      .size = static_cast<uint32_t>(ExtraUniformBindingSize),
      .dynamic = true,
  };
  set.bufferCount = 1;
  return gfx::bind_group_ref(set);
}

gl::Pipeline create_pipeline(const PipelineConfig& config) {
  ZoneScoped;
  const auto kind = static_cast<PipelineKind>(config.kind);
  const auto vertexLayoutKind = static_cast<VertexLayoutKind>(config.vertexLayout);
  const auto stencilMode = static_cast<StencilMode>(config.stencilMode);
  const auto blendMode = static_cast<BlendMode>(config.blendMode);

  const std::string label = fmt::format("RmlUi Pipeline kind {} vtx {}", config.kind, config.vertexLayout);
  const gl::GLuint program = compile_rml_program(vertexLayoutKind, kind, label.c_str());

  // Bake the fixed-function state from the backend-agnostic config. Blend and stencil
  // config survive the cutover; the colour/stencil formats and MSAA sample count are
  // resolved by the render-target/FBO setup, not the pipeline, on GL.
  gl::BakedState state{};
  if (blendMode == BlendMode::Premultiplied) {
    state.blendEnabled = true;
    state.colorOp = gl::BlendOperation::Add;
    state.colorSrc = gl::BlendFactor::One;
    state.colorDst = gl::BlendFactor::OneMinusSrcAlpha;
    state.alphaOp = gl::BlendOperation::Add;
    state.alphaSrc = gl::BlendFactor::One;
    state.alphaDst = gl::BlendFactor::OneMinusSrcAlpha;
  }
  state.writeMask = static_cast<gl::ColorWriteMask>(config.colorWriteMask);
  state.topology = gl::PrimitiveTopology::TriangleList;
  state.frontFace = gl::FrontFace::CW;
  state.cull = gl::CullMode::None;
  // Combined depth-stencil format: the depth aspect is present but unused, so depth
  // writes are off and the depth test always passes (matches the clip-mask passes).
  state.depthTest = false;
  state.depthWrite = false;
  state.depthCompare = gl::CompareFunction::Always;

  if (stencilMode != StencilMode::None) {
    state.stencilTest = true;
    // failOp / depthFailOp are always Keep; only the compare and passOp vary by mode,
    // mirroring the WebGPU StencilFaceState the Dawn path used.
    state.stencilFail = gl::GL_KEEP;
    state.stencilDepthFail = gl::GL_KEEP;
    switch (stencilMode) {
    case StencilMode::EqualKeep:
      state.stencilCompare = gl::CompareFunction::Equal;
      state.stencilPass = gl::GL_KEEP;
      break;
    case StencilMode::ClipIntersect:
      state.stencilCompare = gl::CompareFunction::Equal;
      state.stencilPass = gl::GL_INCR; // IncrementClamp
      break;
    case StencilMode::ClipReplace:
      state.stencilCompare = gl::CompareFunction::Always;
      state.stencilPass = gl::GL_REPLACE;
      break;
    case StencilMode::AlwaysKeep:
    case StencilMode::None:
    default:
      state.stencilCompare = gl::CompareFunction::Always;
      state.stencilPass = gl::GL_KEEP;
      break;
    }
    state.stencilReadMask = 0xFF;
    state.stencilWriteMask = 0xFF;
  }

  gl::Pipeline pipeline{};
  pipeline.program = program;
  pipeline.state = state;
  // Geometry draws feed an interleaved Rml::Vertex buffer (its own VAO); fullscreen and
  // blur pipelines are attribute-less (gl_VertexID), so they use vertexLayout 0.
  pipeline.vertexLayout =
      vertexLayoutKind == VertexLayoutKind::Geometry
          ? gl::kRmlGeometryVertexLayout
          :
#ifdef AURORA_GLES2
          gl::kFullscreenVertexLayout;
#else
          0u;
#endif
  return pipeline;
}

void render(const DrawData& data, gl::PassEncoder& pass) {
  // bind_pipeline early-outs on a missing/empty (program 0) pipeline.
  if (!gfx::bind_pipeline(data.pipeline, pass)) {
    return;
  }

  const auto& commonBindGroup = gfx::find_bind_group(common_bind_group_ref());
  const std::array commonOffsets{data.uniformRange.offset};
  pass.SetBindGroup(0, commonBindGroup, commonOffsets.size(), commonOffsets.data());

  if (data.bindGroup1 != 0) {
    const auto& bindGroup = gfx::find_bind_group(data.bindGroup1);
    if ((data.dynamicBindGroupMask & DynamicGroup1) != 0) {
      const std::array offsets{data.bindGroup1DynamicOffset};
      pass.SetBindGroup(1, bindGroup, offsets.size(), offsets.data());
    } else {
      pass.SetBindGroup(1, bindGroup);
    }
  }

  if (data.bindGroup2 != 0) {
    const auto& bindGroup = gfx::find_bind_group(data.bindGroup2);
    if ((data.dynamicBindGroupMask & DynamicGroup2) != 0) {
      const std::array offsets{data.bindGroup2DynamicOffset};
      pass.SetBindGroup(2, bindGroup, offsets.size(), offsets.data());
    } else {
      pass.SetBindGroup(2, bindGroup);
    }
  }

  if (data.hasBlendConstant != 0) {
    const gl::Color color{data.blendConstant[0], data.blendConstant[1], data.blendConstant[2], data.blendConstant[3]};
    pass.SetBlendConstant(&color);
  }
  pass.SetStencilReference(data.stencilRef);

  if (static_cast<DrawKind>(data.drawKind) == DrawKind::Geometry) {
    pass.SetVertexBuffer(0, gfx::g_vertexBuffer, data.vertexRange.offset, data.vertexRange.size);
    pass.SetIndexBuffer(gfx::g_indexBuffer, gl::IndexFormat::Uint32, data.indexRange.offset, data.indexRange.size);
    pass.DrawIndexed(data.indexCount);
  } else {
    pass.Draw(data.vertexCount);
  }
}

uint32_t sampler_mode() noexcept {
  switch (webgpu::get_resampler()) {
  case SAMPLER_AREA:
    return 1;
  case SAMPLER_BILINEAR:
  default:
    return 0;
  }
}

} // namespace aurora::rmlui
