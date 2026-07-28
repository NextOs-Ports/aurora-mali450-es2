#pragma once

#include "common.hpp"

#include <functional>
#include <optional>

namespace aurora::gfx::clear {
struct PipelineConfig;
} // namespace aurora::gfx::clear

namespace aurora::gx {
struct PipelineConfig;
} // namespace aurora::gx

namespace aurora::rmlui {
struct PipelineConfig;
} // namespace aurora::rmlui

namespace aurora::gfx {

using NewPipelineCallback = std::function<gl::Pipeline()>;

void initialize_pipeline_cache();
void shutdown_pipeline_cache();
void begin_pipeline_frame();
void end_pipeline_frame();

template <typename Config>
PipelineRef find_pipeline(ShaderType type, const Config& config, NewPipelineCallback&& cb);

// Allocation-free lookup for the hot per-draw path: returns the ref when the pipeline is
// already cached and needs no bookkeeping, nullopt otherwise (caller falls back to
// find_pipeline, which may build/queue the pipeline).
template <typename Config>
std::optional<PipelineRef> find_pipeline_cached(ShaderType type, const Config& config);

bool get_pipeline(PipelineRef ref, gl::Pipeline& pipeline);

} // namespace aurora::gfx
