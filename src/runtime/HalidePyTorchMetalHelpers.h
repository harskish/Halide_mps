#ifndef HL_PYTORCH_METAL_HELPERS_H
#define HL_PYTORCH_METAL_HELPERS_H

/** \file
 * Override Halide's Metal hooks so that the Halide code called from PyTorch uses
 * the correct GPU device and stream. This header should be included once in
 * the PyTorch/C++ binding source file (see apps/HelloPyTorch/setup.py for an
 * example).
 */

#include "HalideRuntimeMetal.h"
#include "ATen/mps/MPSStream.h"
#include <iostream>

#include <dispatch/dispatch.h> // GCD

using at::mps::MPSStream;

namespace Halide {
namespace PyTorch {

typedef struct UserContext {
    UserContext(MPSStream *stream) : stream(stream){}; // Why is this constructor needed?
    MPSStream *stream;
} UserContext;

}  // namespace PyTorch
}  // namespace Halide

// Replace Halide weakly-linked Metal handles

namespace Halide {
namespace Runtime {
namespace Internal {
namespace Metal {

typedef halide_metal_device mtl_device;
typedef halide_metal_command_queue mtl_command_queue;

// For debugging symbol names:
// nm -gU .cache/torch_ops/test_fun_gpu/.../test_fun_gpu.so | grep get_default_mtl_device

void sync_stream(void* user_context) {
    MPSStream* stream = ((Halide::PyTorch::UserContext*)user_context)->stream;

    // End kernel coalescing => make sure no work is pending a commit
    //stream->synchronize(at::mps::SyncType::COMMIT);
    //stream->synchronize(at::mps::SyncType::COMMIT_AND_WAIT);
    //stream->synchronize(at::mps::SyncType::NONE);
    //stream->commit(true); // flush
    stream->commit(false); // don't flush
}

void metal_pre_run(void* user_context) {
    MPSStream* stream = ((Halide::PyTorch::UserContext*)user_context)->stream;
    dispatch_sync_f(stream->queue(), user_context, sync_stream);
}

mtl_device *get_default_mtl_device(void *user_context) {
    auto ctx = (Halide::PyTorch::UserContext*)user_context;
    return (mtl_device*)ctx->stream->device();
}

mtl_command_queue *new_command_queue(mtl_device *device, void *user_context) {
    MPSStream* stream = ((Halide::PyTorch::UserContext*)user_context)->stream;
    if ((void*)device != (void*)stream->device()) {
        std::cout << "ERROR: devices don't match" << std::endl;
    }

    return (mtl_command_queue*)stream->commandQueue();
}



// Use mpsAllocator?
// mtl_buffer *new_buffer(mtl_device *device, size_t length) {}

// Call appropriate?
// mtl_command_buffer *new_command_buffer(mtl_command_queue *queue, const char *label, size_t label_len) {}

} // namespace Metal
} // namespace Internal
} // namespace Runtime
} // namespace Halide

#endif /* end of include guard: HL_PYTORCH_METAL_HELPERS_H */
