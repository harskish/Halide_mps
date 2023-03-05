#ifndef HL_PYTORCH_CUDA_HELPERS_H
#define HL_PYTORCH_CUDA_HELPERS_H

/** \file
 * Override Halide's CUDA hooks so that the Halide code called from PyTorch uses
 * the correct GPU device and stream. This header should be included once in
 * the PyTorch/C++ binding source file (see apps/HelloPyTorch/setup.py for an
 * example).
 */

#include "HalideRuntimeCuda.h"
#include "cuda.h"
#include "cuda_runtime.h"
#include <iostream>

namespace Halide {
namespace PyTorch {

typedef struct UserContext {
    UserContext(int id, CUcontext *ctx, cudaStream_t *stream)
        : device_id(id), cuda_context(ctx), stream(stream){};

    int device_id;
    CUcontext *cuda_context;
    cudaStream_t *stream;
} UserContext;

}  // namespace PyTorch
}  // namespace Halide

// Replace Halide weakly-linked CUDA handles
extern "C" {

int halide_cuda_acquire_context_fun(void *user_context, CUcontext *ctx, bool create = true) {
    //std::cout << "---Calling overridden halide_cuda_acquire_context().\n";
    if (user_context != nullptr) {
        Halide::PyTorch::UserContext *user_ctx = (Halide::PyTorch::UserContext *)user_context;
        *ctx = *user_ctx->cuda_context;
    } else {
        *ctx = nullptr;
    }
    return halide_error_code_success;
}

int halide_cuda_get_stream_fun(void *user_context, CUcontext ctx, CUstream *stream) {
    //std::cout << "---Calling overridden halide_cuda_get_stream().\n";
    if (user_context != nullptr) {
        Halide::PyTorch::UserContext *user_ctx = (Halide::PyTorch::UserContext *)user_context;
        *stream = *user_ctx->stream;
    } else {
        *stream = 0;
    }
    return halide_error_code_success;
}

#ifdef _MSC_VER

// MSVC does not support weak linkage, must set overrides at runtime
void set_cuda_fun_overrides() {
    halide_set_cuda_acquire_context((halide_cuda_acquire_context_t)halide_cuda_acquire_context_fun);
    halide_set_cuda_get_stream((halide_cuda_get_stream_t)halide_cuda_get_stream_fun);
    //std::cout << "---TODO: need to override halide_get_gpu_device() as well!\n";
}

#else

void set_cuda_fun_overrides() {
    return; // no-op
}

int halide_cuda_acquire_context(void *user_context, CUcontext *ctx, bool create = true) {
    return halide_cuda_acquire_context_fun(user_context, ctx, create);
}

int halide_cuda_get_stream(void *user_context, CUcontext ctx, CUstream *stream) {
    return halide_cuda_get_stream_fun(user_context, ctx, stream);
}

int halide_get_gpu_device(void *user_context) {
    //std::cout << "---Calling overridden halide_get_gpu_device().\n";
    if (user_context != nullptr) {
        Halide::PyTorch::UserContext *user_ctx = (Halide::PyTorch::UserContext *)user_context;
        return user_ctx->device_id;
    } else {
        return 0;
    }
}

#endif // _MSC_VER

}  // extern "C"

#endif /* end of include guard: HL_PYTORCH_CUDA_HELPERS_H */
