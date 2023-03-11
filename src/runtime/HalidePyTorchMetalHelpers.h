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

mtl_device *get_default_mtl_device(void *user_context) {
    std::cout << "---Using overridden get_default_mtl_device" << std::endl;
    std::cout << "---Metal - Using user_context at: " << user_context << std::endl;
    auto ctx = (Halide::PyTorch::UserContext*)user_context;
    return (mtl_device*)ctx->stream->device();
}

mtl_command_queue *new_command_queue(mtl_device *device, void *user_context) {
    std::cout << "---Using overridden new_command_queue" << std::endl;
    auto ctx = (Halide::PyTorch::UserContext*)user_context;
    if ((void*)device != (void*)ctx->stream->device()) {
        std::cout << "ERROR: devices don't match" << std::endl;
    }
    return (mtl_command_queue*)ctx->stream->commandQueue();
}

} // namespace Metal
} // namespace Internal
} // namespace Runtime
} // namespace Halide

// Get device associated with stream
// typedef void* MTLDevice;
// stream << get_indent() << "void* device = stream->device();\n";
// stream << get_indent() << "std::cout << \"device: \" << device << std::endl;\n";

// Get command queue that was created in stream constructor
// typedef void* MTLCommandQueue_t;
// stream << get_indent() << "void* cmdQueue = stream->commandQueue();\n";
// stream << get_indent() << "std::cout << \"cmdQueue: \" << cmdQueue << std::endl;\n";

// Creates new command buffer from command queue
// MTLCommandQueue.makeCommandBuffer() -> MTLCommandBuffer
// => Should call ObjC code from <MPSStream.mm>, is that file being compiled?
//    (or is the symbol found in torch itself?)
// stream << get_indent() << "MTLCommandBuffer_t commandBuff = stream->commandBuffer();\n"; // <MPSStream.h>
// stream << get_indent() << "std::cout << \"commandBuff: \" << commandBuff << std::endl;\n";

// ObjC calls, not needed? (can call via msgSend on Halide side)
//stream << get_indent() << "MTLComputeCommandEncoder_t compute_encoder = [commandBuff computeCommandEncoder];\n"; // creates new

// stream << get_indent() << "dispatch_queue_t dptQueue = stream->queue();\n";
// stream << get_indent() << "std::cout << \"dptQueue: \" << dptQueue << std::endl;\n";

// stream << get_indent() << "c10::DeviceIndex devIdx = stream->device_index();\n";
// stream << get_indent() << "std::cout << \"devIdx: \" << devIdx << std::endl;\n";

// stream << get_indent() << "c10::Stream streamRaw = stream->unwrap();\n";
// stream << get_indent() << "std::cout << \"streamRaw: \" << streamRaw << std::endl;\n";

#endif /* end of include guard: HL_PYTORCH_METAL_HELPERS_H */
