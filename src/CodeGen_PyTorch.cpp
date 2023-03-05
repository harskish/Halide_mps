#include <iostream>

#include "CodeGen_C.h"
#include "CodeGen_PyTorch.h"
#include "IROperator.h"
#include "Module.h"
#include "Param.h"
#include "Util.h"
#include "Var.h"

namespace Halide {
namespace Internal {

CodeGen_PyTorch::CodeGen_PyTorch(std::ostream &s)
    : IRPrinter(s) {
}

void CodeGen_PyTorch::compile(const Module &module) {
    const Target target = module.target();

    if (target.has_feature(Target::CUDA)) {
        if (!target.has_feature(Target::UserContext)) {
            user_error << "Compile a PyTorch wrapper for a CUDA op requires the "
                          "UserContext feature to properly manage the GPU memory. "
                          "Please add \"-user_context\" to the generator's target options.\n";
        }
        stream << "#include \"ATen/cuda/CUDAContext.h\"\n";
    } else if (target.has_feature(Target::Metal)) {
        stream << "#include \"ATen/mps/MPSStream.h\"\n";
        stream << "#include <c10/core/Stream.h>\n";
        //stream << "#include \"ATen/mps/MPSLibrary.h\"\n"; // custom class in https://github.com/grimoire/TorchMPSCustomOpsDemo/
        //stream << "#include \"ATen/mps/MPSUtils.h\"\n"; // custom, provides setMTLArg
    }
    stream << "#include \"HalideBuffer.h\"\n";
    stream << "#include \"HalidePyTorchHelpers.h\"\n";
    stream << "#include <iostream>\n";

    stream << "\n";

    // Emit extern decls of the Halide-generated functions we use directly
    // into this file, so that we don't have to #include the relevant .h
    // file directly; this simplifies certain compile/build setups (since
    // we don't have to build files in tandem and/or get include paths right),
    // and should be totally safe, since we are using the same codegen logic
    // that would be in the .h file anyway.
    {
        CodeGen_C extern_decl_gen(stream, module.target(), CodeGen_C::CPlusPlusExternDecl);
        extern_decl_gen.compile(module);
    }

    for (const auto &f : module.functions()) {
        if (target.has_feature(Target::CUDA)) {
            std::cout << "Has CUDA\n";
            compile(f, true, false);
        } else if (target.has_feature(Target::Metal)) {
            std::cout << "Has Metal\n";
            compile(f, false, true);
        } else {
            std::cout << "Just CPU\n";
            compile(f, false, false);
        }
    }
}

void CodeGen_PyTorch::compile(const LoweredFunc &f, bool is_cuda, bool is_metal) {
    // Don't put non-external function declarations in headers.
    std::vector<std::string> namespaces;
    std::string simple_name = extract_namespaces(f.name, namespaces);

    if (!namespaces.empty()) {
        for (const auto &ns : namespaces) {
            stream << "namespace " << ns << " {\n";
        }
        stream << "\n";
    }
    const std::vector<LoweredArgument> &args = f.args;
    std::vector<LoweredArgument> buffer_args;

    stream << "HALIDE_FUNCTION_ATTRS\n";
    stream << "inline int " << simple_name << "_th_(";
    for (size_t i = 0; i < args.size(); i++) {
        if (args[i].name == "__user_context") {
            continue;
        } else if (args[i].is_buffer()) {
            buffer_args.push_back(args[i]);
            stream
                << "at::Tensor &"
                << c_print_name(args[i].name);
        } else {
            stream
                << type_to_c_type(args[i].type, true)
                << c_print_name(args[i].name);
        }

        if (i < args.size() - 1) {
            stream << ", ";
        }
    }

    stream << ") {\n";
    indent += 4;

    if (is_cuda) {
        stream << get_indent() << "// Setup CUDA\n";
        stream << get_indent() << "int device_id = at::cuda::current_device();\n";
        stream << get_indent() << "CUcontext ctx = 0;\n";
        stream << get_indent() << "CUresult res = cuCtxGetCurrent(&ctx);\n";
        stream << get_indent() << "AT_ASSERTM(res == 0, \"Could not acquire CUDA context\");\n";
        stream << get_indent() << "cudaStream_t stream = at::cuda::getCurrentCUDAStream(device_id);\n";
        stream << get_indent() << "struct UserContext { int device_id; CUcontext *cuda_context; cudaStream_t *stream; } user_ctx;\n";
        stream << get_indent() << "user_ctx.device_id = device_id;\n";
        stream << get_indent() << "user_ctx.cuda_context = &ctx;\n";
        stream << get_indent() << "user_ctx.stream = &stream;\n";
        stream << get_indent() << "void* __user_context = (void*) &user_ctx;\n\n";
    } else if (is_metal) {
        stream << get_indent() << "// Setup Metal\n";

        // Stream: created on Torch side
        // Creates internal command queue:
        // https://developer.apple.com/documentation/metal/mtldevice/1433388-newcommandqueue
        stream << get_indent() << "using at::mps::MPSStream;\n";
        stream << get_indent() << "MPSStream* stream = at::mps::getCurrentMPSStream();\n";
        stream << get_indent() << "std::cout << \"stream: \" << stream << std::endl;\n";

        // Get command queue that was created in stream constructor
        stream << get_indent() << "MTLCommandQueue_t cmdQueue = stream->commandQueue();\n";
        stream << get_indent() << "std::cout << \"cmdQueue: \" << cmdQueue << std::endl;\n";

        // Creates new command buffer from command queue
        // MTLCommandQueue.makeCommandBuffer() -> MTLCommandBuffer
        // => Should call ObjC code from <MPSStream.mm>, is that file being compiled?
        //    (or is the symbol found in torch itself?)
        stream << get_indent() << "MTLCommandBuffer_t commandBuff = stream->commandBuffer();\n"; // <MPSStream.h>
        stream << get_indent() << "std::cout << \"commandBuff: \" << commandBuff << std::endl;\n";
        
        // ObjC calls, not needed? (can call via msgSend on Halide side)
        //stream << get_indent() << "MTLComputeCommandEncoder_t compute_encoder = [commandBuff computeCommandEncoder];\n"; // creates new

        stream << get_indent() << "dispatch_queue_t dptQueue = stream->queue();\n";
        stream << get_indent() << "std::cout << \"dptQueue: \" << dptQueue << std::endl;\n";

        stream << get_indent() << "c10::DeviceIndex devIdx = stream->device_index();\n";
        stream << get_indent() << "std::cout << \"devIdx: \" << devIdx << std::endl;\n";
        
        stream << get_indent() << "MTLDevice_t device = stream->device();\n";
        stream << get_indent() << "std::cout << \"device: \" << device << std::endl;\n";

        stream << get_indent() << "c10::Stream streamRaw = stream->unwrap();\n";
        stream << get_indent() << "std::cout << \"streamRaw: \" << streamRaw << std::endl;\n";
        
        stream << get_indent() << "struct UserContext { c10::DeviceIndex device_id; MPSStream* stream; } user_ctx;\n";
        stream << get_indent() << "user_ctx.device_id = devIdx;\n";
        stream << get_indent() << "user_ctx.stream = stream;\n";
        stream << get_indent() << "void* __user_context = (void*) &user_ctx;\n\n";

        stream << get_indent() << "std::cout << \"user_ctx.stream: \" << user_ctx.stream << std::endl;\n";
    } else {
        stream << get_indent() << "void* __user_context = nullptr;\n\n";
    }

    stream << get_indent() << "// Check tensors have contiguous memory and are on the correct device\n";
    for (auto &buffer_arg : buffer_args) {
        stream << get_indent();
        stream
            << "HLPT_CHECK_CONTIGUOUS("
            << c_print_name(buffer_arg.name)
            << ");\n";

        if (is_cuda) {
            stream << get_indent();
            stream
                << "HLPT_CHECK_DEVICE("
                << c_print_name(buffer_arg.name)
                << ", device_id);\n";
        }
    }
    stream << "\n";

    stream << get_indent() << "// Wrap tensors in Halide buffers\n";
    for (auto &buffer_arg : buffer_args) {
        if (!buffer_arg.is_buffer()) {
            continue;
        }

        stream << get_indent();
        std::string tp = type_to_c_type(buffer_arg.type, false);
        stream
            << "Halide::Runtime::Buffer<" << tp << "> "
            << c_print_name(buffer_arg.name);
        if (is_cuda) {
            stream
                << "_buffer = Halide::PyTorch::wrap_cuda<" << tp << ">(";
        } else {
            stream
                << "_buffer = Halide::PyTorch::wrap<" << tp << ">(";
        }
        stream
            << c_print_name(buffer_arg.name)
            << ");\n";
    }
    stream << "\n";

    stream << get_indent() << "// Run Halide pipeline\n";

    stream << get_indent() << "int err = " << simple_name << "(";
    for (size_t i = 0; i < args.size(); i++) {
        if (args[i].is_buffer()) {
            stream
                << c_print_name(args[i].name)
                << "_buffer";
        } else {
            stream << c_print_name(args[i].name);
        }
        if (i < args.size() - 1) {
            stream << ", ";
        }
    }
    stream << ");\n";

    stream << "\n";

    stream << get_indent() << "AT_ASSERTM(err == 0, \"Halide call failed\");\n";

    if (is_cuda) {
        stream << get_indent() << "// Make sure data is on device\n";
        for (auto &buffer_arg : buffer_args) {
            if (buffer_arg.is_buffer()) {
                stream << get_indent();
                stream
                    << "AT_ASSERTM(!"
                    << c_print_name(buffer_arg.name) << "_buffer.host_dirty(),"
                    << "\"device not synchronized for buffer "
                    << c_print_name(buffer_arg.name)
                    << ", make sure all update stages are explicitly computed on GPU."
                    << "\");\n";
                stream << get_indent();
                stream
                    << c_print_name(buffer_arg.name) << "_buffer"
                    << ".device_detach_native();\n";
            }
        }
        stream << "\n";
    }

    // TODO(mgharbi): this is not very well documented
    if (get_env_variable("FLUSH_MEMOIZE_CACHE") == "1") {
        stream << get_indent() << "// Flush cache\n";
        if (is_cuda) {
            stream << get_indent() << "halide_memoization_cache_cleanup(__user_context);\n";
        } else {
            stream << get_indent() << "halide_memoization_cache_cleanup(nullptr);\n";
        }
    }

    stream << get_indent() << "return 0;\n";

    indent -= 4;
    stream << "}\n";

    if (!namespaces.empty()) {
        stream << "\n";
        for (size_t i = namespaces.size(); i > 0; i--) {
            stream << "}  // namespace " << namespaces[i - 1] << "\n";
        }
        stream << "\n";
    }
}

}  // namespace Internal
}  // namespace Halide
