"""Synthesizes the cpp wrapper code and builds dynamic Python extension."""
import os
import platform
import re
from setuptools import setup, find_packages
from pathlib import Path

from torch.utils.cpp_extension import BuildExtension
import torch as th


def generate_pybind_wrapper(path, headers, has_cuda):
    s = "#include \"torch/extension.h\"\n\n"
    if has_cuda:
        s += "#include \"HalidePyTorchCudaHelpers.h\"\n"
    s += "#include \"HalidePyTorchHelpers.h\"\n"
    for h in headers:
        s += "#include \"{}\"\n".format(os.path.splitext(h)[0]+".pytorch.h")

    s += "\nPYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {\n"
    for h in headers:
        name = os.path.splitext(h)[0]
        s += "  m.def(\"{}\", &{}_th_, \"PyTorch wrapper of the Halide pipeline {}\");\n".format(
          name, name, name)
    s += "}\n"
    with open(path, 'w') as fid:
        fid.write(s)

# Remove forward declaration of scheduled pipeline
# (which is called internally by lib)
def _fix_scheduled_lib_header(pipeline_name: str, header: Path, remove_cuda_dirty_check=False):
    src_in = header.read_text().splitlines()
    src_out = []

    FIXED_FLAG = '//HL_HEADER_FIXED'
    if FIXED_FLAG in src_in:
        return
    
    inside_broken = False
    n_lines = len(src_in)
    for i in range(n_lines):
        curr = src_in[i]
        next = src_in[i + 1] if i < n_lines - 1 else ''
        
        enter_func = 'HALIDE_FUNCTION_ATTRS' in curr and 'inline int' in next
        enter_correct = enter_func and (f'{pipeline_name}_th_' in next)
        enter_broken = enter_func and not enter_correct
        exit_broken = curr.rstrip() == '}' and inside_broken

        if enter_broken:
            inside_broken = True
            src_out.append('/*')
        
        if remove_cuda_dirty_check and 'host_dirty()' in curr:
            print('Removing CUDA result tensor host_dirty() check')
        else:
            src_out.append(curr)

        if exit_broken:
            inside_broken = False
            src_out.append('*/')
    
    src_out.append(FIXED_FLAG)
    header.write_text('\n'.join(src_out))

if __name__ == "__main__":
    # When debugging:
    # os.environ['BIN'] = 'out/x86-64-linux-avx-avx2-f16c-fma-sse41'; os.environ['HALIDE_DISTRIB_PATH'] = '../../distrib'

    # This is where the generate Halide ops headers live. We also generate the .cpp
    # wrapper in this directory
    build_dir = os.getenv("BIN")
    print('build_dir', build_dir)
    if build_dir is None or not os.path.exists(build_dir):
        raise ValueError("Bin directory {} is invalid".format(build_dir))

    # Path to a distribution of Halide
    halide_dir = os.getenv("HALIDE_DISTRIB_PATH")
    print('halide_dir', halide_dir)
    if halide_dir is None or not os.path.exists(halide_dir):
        raise ValueError("Halide directory {} is invalid".format(halide_dir))

    has_cuda = os.getenv("HAS_CUDA")
    if has_cuda is None or has_cuda == "0":
        has_cuda = False
    else:
        has_cuda = True

    include_dirs = [build_dir, os.path.join(halide_dir, "include")]
    # Note that recent versions of PyTorch (at least 1.7.1) requires C++14
    # in order to compile extensions
    compile_args = ["-std=c++17", "-g"]
    if platform.system() == "Darwin":  # on osx libstdc++ causes trouble
        compile_args += ["-stdlib=libc++"]

    re_cc = re.compile(r".*\.pytorch\.h")
    hl_srcs = [f for f in os.listdir(build_dir) if re_cc.match(f)]

    # Fix headers
    for src in hl_srcs:
        name = src.split('.pytorch.h')[0]
        print('Fixing', src, name)
        _fix_scheduled_lib_header(name, Path(build_dir) / src)

    ext_name = "custom_halide_ops"
    hl_libs = []  # Halide op libraries to link to
    hl_headers = []  # Halide op headers to include in the wrapper
    for f in hl_srcs:
        # Add all Halide generated torch wrapper
        hl_src = os.path.join(build_dir, f)

        # Add all Halide-generated libraries
        hl_lib = hl_src.split(".")[0] + ".a"
        hl_libs.append(hl_lib)

        hl_header = hl_src.split(".")[0] + ".h"
        hl_headers.append(os.path.basename(hl_header))

    # C++ wrapper code that includes so that we get all the Halide ops in a
    # single python extension
    wrapper_path = os.path.join(build_dir, "pybind_wrapper.cpp")
    sources = [wrapper_path]

    if has_cuda:
        print("Generating CUDA wrapper")
        generate_pybind_wrapper(wrapper_path, hl_headers, True)
        from torch.utils.cpp_extension import CUDAExtension
        extension = CUDAExtension(ext_name, sources,
                                  include_dirs=include_dirs,
                                  extra_objects=hl_libs,
                                  libraries=["cuda"],  # Halide ops need the full cuda lib, not just the RT library
                                  extra_compile_args=compile_args,
                                  extra_link_args=['-L/usr/lib/wsl/lib']) # WSL: libcuda in non-standard location
    else:
        print("Generating CPU wrapper")
        generate_pybind_wrapper(wrapper_path, hl_headers, False)
        from torch.utils.cpp_extension import CppExtension
        extension = CppExtension(ext_name, sources,
                                 include_dirs=include_dirs,
                                 extra_objects=hl_libs,
                                 extra_compile_args=compile_args)

    # Build the Python extension module
    setup(name=ext_name,
          verbose=True,
          url="",
          author_email="your@email.com",
          author="Some Author",
          version="0.0.0",
          ext_modules=[extension],
          cmdclass={"build_ext": BuildExtension}
          )
