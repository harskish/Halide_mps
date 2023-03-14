import torch
import numpy as np
from pathlib import Path
from halide_ops.create_wheel import make_editable_install
from halide_ops.custom_ops_hl import get_plugin
from halide_ops.halide_pt_op import build_aot, build_pipeline, build_pt_exts, built_ext, aot_build_pt_exts

halide_path = Path(__file__).parent / 'halide_develop_install'
make_editable_install(halide_path, add_path=True)
import halide as hl # type: ignore
assert Path(hl.__file__).as_posix().startswith(halide_path.as_posix()), 'Wrong import dir'

import os
assert os.getcwd().endswith('HelloPyTorch')

BIN = 'out'
HL_TARGET = 'host'
CUDA_TARGET = 'host-cuda-cuda_capability_61-user_context'

OPS = [
    f'{BIN}/{HL_TARGET}/add_float32.a',
    f'{BIN}/{HL_TARGET}/add_float64.a',
    f'{BIN}/{HL_TARGET}/add_grad_float32.a',
    f'{BIN}/{HL_TARGET}/add_grad_float64.a',
    f'{BIN}/{HL_TARGET}/add_halidegrad_float32.a',
    f'{BIN}/{HL_TARGET}/add_halidegrad_float64.a',
]

CUDA_OPS = [
    f'{BIN}/{HL_TARGET}/add_cuda_float32.a',
    f'{BIN}/{HL_TARGET}/add_cuda_float64.a',
    f'{BIN}/{HL_TARGET}/add_grad_cuda_float32.a',
    f'{BIN}/{HL_TARGET}/add_grad_cuda_float64.a',
    f'{BIN}/{HL_TARGET}/add_halidegrad_cuda_float32.a',
    f'{BIN}/{HL_TARGET}/add_halidegrad_cuda_float64.a',
]

# Workflows:
# @hl_torch_op => build_pipeline => build_pt_exts

# Python Generators:
# https://github.com/halide/Halide/pull/6764
# https://github.com/halide/Halide/blob/main/python_bindings/test/generators/simplepy_generator.py

# Build generator
#build_pipeline()