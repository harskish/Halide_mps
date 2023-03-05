import torch
from pathlib import Path
from halide_ops.create_wheel import make_editable_install
make_editable_install()
import halide as hl

from halide_ops.halide_pt_op import build_pt_exts, build_pipeline

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
# https://github.com/halide/Halide/blob/main/README_python.md#writing-a-generator-in-python

a = torch.ones(1, 2, 8, 8, device='cuda')
b = torch.ones(1, 2, 8, 8, device='cuda')*3
gt = torch.ones(1, 2, 8, 8, device='cuda')*4

#############################
# Make + python hybrid build
#############################

# # 1. Build generator & pipelines with make
# os.system('make gpu')

# # 2. Build torch ops with python
# build_pt_exts(Path('bin/host'), has_cuda=True, has_mps=False, ext_name='custom_halide_ops')

# # 3. Test
# import modules # type: ignore
# assert '.cache' in modules.ops.__file__, 'Wrong op location'
# add = modules.Add('add_grad')
# output = add(a, b)
# diff = (output-gt).sum().item()
# assert diff == 0.0, "Test failed: results differ"
# print('Make + python succeeded')

####################
# Full python build
####################
from halide_ops.halide_pt_op import hl_torch_op

@hl_torch_op
def vadd(a, b, target: hl.Target):
    x, y, c, n = hl.vars('x y c n')
    out = hl.Func('output')
    out[x, y, c, n] = a[x, y, c, n] + b[x, y, c, n]

    tx, xy, cn, allvars = hl.vars('tx xy cn allvars')
    if target.has_gpu_feature():
        return out \
            .fuse(x, y, xy) \
            .fuse(c, n, cn) \
            .fuse(xy, cn, allvars) \
            .gpu_tile(allvars, tx, 128)
    else:
        return out \
            .compute_root() \
            .fuse(c, n, cn) \
            .fuse(y, cn, allvars) \
            .parallel(allvars, 8) \
            .vectorize(x, 8)

output = torch.zeros_like(gt)
vadd(a, b, out=output)
diff = (output-gt).sum().item()
assert diff == 0.0, "Test failed: results differ"
print('Wrapper worked')