import torch
import numpy as np
from pathlib import Path
from halide_ops.create_wheel import make_editable_install

halide_path = Path(__file__).parent / 'halide_develop_install'
make_editable_install(halide_path, add_path=True)
import halide as hl # type: ignore
assert Path(hl.__file__).as_posix().startswith(halide_path.as_posix()), 'Wrong import dir'


