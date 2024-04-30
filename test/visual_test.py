import taichi as ti
from particle import GrainFiled
from contact import ContactInfo
from compress import IsoComp
from grid import GridDomain
from visual import VisualTool

import math
import numpy as np

fp_type = ti.f32
# initialization
ti.init(arch=ti.gpu, device_memory_fraction=0.7,
        random_seed=512, default_fp=fp_type,
        default_ip=ti.i32, debug=True,
        fast_math=False)
gf = GrainFiled(2)
vt = VisualTool(2)
gf.pos[0, 1] = 0.1
gf.pos[1, 1] = -0.1
gf.rad[0] = 0.1
gf.rad[1] = 0.1

vt.update_pos(gf)
while True:
    vt.render(gf)



