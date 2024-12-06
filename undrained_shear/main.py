import taichi as ti
from undrained_shear import UndrainedShear
import sys
sys.path.append("../src")
from fmt import flt_dtype


# initialization
ti.init(arch=ti.cpu, device_memory_fraction=0.5,
        random_seed=512, default_fp=flt_dtype,
        default_ip=ti.i32, debug=True,
        fast_math=False)

number_particle = 1024 * 4

us = UndrainedShear(number_particle, vt_is_on=False) 
us.init()
if __name__ == "__main__":
    us.run()
