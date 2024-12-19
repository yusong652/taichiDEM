import taichi as ti
from cyclic_shear import CyclicShear
import sys
sys.path.append("../src")
from fmt import flt_dtype


# initialization
ti.init(arch=ti.cpu, device_memory_fraction=0.3,
        random_seed=512, default_fp=flt_dtype,
        default_ip=ti.i32, debug=True,
        fast_math=False)

number_particle = 1024 * 64

cs = CyclicShear(number_particle, vt_is_on=False) 
cs.init()
if __name__ == "__main__":
    cs.run()
