import taichi as ti
from heap import Slope
import sys
sys.path.append("../src")
from fmt import flt_dtype


# initialization
ti.init(arch=ti.gpu, device_memory_fraction=0.5,
        random_seed=512, default_fp=flt_dtype,
        default_ip=ti.i32, debug=True,
        fast_math=False)

number_particle = 1024 * 512

ic = Slope(number_particle, vt_is_on=False) 
ic.init()
if __name__ == "__main__":
    ic.run()
