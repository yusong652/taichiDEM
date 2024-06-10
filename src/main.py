import taichi as ti
from heap import Slope
from fmt import flt_dtype


# initialization
ti.init(arch=ti.gpu, device_memory_fraction=0.5,
        random_seed=256, default_fp=flt_dtype,
        default_ip=ti.i32, debug=True,
        fast_math=False)

number_particle = 20000


ic = Slope(number_particle, vt_is_on=True)  # Isotropic compression
ic.init()
if __name__ == "__main__":
    ic.run()
