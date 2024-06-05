import taichi as ti
from particle import Particle
from contact import Contact
from compress import IsoComp
from grid import Domain
from fmt import flt_dtype


# initialization
ti.init(arch=ti.cpu, device_memory_fraction=0.5,
        random_seed=256, default_fp=flt_dtype,
        default_ip=ti.i32, debug=True,
        fast_math=False)


particle = Particle(2000)  # Grain field
contact = Contact(particle.num_ptc)  # Contact info
domain = Domain(num_ptc=particle.num_ptc, rad_max=particle.rad_max[0])  # Grid domain

ic = IsoComp(particle, contact, domain, vt_is_on=True)  # Isotropic compression
ic.init()
if __name__ == "__main__":
    ic.pour()
