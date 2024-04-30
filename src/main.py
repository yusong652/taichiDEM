import taichi as ti
from particle import GrainFiled
from contact import ContactInfo
from compress import IsoComp
from grid import GridDomain
from visual import VisualTool


# initialization
ti.init(arch=ti.cpu, device_memory_fraction=0.7,
        random_seed=1024, default_fp=ti.f32,
        default_ip=ti.i32, debug=True,
        fast_math=False)


gf = GrainFiled(256)  # Grain field
ci = ContactInfo(gf.num_ptc)  # Contact info
gd = GridDomain(num_ptc=gf.num_ptc, rad_max=gf.rad_max[0])  # Grid domain
vt = VisualTool(n=gf.num_ptc)

ic = IsoComp(gf, ci, gd, vt=vt)  # Isotropic compression
ic.init()
if __name__ == "__main__":
    ic.compress()
