import taichi as ti
import csv

ti.init(arch=ti.cpu, device_memory_fraction=0.7,
        random_seed=1024, default_fp=ti.f64,
        default_ip=ti.i32, debug=True,
        fast_math=False)
@ti.data_oriented
class IsoComp(object):
    def __init__(self, **params):
        pass
    def write(self):
        with open('ic_info.csv', 'w') as file:
                writer = csv.writer(file)
                writer.writerow(['test'])

ic = IsoComp()
ic.write()