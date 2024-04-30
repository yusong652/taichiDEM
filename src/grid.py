import taichi as ti
import math

flt_dtype = ti.f32

@ti.data_oriented
class GridDomain:
    def __init__(self, num_ptc, rad_max, domain_size=0.2, ):
        self.domain_size = domain_size
        self.num_grid = math.floor(self.domain_size / (rad_max * 2))
        self.size_grid = self.domain_size / self.num_grid  # Simulation domain of size [domain_size]
        print(f"Grid number: {self.num_grid}x{self.num_grid}x{self.num_grid}")

        self.list_head = ti.field(dtype=ti.i32, shape=self.num_grid * self.num_grid * self.num_grid)
        self.list_cur = ti.field(dtype=ti.i32, shape=self.num_grid * self.num_grid * self.num_grid)
        self.list_tail = ti.field(dtype=ti.i32, shape=self.num_grid * self.num_grid * self.num_grid)

        self.grain_count = ti.field(dtype=ti.i32,
                                    shape=(self.num_grid, self.num_grid, self.num_grid),
                                    name="grain_count")
        self.layer_sum = ti.field(dtype=ti.i32, shape=(self.num_grid, self.num_grid),
                                  name="column_row_sum")
        self.column_sum = ti.field(dtype=ti.i32, shape=self.num_grid, name="column_sum")
        self.prefix_sum = ti.field(dtype=ti.i32, shape=(self.num_grid, self.num_grid, self.num_grid),
                                   name="prefix_sum")
        self.particle_id = ti.field(dtype=ti.i32, shape=num_ptc, name="particle_id")
