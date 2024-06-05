import taichi as ti
vec = ti.math.vec3

ti.init(arch=ti.cpu, device_memory_fraction=0.5,
        random_seed=1024, default_fp=ti.f64,
        default_ip=ti.i32, debug=True,
        fast_math=False)


@ti.data_oriented
class Tester:
    def __init__(self):
        self.field = ti.field(dtype=ti.f32, shape=(6,))
        self.vector = vec(0.0, 1.0, 0.0)

    @ti.func
    def get_two_val(self) -> ti.f32:
        return self.field[0], self.field[1]

    @ti.func
    def get_index(self) -> ti.i32:
        index_i = ti.atomic_add(self.field[0], 1)
        return index_i

    @ti.kernel
    def test(self):
        index = self.get_index()
        print(index)
        print(self.field[0])

    @ti.kernel
    def test1(self):
        vec2 = vec(0.2, 1.5, 3.0)
        normal = self.vector.normalized()
        cross_prod = self.vector.cross(vec2*1.00002)
        res = cross_prod.dot(normal)
        print(cross_prod)

    @ti.kernel
    def test2(self):
        vec3 = vec(-1.0, 0.0, 0)
        res = self.vector.cross(vec3)
        print(res)


tester = Tester()
tester.test2()
