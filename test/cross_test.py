import taichi as ti
vec = ti.math.vec3

ti.init(arch=ti.cpu, device_memory_fraction=0.5,
        random_seed=1024, default_fp=ti.f32,
        default_ip=ti.i32, debug=True,
        fast_math=False)

v1 = vec(0., 0., 1.,)

v2 = vec(0., 1, 1.)

print(v2.cross(v1))
print(v2.normalized())
print(v2.norm())