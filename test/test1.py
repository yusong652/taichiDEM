import taichi as ti
import numpy as np

ti.init(arch=ti.gpu, device_memory_fraction=0.7,
        random_seed=1024, default_fp=ti.f64,
        default_ip=ti.i32, debug=True,
        fast_math=False)
window = ti.ui.Window("Test for Drawing 3d-lines", (768, 768))
canvas = window.get_canvas()
scene = ti.ui.Scene()
camera = ti.ui.Camera()
camera.position(5, 2, 2)
particles_pos = ti.Vector.field(3, dtype=ti.f32, shape=(1))

while window.running:
    camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.RMB)
    scene.set_camera(camera)
    scene.ambient_light((0.8, 0.8, 0.8))
    scene.point_light(pos=(0.5, 1.5, 1.5), color=(1, 1, 1))

    scene.particles(particles_pos, color = (0.68, 0.26, 0.19), radius=0.1)
    # Draw 3d-lines in the scene
    canvas.scene(scene)
    window.show()