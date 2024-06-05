import taichi as ti
import numpy as np
import matplotlib.pyplot as plt
vec = ti.math.vec3


@ti.data_oriented
class VisualTool:
    def __init__(self, n):
        self.vis_pos = ti.Vector.field(3, dtype=ti.f32, shape=n)
        self.rad = ti.field(dtype=ti.f32, shape=(1))
        self.window = ti.ui.Window("Taichi DEM", (1080, 720))
        self.canvas = self.window.get_canvas()
        self.scene = ti.ui.Scene()
        self.camera = ti.ui.Camera()
        self.camera.position(-2.0, -0.0, 0.0)
        self.camera.lookat(0, 0, 0)
        # self.camera.up(0, 0, 0)
        # self.camera.fov(0)
        # self.scene.set_camera(self.camera)
        # self.scene.ambient_light((0.7, 0.7, 0.7))
        # self.scene.point_light(pos=(2, -1.9, 1.2), color=(0.5, 0.5, 0.5))


    @ti.kernel
    def update_pos(self, gf: ti.template()):
        for i in range(self.vis_pos.shape[0]):
            self.vis_pos[i] = vec(gf.pos[i, 0], gf.pos[i, 1], gf.pos[i, 2])

    def render(self, gf: ti.template()):
        # self.camera.track_user_inputs(self.window, movement_speed=0.00, hold_key=ti.ui.RMB)
        self.scene.set_camera(self.camera)
        self.scene.ambient_light((0.8, 0.8, 0.8))
        self.scene.point_light(pos=(1.5, 3.5, 0.3), color=(0.8, 0.8, 0.8))
        self.scene.particles(self.vis_pos, color=(0.7, 0.7, 0.7), radius=gf.rad[0]*0.65)
        self.canvas.scene(self.scene)
        self.window.show()
