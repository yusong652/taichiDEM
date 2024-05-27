import taichi as ti
vec = ti.math.vec3

flt_type = ti.f32

@ti.data_oriented
class Wall:
    def __init__(self, num_wall, pos_x_min, pos_x_max, pos_y_min, pos_y_max, pos_z_min, pos_z_max):
        self.number = num_wall
        self.position = ti.field(dtype=flt_type, shape=(num_wall, 3))
        self.normal = ti.field(dtype=flt_type, shape=(num_wall, 3))
        self.velocity = ti.field(dtype=flt_type, shape=(num_wall, 3))
        self.friction = ti.field(dtype=flt_type, shape=(num_wall,))
        self.stiffnessNorm = ti.field(dtype=flt_type, shape=(num_wall,))
        self.stiffnessShear = ti.field(dtype=flt_type, shape=(num_wall,))
        self.dampNRatio = ti.field(dtype=flt_type, shape=(num_wall,))
        self.dampSRatio = ti.field(dtype=flt_type, shape=(num_wall,))
        self.initialize_box_pos(pos_x_min, pos_x_max, pos_y_min, pos_y_max, pos_z_min, pos_z_max)
        self.initialize_box_normal()
        self.initialize_box_friction()
        self.initialize_box_stiffness(stiff_norm=5.0e7, stiff_shear=1.0e7)
        self.initialize_box_dampNRatio(damp=0.3)
        self.initialize_box_dampSRatio(damp=0.3)

    def initialize_box_pos(self, pos_x_min, pos_x_max, pos_y_min, pos_y_max, pos_z_min, pos_z_max):
        # walls on x-direction
        self.position[0, 0] = pos_x_min
        self.position[0, 1] = 0.0
        self.position[0, 2] = 0.0
        self.position[1, 0] = pos_x_max
        self.position[1, 1] = 0.0
        self.position[1, 2] = 0.0
        # walls on y-direction
        self.position[2, 0] = pos_y_min
        self.position[2, 1] = 0.0
        self.position[2, 2] = 0.0
        self.position[3, 0] = pos_y_max
        self.position[3, 1] = 0.0
        self.position[3, 2] = 0.0
        # walls on z-direction
        self.position[4, 0] = pos_z_min
        self.position[4, 1] = 0.0
        self.position[4, 2] = 0.0
        self.position[5, 0] = pos_z_max
        self.position[5, 1] = 0.0
        self.position[5, 2] = 0.0

    def initialize_box_normal(self, ):
        normal_x = vec(1.0, 0.0, 0.0)
        normal_y = vec(0.0, 1.0, 0.0)
        normal_z = vec(0.0, 0.0, 1.0)
        self.normal[0, 0] = normal_x[0]
        self.normal[0, 1] = normal_x[1]
        self.normal[0, 2] = normal_x[2]
        self.normal[1, 0] = -normal_x[0]
        self.normal[1, 1] = -normal_x[1]
        self.normal[1, 2] = -normal_x[2]
        self.normal[2, 0] = normal_y[0]
        self.normal[2, 1] = normal_y[1]
        self.normal[2, 2] = normal_y[2]
        self.normal[3, 0] = -normal_y[0]
        self.normal[3, 1] = -normal_y[1]
        self.normal[3, 2] = -normal_y[2]
        self.normal[4, 0] = normal_z[0]
        self.normal[4, 1] = normal_z[1]
        self.normal[4, 2] = normal_z[2]
        self.normal[5, 0] = -normal_z[0]
        self.normal[5, 1] = -normal_z[1]
        self.normal[5, 2] = -normal_z[2]

    @ti.kernel
    def initialize_box_friction(self, fric=0.5):
        for i in range(self.number):
            self.friction[i] = fric

    @ti.kernel
    def initialize_box_stiffness(self, stiff_norm=5.0e7, stiff_shear=1.0e7):
        for i in range(self.number):
            self.stiffnessNorm[i] = stiff_norm
            self.stiffnessShear[i] = stiff_shear

    @ti.kernel
    def initialize_box_dampNRatio(self, damp=0.3):
        for i in range(self.number):
            self.dampNRatio[i] = damp

    @ti.kernel
    def initialize_box_dampSRatio(self, damp=0.3):
        for i in range(self.number):
            self.dampSRatio[i] = damp

    def set_velocity(self, index_wall: ti.int32, velocity: vec):
        self.velocity[index_wall, 0] = velocity[0]
        self.velocity[index_wall, 1] = velocity[1]
        self.velocity[index_wall, 2] = velocity[2]

    @ti.kernel
    def update_position(self, timestep: flt_type):
        for i in range(self.number):
            self.position[i, 0] += self.velocity[i, 0] * timestep
            self.position[i, 1] += self.velocity[i, 1] * timestep
            self.position[i, 2] += self.velocity[i, 2] * timestep
