import taichi as ti
from fmt import flt_dtype
vec = ti.math.vec3


@ti.data_oriented
class Particle:
    def __init__(self, num_ptc, radius_max=0.01, radius_min=0.01):
        self.num_ptc = num_ptc
        if radius_min > radius_max:
            raise ValueError('Radius_min can not be larger than radius_max!')
        self.rad_max = ti.field(dtype=flt_dtype, shape=(1,))
        self.rad_max[0] = radius_max
        self.rad_min = ti.field(dtype=flt_dtype, shape=(1,))
        self.rad_min[0] = radius_min
        self.density = ti.field(dtype=flt_dtype, shape=(1,))
        self.density[0] = 2650.0 * 1.0
        self.gravity = 9.81 * 10.0
        self.pos = ti.field(dtype=flt_dtype, shape=(num_ptc, 3),
                            name="position")
        self.grid_idx = ti.field(dtype=ti.i32, shape=(num_ptc, 3),
                                 name="grid_idx")  # id of located grid
        self.mass = ti.field(dtype=flt_dtype, shape=(num_ptc,),
                             name="mass")
        self.rad = ti.field(dtype=flt_dtype, shape=(num_ptc,),
                            name="radius")
        self.inertia = ti.field(dtype=flt_dtype, shape=(num_ptc,),
                                name="inertial moment")
        self.vel = ti.field(dtype=flt_dtype, shape=(num_ptc, 3),
                            name="velocity")
        self.velRot = ti.field(dtype=flt_dtype, shape=(num_ptc, 3),
                               name="rotational velocity")
        self.acc = ti.field(dtype=flt_dtype, shape=(num_ptc, 3),
                            name="acceleration")
        self.accPre = ti.field(dtype=flt_dtype, shape=(num_ptc, 3),
                               name="previous acceleration")
        self.accRot = ti.field(dtype=flt_dtype, shape=(num_ptc, 3),
                               name="rotational acceleration")
        self.accRotPre = ti.field(dtype=flt_dtype, shape=(num_ptc, 3),
                                  name="previous rotational acceleration")
        self.force_n = ti.field(dtype=flt_dtype, shape=(num_ptc, 3),
                                name="contact normal force")
        self.force_s = ti.field(dtype=flt_dtype, shape=(num_ptc, 3),
                                name="contact shear force")
        self.force = ti.field(dtype=flt_dtype, shape=(num_ptc, 3),
                              name="contact force")
        self.moment = ti.field(dtype=flt_dtype, shape=(num_ptc, 3),
                               name="moment")
        self.volume_s = ti.field(dtype=flt_dtype, shape=(1), name='solid volume')

    @ti.kernel
    def init_particle(self, 
        init_x_min: flt_dtype, init_x_max: flt_dtype,
        init_y_min: flt_dtype, init_y_max: flt_dtype,
        init_z_min: flt_dtype, init_z_max: flt_dtype):
        """
        Distribute particles into a cuboid space randomly
        Note that the collapse between particles is inevitable
        Energy should be dissipated in a calm process after
        particle generation.
        Then the basic attributes of radius, total volume, mass, mo-
        ment of inertia
        is applied to each particle.
        :param init_len_x: length of the cuboid space in x direction
        :param init_len_y: length of the cuboid space in y direction
        :param init_len_z: length of the cuboid space in z direction
        :return: None
        """
        for i in range(self.num_ptc):
            # Distribute particles in a cubic enclosed space.
            init_len_x = init_x_max - init_x_min
            init_len_y = init_y_max - init_y_min
            init_len_z = init_z_max - init_z_min
            pos = vec(
                ti.random() * init_len_x + init_x_min,
                ti.random() * init_len_y + init_y_min,
                ti.random() * init_len_z + init_z_min
            )
            self.pos[i, 0] = pos[0]
            self.pos[i, 1] = pos[1]
            self.pos[i, 2] = pos[2]

            self.rad[i] = ti.random() * (self.rad_max[0] - self.rad_min[0]) + self.rad_min[0]
            self.mass[i] = self.density[0] * ti.math.pi * self.rad[i] ** 3 * 4 / 3
            self.inertia[i] = self.mass[i] * self.rad[i] ** 2 * 2.0 / 5.0  # Moment of inertia
        for i in range(self.num_ptc):
            self.volume_s[0] += self.rad[i] ** 3 * 4 / 3 * ti.math.pi

    @ti.kernel
    def update_acc(self, ):
        """
        Transfer force_n and force_s of each particle to force
        :return: None
        """
        for i in range(self.num_ptc):
            # applying gravity
            self.force[i, 0] = 0.0
            self.force[i, 1] = -self.gravity * self.mass[i]
            self.force[i, 2] = -0.0 * self.mass[i]
            for j in range(3):
                self.force[i, j] += self.force_n[i, j]
                self.force[i, j] += self.force_s[i, j]
                self.acc[i, j] = self.force[i, j] / self.mass[i]
                self.accRot[i, j] = self.moment[i, j] / self.inertia[i]

    @ti.kernel
    def record_acc(self, ):
        for i in range(self.num_ptc):
            self.accPre[i, 0] = self.acc[i, 0]
            self.accPre[i, 1] = self.acc[i, 1]
            self.accPre[i, 2] = self.acc[i, 2]
            self.accRotPre[i, 0] = self.accRot[i, 0]
            self.accRotPre[i, 1] = self.accRot[i, 1]
            self.accRotPre[i, 2] = self.accRot[i, 2]

    @ti.kernel
    def update_vel(self, dt: flt_dtype):
        """
        Record the rotational and translational velocity to previous
        velocity field to obtain the initial shear displacement inc-
        rement between two particles
        Update the velocity and rotational velocity
        :param dt: timestep
        :return: None
        """
        for i in range(self.num_ptc):
            for j in range(3):
                self.vel[i, j] += (self.acc[i, j] + self.accPre[i, j]) / 2.0 * dt
                self.velRot[i, j] += (self.accRot[i, j] + self.accRotPre[i, j]) / 2.0 * dt

    @ti.kernel
    def update_pos(self, dt: flt_dtype):
        """
        The position of particle is updated based on
        :param dt:
        :return:
        """
        for i in range(self.num_ptc):
            for j in range(3):
                self.pos[i, j] += (self.vel[i, j] + self.acc[i, j]*dt/2.0) * dt
        # print("vel in update:", self.vel[0, 0], self.vel[0, 1], self.vel[0, 2])

    @ti.kernel
    def clear_force(self):
        """
        Clear normal and shear force
        Contact force in contact resolution process is added
        to normal and shear force field for each particle
        :return:
        """
        for i in range(self.num_ptc):
            for j in range(3):
                self.force_n[i, j] = 0.0
                self.force_s[i, j] = 0.0
                self.moment[i, j] = 0.0

    @ti.kernel
    def calm(self):
        for i in range(self.num_ptc):
            for j in range(3):
                self.vel[i, j] = 0.0

