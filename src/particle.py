import taichi as ti
vec = ti.math.vec3

flt_dtype = ti.f32

@ti.data_oriented
class GrainFiled:
    def __init__(self, num_ptc):
        self.num_ptc = num_ptc
        self.rad_max = ti.field(dtype=flt_dtype, shape=(1,))
        self.rad_max[0] = 0.015
        self.rad_min = ti.field(dtype=flt_dtype, shape=(1,))
        self.rad_min[0] = 0.015
        self.density = ti.field(dtype=flt_dtype, shape=(1,))
        self.density[0] = 2650.0 * 1.0
        self.gravity = 9.81 * 10.0
        self.pos = ti.field(dtype=flt_dtype, shape=(num_ptc, 3),
                            name="position")
        self.pos_pre = ti.field(dtype=flt_dtype, shape=(num_ptc, 3),
                                name="previous position")
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
        self.vel_pre = ti.field(dtype=flt_dtype, shape=(num_ptc, 3),
                                name="previous velocity")
        self.vel_rot = ti.field(dtype=flt_dtype, shape=(num_ptc, 3),
                                name="rotational velocity")
        self.vel_rot_pre = ti.field(dtype=flt_dtype, shape=(num_ptc, 3),
                                    name="previous rotational vel")
        self.acc = ti.field(dtype=flt_dtype, shape=(num_ptc, 3),
                            name="acceleration")
        self.acc_rot = ti.field(dtype=flt_dtype, shape=(num_ptc, 3),
                                name="rotational acceleration")
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

        for i, j in ti.ndrange(self.num_ptc, 3):
            self.pos_pre[i, j] = self.pos[i, j]

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
                # DEBUG Mode *******************************************
                # if i == 3:
                #     print(self.force_s[3, 0])
                # DEBUG Mode *******************************************
                self.acc[i, j] = self.force[i, j] / self.mass[i]
                self.acc_rot[i, j] = self.moment[i, j] / self.inertia[i]

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
                self.vel_pre[i, j] = self.vel[i, j]
                self.vel_rot_pre[i, j] = self.vel_rot[i, j]
        for i in range(self.num_ptc):
            for j in range(3):
                self.vel[i, j] += self.acc[i, j] * dt
                self.vel_rot[i, j] += self.acc_rot[i, j] * dt
                # DEBUG Mode *********************************
                # self.vel[i, j] *= 0.9
                # self.vel[i, j] = 0.0
                # if i == 3:
                #     print(self.vel[i, 0])
                # a = 0.0
                # a_rot = 0.0
                # self.v[i, j] = 0.0
                # DEBUG Mode *********************************
        # self.vel[0, 1] = 0.0

    @ti.kernel
    def update_pos(self, dt: flt_dtype):
        """
        The position of particle is updated based on
        :param dt:
        :return:
        """
        for i in range(self.num_ptc):
            for j in range(3):
                self.pos[i, j] += self.vel[i, j] * dt

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

    def update(self, dt: flt_dtype):
        self.update_acc()
        self.clear_force()
        self.update_vel(dt)
        self.update_pos(dt)

    @ti.kernel
    def calm(self):
        for i in range(self.num_ptc):
            for j in range(3):
                self.vel[i, j] = 0.0

