import taichi as ti
from fmt import flt_dtype
vec = ti.math.vec3
vec4 = ti.math.vec4
mat3x3 = ti.types.matrix(3, 3, flt_dtype)


@ti.data_oriented
class Particle:
    def __init__(self, number, radius_max=0.015, radius_min=0.01):
        self.number = number
        if radius_min > radius_max:
            raise ValueError('Radius_min can not be larger than radius_max!')
        self.radMax = ti.field(dtype=flt_dtype, shape=(1,))
        self.radMax[0] = radius_max
        self.radMin = ti.field(dtype=flt_dtype, shape=(1,))
        self.radMin[0] = radius_min
        self.density = ti.field(dtype=flt_dtype, shape=(1,))
        self.density[0] = 2650.0
        self.pos = ti.field(dtype=flt_dtype, shape=(number, 3),
                            name="position")
        self.verletDisp = ti.field(dtype=flt_dtype, shape=(number, 3))
        self.grid_idx = ti.field(dtype=ti.i32, shape=(number, 3),
                                 name="grid_idx")  # id of located grid
        self.mass = ti.field(dtype=flt_dtype, shape=(number,),
                             name="mass")
        self.rad = ti.field(dtype=flt_dtype, shape=(number,),
                            name="radius")
        self.inertia = ti.field(dtype=flt_dtype, shape=(number, 3),
                                name="inertial moment")
        self.inv_i = ti.field(dtype=flt_dtype, shape=(number, 3),name="inverse")
        self.vel = ti.field(dtype=flt_dtype, shape=(number, 3),
                            name="velocity")
        self.velRot = ti.field(dtype=flt_dtype, shape=(number, 3),
                               name="rotational velocity")
        self.acc = ti.field(dtype=flt_dtype, shape=(number, 3))
        self.accRot = ti.field(dtype=flt_dtype, shape=(number, 3))
        self.angmoment = ti.field(dtype=flt_dtype, shape=(number, 3), name="angular moment")

        self.forceContact = ti.field(dtype=flt_dtype, shape=(number, 3),
                                     name="contact force")
        self.torque = ti.field(dtype=flt_dtype, shape=(number, 3),
                               name="moment")
        self.q = ti.field(dtype=flt_dtype, shape=(number, 4), name="quaternion")
        self.damp_f = ti.field(dtype=flt_dtype, shape=(1, ),)
        self.damp_f[0] = 0.0
        self.damp_t = ti.field(dtype=flt_dtype, shape=(1, ),)
        self.damp_t[0] = 0.0
        self.volumeSolid = ti.field(dtype=flt_dtype, shape=(1), name='solid volume')

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
        for i in range(self.number):
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

            self.rad[i] = ti.random() * (self.radMax[0] - self.radMin[0]) + self.radMin[0]
            self.mass[i] = self.density[0] * ti.math.pi * self.rad[i] ** 3 * 4 / 3
            self.inertia[i, 0] = self.mass[i] * self.rad[i] ** 2 * 2.0 / 5.0  # Moment of inertia
            self.inertia[i, 1] = self.mass[i] * self.rad[i] ** 2 * 2.0 / 5.0
            self.inertia[i, 2] = self.mass[i] * self.rad[i] ** 2 * 2.0 / 5.0
            self.inv_i[i, 0] = 1.0 / self.inertia[i, 0]
            self.inv_i[i, 1] = 1.0 / self.inertia[i, 1]
            self.inv_i[i, 2] = 1.0 / self.inertia[i, 2]
        for i in range(self.number):
            self.volumeSolid[0] += self.rad[i] ** 3 * 4 / 3 * ti.math.pi

    @ti.func
    def damp_resultant_force(self, damp: flt_dtype, resultant_force:vec, vel:vec) -> vec:
        resultant_force[0] *= 1.0 - damp * ti.math.sign(resultant_force[0]*vel[0])
        resultant_force[1] *= 1.0 - damp * ti.math.sign(resultant_force[1]*vel[1])
        resultant_force[2] *= 1.0 - damp * ti.math.sign(resultant_force[2]*vel[2])
        return resultant_force

    @ti.func
    def SetDQ(self, q, omega):
        qw, qx, qy, qz = q[3], q[0], q[1], q[2]
        ox, oy, oz = omega[0], omega[1], omega[2]
        return 0.5 * vec4([
            ox * qw - oy * qz + oz * qy,
            oy * qw - oz * qx + ox * qz,
            oz * qw - ox * qy + oy * qx,
            -ox * qx - oy * qy - oz * qz])

    @ti.func
    def SetToRotate(self, q):
        qw, qx, qy, qz = q[3], q[0], q[1], q[2]
        return mat3x3([[1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)], 
                       [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)], 
                       [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)]])

    @ti.func
    def w_dot(self, w, torque, inertia, inv_inertia):
        return vec((torque[0] + w[1] * w[2] * (inertia[1] - inertia[2])) * inv_inertia[0],
            (torque[1] + w[2] * w[0] * (inertia[2] - inertia[0])) * inv_inertia[1],
            (torque[2] + w[0] * w[1] * (inertia[0] - inertia[1])) * inv_inertia[2])

    @ti.func
    def normalize_quaternion(self, quaternion: vec4) -> vec4:
        res = vec4(0.0, 0.0, 0.0, 0.0)
        if quaternion.norm() > 0.0:
            res = quaternion / quaternion.norm()
        return res

    @ti.kernel
    def update_pos_euler(self, dt: flt_dtype, gravity: vec):
        """
        The position of particle is updated based on euler integration
        :param dt:
        :return:
        """
        for i in range(self.number):
            fdamp = self.damp_f[0]
            tdamp = self.damp_t[0]
            
            cforce, ctorque = self.get_force_contact(i), self.get_torque_contact(i)

            mass = self.get_mass(i)
            old_vel, old_disp = self.get_vel(i), self.get_verlet_disp(i)
            force = self.cundall_damp1st(fdamp, cforce + gravity * mass , old_vel)
            
            av = force / mass 
            vel = old_vel + dt * av
            delta_x = dt * vel
            self.set_acc(i, av)
            self.set_vel(i, vel)
            x = self.get_pos(i)
            self.set_pos(i, x + delta_x)
            self.set_verlet_disp(i, old_disp + delta_x)

            inv_i = self.get_inv_i(i)
            old_omega = self.get_vel_rot(i)

            torque = self.cundall_damp1st(tdamp, ctorque, old_omega)
            aw = torque * inv_i 
            omega = old_omega + dt * aw
            self.set_vel_rot(i, omega)

    @ti.kernel
    def update_pos_verlet(self, dt: flt_dtype, gravity: vec):
        """
        The position of particle is updated based on Verlet integration
        :param dt:
        :return:
        """
        for i in range(self.number):
            
            fdamp = self.damp_f[0]
            tdamp = self.damp_t[0]
            
            cforce, ctorque = self.get_force_contact(i), self.get_torque_contact(i)

            mass = self.get_mass(i)
            old_av, old_vel, old_pos = self.get_acc(i), self.get_vel(i), self.get_pos(i)

            vel_half = old_vel + 0.5 * dt * old_av
            force = self.cundall_damp1st(fdamp, cforce + gravity * mass , vel_half)
            pos = old_pos + dt * vel_half 
            av = force / mass
            vel = vel_half + 0.5 * av * dt
            
            self.set_acc(i, av)
            self.set_vel(i, vel)
            self.set_pos(i, pos)
            
            inv_i = self.get_inv_i(i)
            inertia = 1. / inv_i
            old_omega, old_q = self.get_vel_rot(i), self.get_q(i)

            torque = self.cundall_damp1st(tdamp, ctorque, old_omega)
            rotation_matrix = self.SetToRotate(old_q)

            torque_local = rotation_matrix.transpose() @ torque
            omega_local = rotation_matrix.transpose() @ old_omega
            K1 = dt * self.w_dot(omega_local, torque_local, inertia, inv_i)
            K2 = dt * self.w_dot(omega_local + K1, torque_local, inertia, inv_i)
            K3 = dt * self.w_dot(omega_local + 0.25 * (K1 + K2), torque_local, inertia, inv_i)
            omega_local += (K1 + K2 + 4. * K3) / 6.
            omega = rotation_matrix @ omega_local

            dq = dt * self.SetDQ(old_q, omega_local)
            q = self.normalize_quaternion(old_q + dq)

            self.set_vel_rot(i, omega)
            self.set_q(i, q)

    @ti.kernel
    def clear_force(self):
        for i in range(self.number):
            for j in range(3):
                self.forceContact[i, j] = 0.0
                self.torque[i, j] = 0.0

    @ti.kernel
    def calm(self):
        """
        clear translational and rotational velocity
        :return:
        """
        for i in range(self.number):
            for j in range(3):
                self.vel[i, j] = 0.0
                self.velRot[i, j] = 0.0

    @ti.func
    def add_force_to_ball(self, i: ti.i32, force: vec, torque: vec):
        """

        :param i: id of the particle
        :param force: force at the contact point
        :param torque: torque at the contact point
        :return: None
        """
        self.forceContact[i, 0] += force[0]
        self.forceContact[i, 1] += force[1]
        self.forceContact[i, 2] += force[2]
        self.torque[i, 0] += torque[0]
        self.torque[i, 1] += torque[1]
        self.torque[i, 2] += torque[2]

    @ti.func
    def sgn(self, x: flt_dtype) -> flt_dtype:
        if x != 0:
            x /= ti.abs(x)
        return x

    @ti.func
    def cundall_damp1st(self, damp: flt_dtype, force: vec, vel: vec) -> vec:
        force[0] *= 1. - damp * self.sgn(force[0] * vel[0])
        force[1] *= 1. - damp * self.sgn(force[1] * vel[1])
        force[2] *= 1. - damp * self.sgn(force[2] * vel[2])
        return force

    @ti.func
    def get_radius(self, i: ti.i32) -> flt_dtype:
        return self.rad[i]

    @ti.func
    def get_mass(self, i: ti.i32) -> flt_dtype:
        return self.mass[i]

    @ti.func
    def get_inv_i(self, i: ti.i32) -> flt_dtype:
        return vec(self.inv_i[i, 0] , self.inv_i[i, 1], self.inv_i[i, 2])

    @ti.func
    def get_pos(self, i: ti.i32) -> vec:
        return vec(self.pos[i, 0], self.pos[i, 1], self.pos[i, 2])

    @ti.func
    def set_pos(self, i: ti.i32, pos: vec):
        self.pos[i, 0] = pos[0]
        self.pos[i, 1] = pos[1]
        self.pos[i, 2] = pos[2]

    @ti.func
    def get_verlet_disp(self, i: ti.i32) -> vec:
        return vec(self.verletDisp[i, 0], self.verletDisp[i, 1], self.verletDisp[i, 2])

    @ti.func
    def set_verlet_disp(self, i: ti.i32, verletDisp: vec):
        self.verletDisp[i, 0] = verletDisp[0]
        self.verletDisp[i, 1] = verletDisp[1]
        self.verletDisp[i, 2] = verletDisp[2]

    @ti.func
    def get_vel(self, i: ti.i32) -> vec:
        return vec(self.vel[i, 0], self.vel[i, 1], self.vel[i, 2])

    @ti.func
    def set_vel(self, i: ti.i32, vel: vec):
        self.vel[i, 0] = vel[0]
        self.vel[i, 1] = vel[1]
        self.vel[i, 2] = vel[2]

    @ti.func
    def get_vel_rot(self, i: ti.i32) -> vec:
        return vec(self.velRot[i, 0], self.velRot[i, 1], self.velRot[i, 2])

    @ti.func
    def set_vel_rot(self, i: ti.i32, velRot: vec):
        self.velRot[i, 0] = velRot[0]
        self.velRot[i, 1] = velRot[1]
        self.velRot[i, 2] = velRot[2]

    @ti.func
    def get_angmoment(self, i: ti.i32) -> vec:
        return vec(self.angmoment[i, 0], self.angmoment[i, 1], self.angmoment[i, 2])

    @ti.func 
    def set_angmoment(self, i: ti.i32, angmoment: vec):
        self.angmoment[i, 0] = angmoment[0]
        self.angmoment[i, 1] = angmoment[1]
        self.angmoment[i, 2] = angmoment[2]

    @ti.func
    def get_acc(self, i: ti.i32) -> vec:
        return vec(self.acc[i, 0], self.acc[i, 1], self.acc[i, 2])

    @ti.func
    def get_acc_rot(self, i: ti.i32) -> vec:
        return vec(self.accRot[i, 0], self.accRot[i, 1], self.accRot[i, 2])

    @ti.func
    def set_acc(self, i: ti.i32, acc: vec):
        self.acc[i, 0] = acc[0]
        self.acc[i, 1] = acc[1]
        self.acc[i, 2] = acc[2]

    @ti.func
    def set_acc_rot(self, i: ti.i32, accRot: vec):
        self.accRot[i, 0] = accRot[0]
        self.accRot[i, 1] = accRot[1]
        self.accRot[i, 2] = accRot[2]

    @ti.func
    def set_q(self, i: ti.i32, q: vec4):
        self.q[i, 0] = q[0]
        self.q[i, 1] = q[1]
        self.q[i, 2] = q[2]
        self.q[i, 3] = q[3]

    @ti.func
    def get_q(self, i: ti.i32) -> vec4:
        return vec4(self.q[i, 0], self.q[i, 1], self.q[i, 2], self.q[i, 3])

    @ti.func
    def get_force_contact(self, i: ti.i32) -> vec:
        return vec(self.forceContact[i, 0], self.forceContact[i, 1], self.forceContact[i, 2])

    @ti.func
    def get_torque_contact(self, i: ti.i32) -> vec:
        return vec(self.torque[i, 0], self.torque[i, 1], self.torque[i, 2])
