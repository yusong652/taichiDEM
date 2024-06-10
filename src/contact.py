import taichi as ti
import numpy as np
from fmt import flt_dtype
vec = ti.math.vec3


@ti.data_oriented
class Contact(object):
    """
    # Allocate fields with fixed size for shear force information storage
    """

    def __init__(self, n, fric=0.5, stiff_n=5.0e7, stiff_s=2.5e7, ):
        self.n = n  # number of particles or rows for contact info storage
        self.frictionBallBall = ti.field(dtype=flt_dtype, shape=(1,))
        self.frictionBallBall[0] = fric
        self.frictionBallWall = ti.field(dtype=flt_dtype, shape=(1,))
        self.frictionBallWall[0] = 0.5
        self.stiffnessNorm = ti.field(dtype=flt_dtype, shape=(1,))
        self.stiffnessNorm[0] = stiff_n
        self.stiffnessShear = ti.field(dtype=flt_dtype, shape=(1,))
        self.stiffnessShear[0] = stiff_s
        self.dampBallBallNorm = ti.field(dtype=flt_dtype, shape=(1,))
        self.dampBallBallNorm[0] = 0.5
        self.dampBallBallShear = ti.field(dtype=flt_dtype, shape=(1,))
        self.dampBallBallShear[0] = 0.3
        self.dampBallWallNorm = ti.field(dtype=flt_dtype, shape=(1,))
        self.dampBallWallNorm[0] = 0.5
        self.dampBallWallShear = ti.field(dtype=flt_dtype, shape=(1,))
        self.dampBallWallShear[0] = 0.3
        self.lenContactRecord = 16
        # id of particles in contact
        self.contacts = ti.field(dtype=ti.i32, shape=(self.n, self.lenContactRecord),
                                 name="contacts")
        # id of particles in contact in the last cycle
        self.contactsPre = ti.field(dtype=ti.i32, shape=(self.n, self.lenContactRecord),
                                    name="contacts_pre")
        # contact number on one particle
        self.contactCounter = ti.field(dtype=ti.i32, shape=(self.n,))
        self.contactDistX = ti.field(dtype=flt_dtype, shape=(self.n, self.lenContactRecord))
        self.contactDistY = ti.field(dtype=flt_dtype, shape=(self.n, self.lenContactRecord))
        self.contactDistZ = ti.field(dtype=flt_dtype, shape=(self.n, self.lenContactRecord))
        # shear force components
        self.forceShearX = ti.field(dtype=flt_dtype, shape=(self.n, self.lenContactRecord))
        self.forceShearY = ti.field(dtype=flt_dtype, shape=(self.n, self.lenContactRecord))
        self.forceShearZ = ti.field(dtype=flt_dtype, shape=(self.n, self.lenContactRecord))
        # normal force components
        self.force_n_x = ti.field(dtype=flt_dtype, shape=(self.n, self.lenContactRecord))
        self.force_n_y = ti.field(dtype=flt_dtype, shape=(self.n, self.lenContactRecord))
        self.force_n_z = ti.field(dtype=flt_dtype, shape=(self.n, self.lenContactRecord))
        # shear force component in the last cycle
        self.forceShearXPre = ti.field(dtype=flt_dtype, shape=(self.n, self.lenContactRecord))
        self.forceShearYPre = ti.field(dtype=flt_dtype, shape=(self.n, self.lenContactRecord))
        self.forceShearZPre = ti.field(dtype=flt_dtype, shape=(self.n, self.lenContactRecord))

        self.contactsWall = ti.field(dtype=flt_dtype, shape=(self.n, 6))
        self.contactsWallPre = ti.field(dtype=flt_dtype, shape=(self.n, 6))
        self.forceShearWallX = ti.field(dtype=flt_dtype, shape=(self.n, 6))
        self.forceShearWallY = ti.field(dtype=flt_dtype, shape=(self.n, 6))
        self.forceShearWallZ = ti.field(dtype=flt_dtype, shape=(self.n, 6))
        self.forceShearWallXPre = ti.field(dtype=flt_dtype, shape=(self.n, 6))
        self.forceShearWallYPre = ti.field(dtype=flt_dtype, shape=(self.n, 6))
        self.forceShearWallZPre = ti.field(dtype=flt_dtype, shape=(self.n, 6))

    def init_contact(self, dt,):
        self.contacts.fill(-1)
        self.contactsPre.fill(-1)
        self.contactCounter.fill(0)
        self.dt = dt
        # self.detect(gf, gd)

    @ti.kernel
    def clear_contact(self):
        """
        Record the current contact information to the previous contact field for the
        shear force update.
        :return: None
        """
        # Record the current contact field and initialize the current contact field.
        for i, k in self.contacts:
            self.contactsPre[i, k] = self.contacts[i, k]
        self.contacts.fill(-1)  # Renew [num_particle, len_rec]
        # Zero the counting list [num_particle * 1, None]
        self.contactCounter.fill(0)  # Initialize
        for i, j in self.forceShearX:
            self.forceShearX[i, j] = 0.0
            self.forceShearY[i, j] = 0.0
            self.forceShearZ[i, j] = 0.0

    @ti.kernel
    def detect(self, gf: ti.template(), gd: ti.template()):
        """
        Handle the inter-particle force
        This method includes fast detection algorithm and contact force calculation
        :param gf: grain field
        :param gd: grid domain
        :return: None
        """
        # Fast detection of the contact between particles
        # Zero the counting field for every grid to initialize a cycle
        gd.grain_count.fill(0)

        # Count the number of particle located in every grid parallely
        for i in range(gf.number):
            grid_idx = ti.math.floor(vec((gf.pos[i, 0] + gd.domain_size / 2) / gd.size_grid,
                                         (gf.pos[i, 1] + gd.domain_size / 2) / gd.size_grid,
                                         (gf.pos[i, 2] + gd.domain_size / 2) / gd.size_grid),
                                     int)  # which grid it is located in. ((0, 1, 2) means
            # the 1st layer, 2nd row, and 3rd column)
            # If a particle with id-i is located in the grid, 1 is added to the grain-count
            # field.
            gd.grain_count[grid_idx] += 1

        # Sum the number of particle in multiple layers to a one-layer-shaped field
        # num_grid indicates the number of grid in each dimension.
        # The total number of grid is num_grid**3
        # Parallely in row and column direction
        # Might be optimized to a parallel process in i, j, and k
        for i, j in ti.ndrange(gd.num_grid, gd.num_grid):
            sum = 0  # sum of particle
            # Serially in k direction
            for k in range(gd.num_grid):
                sum += gd.grain_count[i, j, k]
            # Record in an n*n shape field called layer_sum
            gd.layer_sum[i, j] = sum

        # Sum the number of particle to a row
        # Might be optimized to a parallel process in i and j
        for i in range(gd.num_grid):
            sum = 0
            for j in range(gd.num_grid):
                sum += gd.layer_sum[i, j]
            gd.column_sum[i] = sum

        # The prefix_sum indicates the number of particle in the grids labeled prior to a grid.
        # For instance, grid(0, 1, 1) labeled as 0 * num_grid**2 + 1 * num_grid + 1  should be
        # prior to grid(0, 1, 2). The prefix_sum field of grid(0, 1, 2) records the total numb-
        # er of particles in the prior grids.
        gd.prefix_sum[0, 0, 0] = 0  # Initialize the first element
        # The first row:
        # This process has to be serially summed up because the order affects the result.
        ti.loop_config(serialize=True)  # Validate the serialized mode.
        for i in range(1, gd.num_grid):
            gd.prefix_sum[i, 0, 0] = gd.prefix_sum[i - 1, 0, 0] + gd.column_sum[i - 1]
        # The first layer:
        # This process does not need a serial process, because the value in its first row is -
        # determined. However, summation in j direction has to be serially processed.
        ti.loop_config(serialize=False)  # No need to invalidate manually, because it is False
        # by default
        for i in range(gd.num_grid):  # Parallel
            for j in range(1, gd.num_grid):  # Serial
                gd.prefix_sum[i, j, 0] = gd.prefix_sum[i, j - 1, 0] + gd.layer_sum[i, j - 1]
        # The whole n*n*n shaped field:
        for i, j in ti.ndrange(gd.num_grid, gd.num_grid):  # Parallel
            for k in range(1, gd.num_grid):  # Serial
                gd.prefix_sum[i, j, k] = gd.prefix_sum[i, j, k - 1] + gd.grain_count[i, j,
                k - 1]
        # Record the start point end point in linearly sequenced list (size = num_grid**3)
        # The current list is used to count the number of particle counted in each grid
        for i, j, k in ti.ndrange(gd.num_grid, gd.num_grid, gd.num_grid):  # Parallel
            linear_idx = i * gd.num_grid * gd.num_grid + j * gd.num_grid + k
            gd.list_head[linear_idx] = gd.prefix_sum[i, j, k]
            gd.list_cur[linear_idx] = gd.list_head[linear_idx]  # Current position in a grid
            # The current list is equal to head list because it is under an initial state.
            # Once a particle is counted, the element in the current list should be added 1.
            gd.list_tail[linear_idx] = gd.prefix_sum[i, j, k] + gd.grain_count[i, j, k]

        # Record the id of particle to a linear list (size = num_particle)
        for i in range(self.n):  # Parallel
            grid_idx = ti.floor(vec((gf.pos[i, 0] + gd.domain_size / 2) / gd.size_grid,
                                    (gf.pos[i, 1] + gd.domain_size / 2) / gd.size_grid,
                                    (gf.pos[i, 2] + gd.domain_size / 2) / gd.size_grid),
                                int)  # the grid it is located in
            # Convert the id to a linear form:
            linear_idx = grid_idx[0] * gd.num_grid * gd.num_grid + grid_idx[1] * gd.num_grid + \
                         grid_idx[2]  # n_layer * n_grid**2 + n_row * n_grid + n_grid
            grain_location = ti.atomic_add(gd.list_cur[linear_idx], 1)
            # Record the particle id i to the particle id list:
            gd.particle_id[grain_location] = i

        #######################################################################################
        #  particle id list arrangement finished ###### particle id list arrangement finished #
        #######################################################################################

        # Brute-force collision detection (Not used here)
        '''
        for i in range(n):
            for j in range(i + 1, n):
                resolve(i, j)
        '''
        #######################################################################################
        # Fast collision detection (Adopted here)
        #######################################################################################

        # Find the id of neighboring grids
        for i in range(self.n):
            grid_idx = vec(ti.floor((gf.pos[i, 0] + gd.domain_size / 2) / gd.size_grid, int),
                           ti.floor((gf.pos[i, 1] + gd.domain_size / 2) / gd.size_grid, int),
                           ti.floor((gf.pos[i, 2] + gd.domain_size / 2) / gd.size_grid, int))
            x_begin = ti.max(grid_idx[0] - 1, 0)
            x_end = ti.min(grid_idx[0] + 2, gd.num_grid)

            y_begin = ti.max(grid_idx[1] - 1, 0)
            y_end = ti.min(grid_idx[1] + 2, gd.num_grid)

            z_begin = ti.max(grid_idx[2] - 1, 0)
            z_end = ti.min(grid_idx[2] + 2, gd.num_grid)

            # Search for the particles in the 27(=3*3*3) neighboring grids
            for neigh_i in range(x_begin, x_end):
                for neigh_j in range(y_begin, y_end):
                    for neigh_k in range(z_begin, z_end):
                        neigh_linear_idx = neigh_i * gd.num_grid * gd.num_grid + neigh_j * \
                                           gd.num_grid + neigh_k
                        for p_idx in range(gd.list_head[neigh_linear_idx],
                                           gd.list_tail[neigh_linear_idx]):
                            j = gd.particle_id[p_idx]
                            if i < j:
                                gap = self.get_gap(gf, i, j)
                                if gap < 0:  # Particle in contact detected
                                    force_norm = self.resolve_ball_ball_normal_force(gf, i, j)
                                    index_i, index_j = self.get_cur_col(i, j)
                                    index_pre = self.get_index_pre(i, j)
                                    self.resolve_ball_ball_shear_force(
                                        gf, i, j, force_norm, index_i, index_j, index_pre)
                                else:
                                    pass
                            else:
                                pass

    @ti.func
    def resolve_ball_ball_normal_force(self, gf, i, j) -> vec:
        rel_pos = vec(gf.pos[j, 0] - gf.pos[i, 0],
                      gf.pos[j, 1] - gf.pos[i, 1],
                      gf.pos[j, 2] - gf.pos[i, 2])
        # Obtain the relative velocity (vec3)
        # Distance between two particle
        dist = self.get_magnitude(rel_pos)
        gap = self.get_gap(gf, i, j)
        # Normalize the direction
        normal = rel_pos / dist
        force_norm_lin = vec(normal[0] * gap * self.stiffnessNorm[0],
                             normal[1] * gap * self.stiffnessNorm[0],
                             normal[2] * gap * self.stiffnessNorm[0])
        gf.forceNorm[i, 0] += force_norm_lin[0]
        gf.forceNorm[i, 1] += force_norm_lin[1]
        gf.forceNorm[i, 2] += force_norm_lin[2]
        gf.forceNorm[j, 0] -= force_norm_lin[0]
        gf.forceNorm[j, 1] -= force_norm_lin[1]
        gf.forceNorm[j, 2] -= force_norm_lin[2]
        force_norm_damp = self.get_force_norm_damp(gf, i, j)
        gf.forceNorm[i, 0] += force_norm_damp[0]
        gf.forceNorm[i, 1] += force_norm_damp[1]
        gf.forceNorm[i, 2] += force_norm_damp[2]
        gf.forceNorm[j, 0] -= force_norm_damp[0]
        gf.forceNorm[j, 1] -= force_norm_damp[1]
        gf.forceNorm[j, 2] -= force_norm_damp[2]
        force_norm = force_norm_lin + force_norm_damp
        return force_norm

    @ti.func
    def get_gap(self, gf: ti.template(), i: ti.i32, j: ti.i32) -> flt_dtype:
        """
        the gap between two particle is ([distance] - [sum of radii])
        :param gf: grain fields
        :param i: id of the first particle
        :param j: id of the second particle found in one of the neighboring grids
        :return: the gap between two particle. If the gap is < 0, the two particles are in
        contact.
        """
        rel_pos = vec(gf.pos[j, 0] - gf.pos[i, 0],
                      gf.pos[j, 1] - gf.pos[i, 1],
                      gf.pos[j, 2] - gf.pos[i, 2])

        dist = ti.sqrt(rel_pos[0] ** 2 + rel_pos[1] ** 2 + rel_pos[2] ** 2)

        gap = dist - gf.rad[i] - gf.rad[j]  # gap = d - 2 * r
        return gap

    @ti.func
    def get_cur_col(self, i: ti.i32, j: ti.i32):
        """
        Append the id of particles to the contact field
        :param i: id of the particle
        :param j: id of the other particle
        :return: None
        """
        # Obtain the index of current column in the contact list
        index_i = ti.atomic_add(self.contactCounter[i], 1)  # Get the value and add 1 to it
        # Put the id of another particle to the corresponding position:
        self.contacts[i, index_i] = j
        # Index of the column for another particle
        index_j = ti.atomic_add(self.contactCounter[j], 1)
        self.contacts[j, index_j] = i

        return index_i, index_j

    @ti.func
    def get_force_norm_damp(self, gf: ti.template(), i: ti.i32, j: ti.i32) -> vec:
        rel_pos = vec(gf.pos[j, 0] - gf.pos[i, 0],
                      gf.pos[j, 1] - gf.pos[i, 1],
                      gf.pos[j, 2] - gf.pos[i, 2])
        # Obtain the relative velocity (vec3)
        rel_vel = vec(gf.vel[i, 0] - gf.vel[j, 0],
                      gf.vel[i, 1] - gf.vel[j, 1],
                      gf.vel[i, 2] - gf.vel[j, 2])
        # Distance between two particle
        dist = self.get_magnitude(rel_pos)
        # Normalize the direction
        normal = rel_pos / dist
        # Damping force
        M = (gf.mass[i] * gf.mass[j]) / (gf.mass[i] + gf.mass[j])
        K = self.stiffnessNorm[0]
        C = 2. * self.dampBallBallNorm[0] * ti.sqrt(K * M)
        V = ti.math.dot(rel_vel, normal)
        force_norm_damp = -C * V * normal
        return force_norm_damp

    @ti.func
    def get_index(self, i: ti.i32, j: ti.i32) -> ti.i32:
        """
        Obtain the position of particle j in the contact list of particle i
        :param i: id of the first particle
        :param j: id of the other particle
        :return: None
        """
        index_cur = -1
        for index in range(self.lenContactRecord):
            if self.contacts[i, index] == j:
                index_cur = index
                break
            else:
                pass
        return index_cur

    @ti.func
    def get_index_pre(self, i: ti.i32, j: ti.i32) -> ti.i32:
        index_pre = -1
        for l in range(self.lenContactRecord):
            if self.contactsPre[i, l] == -1:
                break
            elif self.contactsPre[i, l] == j:
                index_pre = l
                break
            else:
                pass
        return index_pre

    @ti.func
    def get_magnitude(self, force: vec) -> flt_dtype:
        """
        Obtain the magnitude of the vector
        :param n-dimensional vector:
        :return: magnitude of the vector
        """
        res = ti.math.sqrt(force[0]**2 + force[1]**2 + force[2]**2)
        return res

    @ti.func
    def normalize(self, force: vec) -> vec:
        res = vec(0.0, 0.0, 0.0)
        if force.norm() > 0.0:
            res = force / force.norm()
        else:
            res = vec(0.0, 0.0, 0.0)
        return res

    @ti.func
    def resolve_ball_ball_shear_force(self, particle: ti.template(), i, j, force_norm, index_i, index_j, index_pre):
        """
        Transform shear force to the new contact plane
        :param particle: grain field
        :return: None
        """
        #######################################################################################
        # Shear force # Shear force # Shear force # Shear force # Shear force # Shear Force   #
        #######################################################################################
        # in parallel
        pos_rel = (vec(particle.pos[j, 0], particle.pos[j, 1], particle.pos[j, 2]) -
                   vec(particle.pos[i, 0], particle.pos[i, 1], particle.pos[i, 2]))
        vel_i = vec(particle.vel[i, 0], particle.vel[i, 1], particle.vel[i, 2])
        vel_j = vec(particle.vel[j, 0], particle.vel[j, 1], particle.vel[j, 2])
        vel_rel = vel_i - vel_j
        dist = self.get_magnitude(pos_rel)
        gap = dist - particle.rad[i] - particle.rad[j]  # gap = d - 2 * r
        normal = pos_rel / dist
        contact_dist_i = (particle.rad[i] + gap / 2.0) * normal
        contact_dist_j = - (particle.rad[j] + gap / 2.0) * normal

        force_transformed = vec(0.0, 0.0, 0.0)
        if index_pre != -1:
            # Previous shear force transformation
            force_shear_pre = vec(self.forceShearXPre[i, index_pre],
                                  self.forceShearYPre[i, index_pre],
                                  self.forceShearZPre[i, index_pre])
            force_reduced = force_shear_pre - force_shear_pre.dot(normal) * normal
            force_transformed = self.normalize(force_reduced) * force_shear_pre.norm()
        # Linear relative velocity induced part:
        vel_rel_shear_lin = vel_rel - vel_rel.dot(normal) * normal
        vel_rot_i = vec(particle.velRot[i, 0], particle.velRot[i, 1], particle.velRot[i, 2])
        vel_rot_j = vec(particle.velRot[j, 0], particle.velRot[j, 1], particle.velRot[j, 2])
        vel_rel_shear_rot = vel_rot_i.cross(contact_dist_i) - vel_rot_j.cross(contact_dist_j)
        vel_rel_shear = vel_rel_shear_lin + vel_rel_shear_rot
        disp_inc = vel_rel_shear * self.dt
        force_shear_inc = - disp_inc * self.stiffnessShear[0]
        # Damping shear force
        M = (particle.mass[i] * particle.mass[j]) / (particle.mass[i] + particle.mass[j])
        K_s = self.stiffnessShear[0]
        C_s = 2. * self.dampBallBallShear[0] * ti.sqrt(K_s * M)
        force_shear_damp = -vel_rel_shear * C_s

        # Linear part
        force_shear_trial = force_transformed + force_shear_inc
        force_shear_trial_mag = self.get_magnitude(force_shear_trial)
        force_norm_mag = self.get_magnitude(force_norm)

        force_shear_lin_lim = force_norm_mag * self.frictionBallBall[0]  # Coulomb limit
        direction_force_trial = self.normalize(force_shear_trial)

        force_shear_lin = vec(0.0, 0.0, 0.0)
        force_shear_total = vec(0.0, 0.0, 0.0)
        if force_shear_trial_mag > force_shear_lin_lim:
            force_shear_lin = direction_force_trial * force_shear_lin_lim
            force_shear_total = force_shear_lin
        else:
            force_shear_lin = force_shear_trial
            force_shear_total = force_shear_lin + force_shear_damp
        self.forceShearXPre[i, index_i] = force_shear_lin[0]
        self.forceShearYPre[i, index_i] = force_shear_lin[1]
        self.forceShearZPre[i, index_i] = force_shear_lin[2]
        self.forceShearXPre[j, index_j] = -force_shear_lin[0]
        self.forceShearYPre[j, index_j] = -force_shear_lin[1]
        self.forceShearZPre[j, index_j] = -force_shear_lin[2]

        # Force and moment sum
        particle.forceShear[i, 0] += force_shear_total[0]
        particle.forceShear[i, 1] += force_shear_total[1]
        particle.forceShear[i, 2] += force_shear_total[2]
        particle.forceShear[j, 0] -= force_shear_total[0]
        particle.forceShear[j, 1] -= force_shear_total[1]
        particle.forceShear[j, 2] -= force_shear_total[2]

        moment_i = contact_dist_i.cross(force_shear_total)
        particle.moment[i, 0] += moment_i[0]
        particle.moment[i, 1] += moment_i[1]
        particle.moment[i, 2] += moment_i[2]
        moment_j = contact_dist_j.cross(-force_shear_total)
        particle.moment[j, 0] += moment_j[0]
        particle.moment[j, 1] += moment_j[1]
        particle.moment[j, 2] += moment_j[2]

    @ti.kernel
    def resolve_ball_wall_force(self, particle: ti.template(), wall: ti.template()):
        for i in range(particle.number):
            # Gap from the boundary in negative x direction:
            for j in range(wall.number):
                pos_dif = vec(particle.pos[i, 0] - wall.position[j, 0],
                              particle.pos[i, 1] - wall.position[j, 1],
                              particle.pos[i, 2] - wall.position[j, 2])
                normal = vec(wall.normal[j, 0], wall.normal[j, 1], wall.normal[j, 2])
                gap = pos_dif.dot(normal) - particle.rad[i]
                force_shear_lin = vec(0.0, 0.0, 0.0)
                force_shear_total = vec(0.0, 0.0, 0.0)
                if gap < 0:
                    # Normal direction
                    force_normal_lin = -gap * self.stiffnessNorm[0] * normal
                    particle.forceNorm[i, 0] += force_normal_lin[0]
                    particle.forceNorm[i, 1] += force_normal_lin[1]
                    particle.forceNorm[i, 2] += force_normal_lin[2]
                    vel_rel_bw = vec(particle.vel[i, 0] - wall.velocity[j, 0],
                                     particle.vel[i, 1] - wall.velocity[j, 1],
                                     particle.vel[i, 2] - wall.velocity[j, 2])
                    vel_rel_norm_dot = vel_rel_bw.dot(normal)
                    M = particle.mass[i]
                    K = self.stiffnessNorm[0]
                    C = 2. * self.dampBallWallNorm[0] * ti.sqrt(K * M)
                    V = vel_rel_norm_dot * normal
                    force_normal_damp = -C * V
                    particle.forceNorm[i, 0] += force_normal_damp[0]
                    particle.forceNorm[i, 1] += force_normal_damp[1]
                    particle.forceNorm[i, 2] += force_normal_damp[2]
                    force_n_total = force_normal_lin + force_normal_damp
                    force_shear_lin_limit = self.get_magnitude(force_n_total) * self.frictionBallWall[0]
                    # Shear direction
                    force_shear_pre = vec(self.forceShearWallXPre[i, j],
                                          self.forceShearWallYPre[i, j],
                                          self.forceShearWallZPre[i, j])
                    # coordinate transform
                    force_reduced = force_shear_pre - force_shear_pre.dot(normal) * normal
                    force_shear_transformed = self.normalize(force_reduced) * force_shear_pre.norm()
                    vel_rel_shear_lin = vel_rel_bw - vel_rel_norm_dot * normal
                    distance_cp = - normal*particle.rad[i]
                    vel_rel_shear_rot = vec(particle.velRot[i, 0],
                                            particle.velRot[i, 1],
                                            particle.velRot[i, 2]).cross(distance_cp)
                    vel_rel_shear = vel_rel_shear_lin + vel_rel_shear_rot
                    force_shear_increment = - vel_rel_shear * self.stiffnessShear[0] * self.dt
                    force_shear_trial = force_shear_transformed + force_shear_increment
                    direction_trial = self.normalize(force_shear_trial)
                    force_trial_mag = self.get_magnitude(force_shear_trial)

                    # Damping shear force
                    M = particle.mass[i]
                    K_s = self.stiffnessShear[0]
                    C_s = 2. * self.dampBallWallShear[0] * ti.sqrt(K_s * M)
                    force_shear_damp = -vel_rel_shear * C_s


                    if force_trial_mag > force_shear_lin_limit:
                        force_shear_lin = direction_trial * force_shear_lin_limit
                        force_shear_total = force_shear_lin
                    else:
                        force_shear_lin = force_shear_trial
                        force_shear_total = force_shear_lin + force_shear_damp

                    particle.forceShear[i, 0] += force_shear_total[0]
                    particle.forceShear[i, 1] += force_shear_total[1]
                    particle.forceShear[i, 2] += force_shear_total[2]
                    moment = distance_cp.cross(force_shear_total)
                    particle.moment[i, 0] += moment[0]
                    particle.moment[i, 1] += moment[1]
                    particle.moment[i, 2] += moment[2]

                self.forceShearWallXPre[i, j] = force_shear_lin[0]
                self.forceShearWallYPre[i, j] = force_shear_lin[1]
                self.forceShearWallZPre[i, j] = force_shear_lin[2]

