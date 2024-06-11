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
        self.stiffnessNormWall = ti.field(dtype=flt_dtype, shape=(1,))
        self.stiffnessNormWall[0] = stiff_n
        self.stiffnessShear = ti.field(dtype=flt_dtype, shape=(1,))
        self.stiffnessShear[0] = stiff_s
        self.stiffnessShearWall = ti.field(dtype=flt_dtype, shape=(1,))
        self.stiffnessShearWall[0] = stiff_s
        self.dampBallBallNorm = ti.field(dtype=flt_dtype, shape=(1,))
        self.dampBallBallNorm[0] = 0.5
        self.dampBallBallShear = ti.field(dtype=flt_dtype, shape=(1,))
        self.dampBallBallShear[0] = 0.4
        self.dampBallWallNorm = ti.field(dtype=flt_dtype, shape=(1,))
        self.dampBallWallNorm[0] = 0.7
        self.dampBallWallShear = ti.field(dtype=flt_dtype, shape=(1,))
        self.dampBallWallShear[0] = 0.4
        self.lenContactBallBallRecord = 16
        self.lenContactBallWallRecord = 6
        # id of particles in contact
        self.contacts = ti.field(dtype=ti.i32, shape=(self.n, self.lenContactBallBallRecord),
                                 name="contacts")
        # id of particles in contact in the last cycle
        self.contactsPre = ti.field(dtype=ti.i32, shape=(self.n, self.lenContactBallBallRecord),
                                    name="contacts_pre")
        # contact number on one particle
        self.contactCounter = ti.field(dtype=ti.i32, shape=(self.n,))
        self.contactDistX = ti.field(dtype=flt_dtype, shape=(self.n, self.lenContactBallBallRecord))
        self.contactDistY = ti.field(dtype=flt_dtype, shape=(self.n, self.lenContactBallBallRecord))
        self.contactDistZ = ti.field(dtype=flt_dtype, shape=(self.n, self.lenContactBallBallRecord))
        # shear force components
        self.forceShearX = ti.field(dtype=flt_dtype, shape=(self.n, self.lenContactBallBallRecord))
        self.forceShearY = ti.field(dtype=flt_dtype, shape=(self.n, self.lenContactBallBallRecord))
        self.forceShearZ = ti.field(dtype=flt_dtype, shape=(self.n, self.lenContactBallBallRecord))
        # normal force components
        self.force_n_x = ti.field(dtype=flt_dtype, shape=(self.n, self.lenContactBallBallRecord))
        self.force_n_y = ti.field(dtype=flt_dtype, shape=(self.n, self.lenContactBallBallRecord))
        self.force_n_z = ti.field(dtype=flt_dtype, shape=(self.n, self.lenContactBallBallRecord))
        # shear force component in the last cycle
        self.forceShearXPre = ti.field(dtype=flt_dtype, shape=(self.n, self.lenContactBallBallRecord))
        self.forceShearYPre = ti.field(dtype=flt_dtype, shape=(self.n, self.lenContactBallBallRecord))
        self.forceShearZPre = ti.field(dtype=flt_dtype, shape=(self.n, self.lenContactBallBallRecord))

        self.tangOverlapBallBallOldX = ti.field(dtype=flt_dtype, shape=(self.n, self.lenContactBallBallRecord))
        self.tangOverlapBallBallOldY = ti.field(dtype=flt_dtype, shape=(self.n, self.lenContactBallBallRecord))
        self.tangOverlapBallBallOldZ = ti.field(dtype=flt_dtype, shape=(self.n, self.lenContactBallBallRecord))

        self.tangOverlapBallWallOldX = ti.field(dtype=flt_dtype, shape=(self.n, self.lenContactBallWallRecord))
        self.tangOverlapBallWallOldY = ti.field(dtype=flt_dtype, shape=(self.n, self.lenContactBallWallRecord))
        self.tangOverlapBallWallOldZ = ti.field(dtype=flt_dtype, shape=(self.n, self.lenContactBallWallRecord))

        self.forceShearWallXPre = ti.field(dtype=flt_dtype, shape=(self.n, 6))
        self.forceShearWallYPre = ti.field(dtype=flt_dtype, shape=(self.n, 6))
        self.forceShearWallZPre = ti.field(dtype=flt_dtype, shape=(self.n, 6))

    def init_contact(self, dt):
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
                                gap = self.get_ball_ball_gap(gf, i, j)
                                if gap < 0:  # Particle in contact detected
                                    index_i, index_j = self.get_cur_col(i, j)
                                    index_pre = self.get_index_pre(i, j)
                                    normal = self.get_normal(gf, i, j)
                                    cpos = self.get_ball_ball_cpos(gf, i, j)
                                    self.resolve_ball_ball_force(
                                        gf, i, j, index_i, index_j, index_pre, gap, normal, cpos)
                                else:
                                    pass
                            else:
                                pass

    @ti.func
    def get_ball_ball_gap(self, gf: ti.template(), i: ti.i32, j: ti.i32) -> flt_dtype:
        """
        the gap between two particle is ([distance] - [sum of radii])
        :param gf: grain fields
        :param i: id of the first particle
        :param j: id of the second particle found in one of the neighboring grids
        :return: the gap between two particle. If the gap is < 0, the two particles are in
        contact.
        """
        pos1, pos2 = gf.get_pos(i), gf.get_pos(j)
        pos_rel = pos2 - pos1
        dist = self.get_magnitude(pos_rel)
        gap = dist - gf.rad[i] - gf.rad[j]  # gap = d - 2 * r
        return gap

    @ti.func
    def get_ball_wall_gap(self, gf: ti.template(), wall: ti.template(), i: ti.i32, j: ti.i32) -> flt_dtype:
        pos1, pos2 = gf.get_pos(i), wall.get_pos(j)
        pos_rel = pos1 - pos2
        normal = wall.get_normal(j)
        dist = pos_rel.dot(normal)
        gap = dist - gf.rad[i]
        return gap

    @ti.func
    def get_normal(self, gf: ti.template(), i: ti.i32, j: ti.i32) -> vec:
        pos1, pos2 = gf.get_pos(i), gf.get_pos(j)
        pos_rel = pos2 - pos1
        dist = self.get_magnitude(pos_rel)
        normal = pos_rel / dist
        return normal

    @ti.func
    def get_ball_ball_cpos(self, gf: ti.template(), i: ti.i32, j: ti.i32) -> vec:
        pos1, pos2 = gf.get_pos(i), gf.get_pos(j)
        normal = self.get_normal(gf, i, j)
        gap = self.get_ball_ball_gap(gf, i, j)
        cpos = pos1 + (gf.rad[i] + gap * 0.5) * normal
        return cpos

    @ti.func
    def get_ball_wall_cpos(self, gf: ti.template(), wall: ti.template(), i: ti.i32, j: ti.i32) -> vec:
        pos1, pos2 = gf.get_pos(i), wall.get_pos(j)
        normal = wall.get_normal(j)
        gap = self.get_ball_wall_gap(gf, wall, i, j)
        cpos = pos1 + (gf.rad[i] + gap * 0.5) * (-normal)
        return cpos

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
    def get_index(self, i: ti.i32, j: ti.i32) -> ti.i32:
        """
        Obtain the position of particle j in the contact list of particle i
        :param i: id of the first particle
        :param j: id of the other particle
        :return: None
        """
        index_cur = -1
        for index in range(self.lenContactBallBallRecord):
            if self.contacts[i, index] == j:
                index_cur = index
                break
            else:
                pass
        return index_cur

    @ti.func
    def get_index_pre(self, i: ti.i32, j: ti.i32) -> ti.i32:
        index_pre = -1
        for l in range(self.lenContactBallBallRecord):
            if self.contactsPre[i, l] == -1:
                break
            elif self.contactsPre[i, l] == j:
                index_pre = l
                break
            else:
                pass
        return index_pre

    @ti.func
    def get_magnitude(self, vector: vec) -> flt_dtype:
        """
        Obtain the magnitude of the vector
        :param n-dimensional vector:
        :return: magnitude of the vector
        """
        return ti.math.sqrt(vector[0] ** 2 + vector[1] ** 2 + vector[2] ** 2)

    @ti.func
    def normalize(self, force: vec) -> vec:
        res = vec(0.0, 0.0, 0.0)
        if force.norm() > 0.0:
            res = force / force.norm()
        else:
            res = vec(0.0, 0.0, 0.0)
        return res

    @ti.func
    def get_ball_ball_tang_overlap_old(self, i: ti.i32, index_i: ti.int32) -> vec:
        tang_overlap_old = vec(self.tangOverlapBallBallOldX[i, index_i],
                               self.tangOverlapBallBallOldY[i, index_i],
                               self.tangOverlapBallBallOldZ[i, index_i])
        return tang_overlap_old

    @ti.func
    def get_ball_wall_tang_overlap_old(self, i: ti.i32, index_i: ti.int32) -> vec:
        tang_overlap_old = vec(self.tangOverlapBallWallOldX[i, index_i],
                               self.tangOverlapBallWallOldY[i, index_i],
                               self.tangOverlapBallWallOldZ[i, index_i])
        return tang_overlap_old

    @ti.func
    def get_effective_value(self, value_1: flt_dtype, value_2: flt_dtype) -> flt_dtype:
        return value_1 * value_2 / (value_1 + value_2)

    @ti.func
    def record_ball_ball_shear_info(self, i: ti.i32, index_i: ti.i32, tangOverlap: vec):
        self.tangOverlapBallBallOldX[i, index_i] = tangOverlap[0]
        self.tangOverlapBallBallOldY[i, index_i] = tangOverlap[1]
        self.tangOverlapBallBallOldZ[i, index_i] = tangOverlap[2]

    @ti.func
    def record_ball_wall_shear_info(self, i: ti.i32, index_i: ti.i32, tangOverlap: vec):
        self.tangOverlapBallWallOldX[i, index_i] = tangOverlap[0]
        self.tangOverlapBallWallOldY[i, index_i] = tangOverlap[1]
        self.tangOverlapBallWallOldZ[i, index_i] = tangOverlap[2]

    @ti.func
    def resolve_ball_ball_force(self, particle: ti.template(), i: ti.i32, j: ti.i32, index_i: ti.i32, index_j: ti.i32,
                                index_pre: ti.i32, gap: flt_dtype, normal: vec, cpos: vec):
        """
        Transform shear force to the new contact plane
        :param particle: grain field
        :return: None
        """
        #######################################################################################
        #  Ball-ball force # Ball-ball force # Ball-ball force # Ball-ball force #Ball-ball   #
        #######################################################################################
        # in parallel
        pos1, pos2 = particle.get_pos(i), particle.get_pos(j)
        rad1, rad2 = particle.get_radius(i), particle.get_radius(j)
        mass1, mass2 = particle.get_mass(i), particle.get_mass(j)
        vel1, vel2 = particle.get_vel(i), particle.get_vel(j)
        w1, w2 = particle.get_vel_rot(i), particle.get_vel_rot(j)

        m_eff = self.get_effective_value(mass1, mass2)
        kn, ks = self.stiffnessNorm[0], self.stiffnessShear[0]
        ndratio, sdratio = self.dampBallBallNorm[0], self.dampBallBallShear[0]
        miu = self.frictionBallBall[0]

        v_rel = vel1 + w1.cross(cpos - pos1) - (vel2 + w2.cross(cpos - pos2))
        vn = v_rel.dot(normal)
        vs = v_rel - v_rel.dot(normal) * normal

        normal_contact_force = -kn * gap
        normal_damping_force = -2.0 * ndratio * ti.math.sqrt(m_eff * kn) * vn
        normal_force = (-normal_contact_force + normal_damping_force) * normal
        tangOverlapOld = vec(0.0, 0.0, 0.0)
        if index_pre != -1:
            tangOverlapOld = self.get_ball_ball_tang_overlap_old(i, index_pre)
        tangOverlapRot = tangOverlapOld - tangOverlapOld.dot(normal) * normal
        tangOverTemp = vs * self.dt + tangOverlapOld.norm() * self.normalize(tangOverlapRot)
        trial_ft = - ks * tangOverTemp
        tang_damping_force = - 2.0 * sdratio * ti.math.sqrt(m_eff * ks) * vs

        fric = miu * ti.abs(normal_contact_force + normal_damping_force)
        tangential_force = vec(0.0, 0.0, 0.0)

        if trial_ft.norm() > fric:
            tangential_force = fric * trial_ft.normalized()
            tangOverTemp = - tangential_force / ks
        else:
            tangential_force = trial_ft + tang_damping_force
        Ftotal = normal_force + tangential_force
        torque = tangential_force.cross(- normal)
        self.record_ball_ball_shear_info(i, index_i, tangOverTemp)
        self.record_ball_ball_shear_info(j, index_j, -tangOverTemp)
        particle.add_force_to_ball(i, Ftotal, torque * (rad1 + gap*0.5))
        particle.add_force_to_ball(j, -Ftotal, torque * (rad2 + gap*0.5))

    @ti.kernel
    def resolve_ball_wall_force(self, particle: ti.template(), wall: ti.template()):
        #######################################################################################
        #  Ball-wall force # Ball-wall force # Ball-wall force # Ball-wall force #Ball-wall   #
        #######################################################################################
        for i in range(particle.number):
            # Gap from the boundary in negative x direction:
            for j in range(wall.number):
                gap = self.get_ball_wall_gap(particle, wall, i, j)
                tangOverTemp = vec(0.0, 0.0, 0.0)
                if gap < 0.0:
                    pos1, pos2 = particle.get_pos(i), wall.get_pos(j)
                    rad1 = particle.get_radius(i)
                    vel1, vel2 = particle.get_vel(i), wall.get_vel(j)
                    w1 = particle.get_vel_rot(i)
                    normal = wall.get_normal(j)
                    m_eff = particle.get_mass(i)

                    kn, ks = self.stiffnessNormWall[0], self.stiffnessShearWall[0]
                    ndratio, sdratio = self.dampBallWallNorm[0], self.dampBallWallShear[0]
                    miu = self.frictionBallWall[0]

                    cpos = self.get_ball_wall_cpos(particle, wall, i, j)
                    v_rel = vel1 + w1.cross(cpos - pos1) - vel2
                    vn = v_rel.dot(normal)
                    vs = v_rel - v_rel.dot(normal) * normal

                    normal_contact_force = -kn * gap
                    normal_damping_force = -2.0 * ndratio * ti.math.sqrt(m_eff * kn) * vn
                    normal_force = (normal_contact_force + normal_damping_force) * normal

                    tangOverlapOld = self.get_ball_wall_tang_overlap_old(i, j)
                    tangOverlapRot = tangOverlapOld - tangOverlapOld.dot(normal) * normal
                    tangOverTemp = vs * self.dt + tangOverlapOld.norm() * self.normalize(tangOverlapRot)
                    trial_ft = -ks * tangOverTemp
                    tang_damping_force = -2.0 * sdratio * ti.math.sqrt(m_eff * ks) * vs

                    fric = miu * ti.abs(normal_contact_force + normal_damping_force)
                    tangential_force = vec(0.0, 0.0, 0.0)
                    if trial_ft.norm() > fric:
                        tangential_force = fric * trial_ft.normalized()
                        tangOverTemp = - tangential_force / ks
                    else:
                        tangential_force = trial_ft + tang_damping_force

                    Ftotal = normal_force + tangential_force
                    torque = tangential_force.cross(normal)
                    particle.add_force_to_ball(i, Ftotal, torque * (rad1 + gap*0.5))

                else:
                    pass

                self.record_ball_wall_shear_info(i, j, tangOverTemp)

