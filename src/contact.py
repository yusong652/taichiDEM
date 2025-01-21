import taichi as ti
from fmt import flt_dtype
vec = ti.math.vec3


@ti.data_oriented
class Contact(object):
    """
    # Allocate fields with fixed size for shear force information storage
    """

    def __init__(self, n, fric=0.5, fric_bw=0.0, stiff_n=5.0e7, stiff_s=2.5e7, model="linear"):
        self.n = n  # number of particles or rows for contact info storage
        self.model = model
        self.frictionBallBall = ti.field(dtype=flt_dtype, shape=(1,))
        self.frictionBallBall[0] = fric
        self.frictionBallWall = ti.field(dtype=flt_dtype, shape=(1,))
        self.frictionBallWall[0] = fric_bw
        self.stiffnessNorm = ti.field(dtype=flt_dtype, shape=(1,))
        self.stiffnessNorm[0] = stiff_n
        self.stiffnessNormWall = ti.field(dtype=flt_dtype, shape=(1,))
        self.stiffnessNormWall[0] = stiff_n
        self.stiffnessShear = ti.field(dtype=flt_dtype, shape=(1,))
        self.stiffnessShear[0] = stiff_s
        self.stiffnessShearWall = ti.field(dtype=flt_dtype, shape=(1,))
        self.stiffnessShearWall[0] = stiff_s
        self.effective_E = ti.field(dtype=flt_dtype, shape=(1,))
        self.effective_E[0] = 5.0e8
        self.effective_G = ti.field(dtype=flt_dtype, shape=(1,))
        self.effective_G[0] = 2.0e8
        self.dampBallBallNorm = ti.field(dtype=flt_dtype, shape=(1,))
        self.dampBallBallNorm[0] = 0.7
        self.dampBallBallShear = ti.field(dtype=flt_dtype, shape=(1,))
        self.dampBallBallShear[0] = 0.5
        self.dampBallWallNorm = ti.field(dtype=flt_dtype, shape=(1,))
        self.dampBallWallNorm[0] = 0.0
        self.dampBallWallShear = ti.field(dtype=flt_dtype, shape=(1,))
        self.dampBallWallShear[0] = 0.0
        self.lenContactBallBallRecord = 32
        self.lenContactBallWallRecord = 6
        # id of particles in contact
        self.contacts = ti.field(dtype=ti.i32, shape=(self.n, self.lenContactBallBallRecord),
                                 name="contacts")
        # id of particles in contact in the last cycle
        self.contactsPre = ti.field(dtype=ti.i32, shape=(self.n, self.lenContactBallBallRecord),
                                    name="contacts_pre")
        self.contactsBallWall = ti.field(dtype=ti.i32, shape=(self.n, self.lenContactBallWallRecord))
        self.contactsBallWallPre = ti.field(dtype=ti.i32, shape=(self.n, self.lenContactBallWallRecord))
        # contact number on one particle
        self.contactCounter = ti.field(dtype=ti.i32, shape=(self.n,))
        # force components
        self.forceX = ti.field(dtype=flt_dtype, shape=(self.n, self.lenContactBallBallRecord))
        self.forceY = ti.field(dtype=flt_dtype, shape=(self.n, self.lenContactBallBallRecord))
        self.forceZ = ti.field(dtype=flt_dtype, shape=(self.n, self.lenContactBallBallRecord))
        self.forceBallWallX = ti.field(dtype=flt_dtype, shape=(self.n, self.lenContactBallWallRecord))
        self.forceBallWallY = ti.field(dtype=flt_dtype, shape=(self.n, self.lenContactBallWallRecord))
        self.forceBallWallZ = ti.field(dtype=flt_dtype, shape=(self.n, self.lenContactBallWallRecord))
        # position
        self.positionX = ti.field(dtype=flt_dtype, shape=(self.n, self.lenContactBallBallRecord))
        self.positionY = ti.field(dtype=flt_dtype, shape=(self.n, self.lenContactBallBallRecord))
        self.positionZ = ti.field(dtype=flt_dtype, shape=(self.n, self.lenContactBallBallRecord))
        self.positionBallWallX = ti.field(dtype=flt_dtype, shape=(self.n, self.lenContactBallWallRecord))
        self.positionBallWallY = ti.field(dtype=flt_dtype, shape=(self.n, self.lenContactBallWallRecord))
        self.positionBallWallZ = ti.field(dtype=flt_dtype, shape=(self.n, self.lenContactBallWallRecord))
        self.tangOverlapBallBallOldX = ti.field(dtype=flt_dtype, shape=(self.n, self.lenContactBallBallRecord))
        self.tangOverlapBallBallOldY = ti.field(dtype=flt_dtype, shape=(self.n, self.lenContactBallBallRecord))
        self.tangOverlapBallBallOldZ = ti.field(dtype=flt_dtype, shape=(self.n, self.lenContactBallBallRecord))
        self.tangOverlapBallWallOldX = ti.field(dtype=flt_dtype, shape=(self.n, self.lenContactBallWallRecord))
        self.tangOverlapBallWallOldY = ti.field(dtype=flt_dtype, shape=(self.n, self.lenContactBallWallRecord))
        self.tangOverlapBallWallOldZ = ti.field(dtype=flt_dtype, shape=(self.n, self.lenContactBallWallRecord))
        self.dt = ti.field(dtype=flt_dtype, shape=(1,))

    def init_contact(self, dt):
        self.contacts.fill(-1)
        self.contactsPre.fill(-1)
        self.contactsBallWall.fill(-1)
        self.contactsBallWallPre.fill(-1)
        self.contactCounter.fill(0)
        self.dt[0] = dt
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
        for i, k in self.contactsBallWall:
            self.contactsBallWallPre[i, k] = self.contactsBallWall[i, k]
        self.contactsBallWall.fill(-1)

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
            # which grid it is located in. ((0, 1, 2) means
            # the 1st layer, 2nd row, and 3rd column)
            grid_idx = ti.math.floor(vec((gf.pos[i, 0] + gd.domain_size / 2) / gd.size_grid,
                                         (gf.pos[i, 1] + gd.domain_size / 2) / gd.size_grid,
                                         (gf.pos[i, 2] + gd.domain_size / 2) / gd.size_grid),
                                     int)  # which grid it is located in. ((0, 1, 2) means
            # the 1st layer, 2nd row, and 3rd column

            # If a particle with id-i is located in the grid, 1 is added to the grain-count
            # field.
            gd.grain_count[grid_idx] += 1  # atomic add by default

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

            # Search for the particles in the 27(=3*3*3) neighbor grids
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
                                    if self.model == "linear":
                                        self.resolve_ball_ball_force(
                                            gf, i, j, index_i, index_j, index_pre, gap, normal, cpos, 1)
                                    elif self.model == "hertz":
                                        self.resolve_ball_ball_force_hertz(
                                            gf, i, j, index_i, index_j, index_pre, gap, normal, cpos, 0)
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
        gap = dist - gf.get_radius(i)
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
    def calculate_effective_val(self, value_1: flt_dtype, value_2: flt_dtype) -> flt_dtype:
        return 2 * value_1 * value_2 / (value_1 + value_2)

    @ti.func
    def record_ball_ball_shear_info(self, i: ti.i32, index_i: ti.i32, tangOverlap: vec, force: vec, pos: vec):
        self.tangOverlapBallBallOldX[i, index_i] = tangOverlap[0]
        self.tangOverlapBallBallOldY[i, index_i] = tangOverlap[1]
        self.tangOverlapBallBallOldZ[i, index_i] = tangOverlap[2]
        self.forceX[i, index_i] = force[0]
        self.forceY[i, index_i] = force[1]
        self.forceZ[i, index_i] = force[2]
        self.positionX[i, index_i] = pos[0]
        self.positionY[i, index_i] = pos[1]
        self.positionZ[i, index_i] = pos[2]

    @ti.func
    def record_ball_wall_shear_info(self, i: ti.i32, index_i: ti.i32, index_wall: ti.i32, tangOverlap: vec, force: vec, pos: vec):
        self.contactsBallWall[i, index_i] = index_wall
        self.tangOverlapBallWallOldX[i, index_i] = tangOverlap[0]
        self.tangOverlapBallWallOldY[i, index_i] = tangOverlap[1]
        self.tangOverlapBallWallOldZ[i, index_i] = tangOverlap[2]
        self.forceBallWallX[i, index_i] = force[0]
        self.forceBallWallY[i, index_i] = force[1]
        self.forceBallWallZ[i, index_i] = force[2]
        self.positionBallWallX[i, index_i] = pos[0]
        self.positionBallWallY[i, index_i] = pos[1]
        self.positionBallWallZ[i, index_i] = pos[2]

    @ti.func
    def resolve_ball_ball_force(self, particle: ti.template(), i: ti.i32, j: ti.i32, index_i: ti.i32, index_j: ti.i32,
                                index_pre: ti.i32, gap: flt_dtype, normal: vec, cpos: vec, dp_mode: ti.i32):
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

        m_eff = self.calculate_effective_val(mass1, mass2)
        kn, ks = self.stiffnessNorm[0], self.stiffnessShear[0]
        ndratio, sdratio = self.dampBallBallNorm[0], self.dampBallBallShear[0]
        mu = self.frictionBallBall[0]
        mu_dynamic = 0.96 * mu

        v_rel = vel1 + w1.cross(cpos - pos1) - (vel2 + w2.cross(cpos - pos2))
        vn = v_rel.dot(normal)
        vs = v_rel - vn * normal

        normal_contact_force = -kn * gap
        normal_damping_force = -2.0 * ndratio * ti.math.sqrt(m_eff * kn) * vn
        if dp_mode == 1:
            normal_damping_force = ti.min(normal_damping_force, 0)
        normal_force = (-normal_contact_force + normal_damping_force) * normal
        tangOverlapOld = vec(0.0, 0.0, 0.0)
        if index_pre != -1:
            tangOverlapOld = self.get_ball_ball_tang_overlap_old(i, index_pre)
        tangOverlapRot = tangOverlapOld - tangOverlapOld.dot(normal) * normal
        tangOverTemp = vs * self.dt[0] + tangOverlapOld.norm() * self.normalize(tangOverlapRot)
        trial_ft = - ks * tangOverTemp
        tang_damping_force = - 2.0 * sdratio * ti.math.sqrt(m_eff * ks) * vs

        fric = mu * ti.abs(normal_contact_force - normal_damping_force)
        tangential_force = vec(0.0, 0.0, 0.0)

        if trial_ft.norm() > fric:
            tangential_force = mu_dynamic * ti.abs(normal_contact_force - normal_damping_force) * trial_ft.normalized()
            tangOverTemp = - tangential_force / ks
        else:
            tangential_force = trial_ft + tang_damping_force
        Ftotal = normal_force + tangential_force
        torque = tangential_force.cross(- normal)
        self.record_ball_ball_shear_info(i, index_i, tangOverTemp, Ftotal, cpos)
        self.record_ball_ball_shear_info(j, index_j, -tangOverTemp, -Ftotal, cpos )
        particle.add_force_to_ball(i, Ftotal, torque * (rad1 + gap*0.5))
        particle.add_force_to_ball(j, -Ftotal, torque * (rad2 + gap*0.5))

    @ti.func
    def resolve_ball_ball_force_hertz(self, particle: ti.template(), i: ti.i32, j: ti.i32, index_i: ti.i32, index_j: ti.i32,
        index_pre: ti.i32, gap: flt_dtype, normal: vec, cpos: vec, dp_mode: ti.i32):
        """
        Transform shear force to the new contact plane
        :param particle: grain field
        :return: None
        """
        #######################################################################################
        #  Ball-ball force # Ball-ball force # Ball-ball force # Ball-ball force #Ball-ball   #
        #######################################################################################
        # in parallel
        effective_E = 5.0e8
        effective_G = 2.5e8
        pos1, pos2 = particle.get_pos(i), particle.get_pos(j)
        rad1, rad2 = particle.get_radius(i), particle.get_radius(j)
        mass1, mass2 = particle.get_mass(i), particle.get_mass(j)
        vel1, vel2 = particle.get_vel(i), particle.get_vel(j)
        w1, w2 = particle.get_vel_rot(i), particle.get_vel_rot(j)

        m_eff = self.calculate_effective_val(mass1, mass2)
        rad_eff = self.calculate_effective_val(rad1, rad2)
        contactAreaRad = ti.math.sqrt(-gap * rad_eff)
        kn = 2.0 * effective_E * contactAreaRad
        ks = 8.0 * effective_G * contactAreaRad

        ndratio, sdratio = self.dampBallBallNorm[0], self.dampBallBallShear[0]
        mu = self.frictionBallBall[0]
        mu_dynamic = 0.99 * mu

        v_rel = vel1 + w1.cross(cpos - pos1) - (vel2 + w2.cross(cpos - pos2))
        vn = v_rel.dot(normal)
        vs = v_rel - vn * normal

        normal_contact_force = -2.0/3.0 * kn * gap
        normal_damping_force = -1.8257 * ndratio * ti.math.sqrt(m_eff * kn) * vn
        if dp_mode == 1:
            normal_damping_force = ti.min(normal_damping_force, 0)
        normal_force = (-normal_contact_force + normal_damping_force) * normal
        tangOverlapOld = vec(0.0, 0.0, 0.0)
        if index_pre != -1:
            tangOverlapOld = self.get_ball_ball_tang_overlap_old(i, index_pre)
        tangOverlapRot = tangOverlapOld - tangOverlapOld.dot(normal) * normal
        tangOverTemp = vs * self.dt[0] + tangOverlapOld.norm() * self.normalize(tangOverlapRot)
        trial_ft = - ks * tangOverTemp
        fric = mu * ti.abs(normal_contact_force - normal_damping_force)
        tangential_force = vec(0.0, 0.0, 0.0)
        if trial_ft.norm() > fric:
            tangential_force = mu_dynamic * ti.abs(normal_contact_force - normal_damping_force) * trial_ft.normalized()
            tangOverTemp = - tangential_force / ks
        else:
            tang_damping_force = - 1.8257 * sdratio * ti.math.sqrt(m_eff * ks) * vs
            tangential_force = trial_ft + tang_damping_force
            if tangential_force.norm() > fric:
                tangential_force = mu * ti.abs(normal_contact_force - normal_damping_force) * tangential_force.normalized()
        Ftotal = normal_force + tangential_force
        torque = tangential_force.cross(- normal)
        self.record_ball_ball_shear_info(i, index_i, tangOverTemp, Ftotal, cpos)
        self.record_ball_ball_shear_info(j, index_j, -tangOverTemp, -Ftotal, cpos)
        particle.add_force_to_ball(i, Ftotal, torque * (rad1 + gap*0.5))
        particle.add_force_to_ball(j, -Ftotal, torque * (rad2 + gap*0.5))

    @ti.kernel
    def resolve_ball_wall_force(self, particle: ti.template(), wall: ti.template(), dp_mode:ti.int32):
        #######################################################################################
        #  Ball-wall force # Ball-wall force # Ball-wall force # Ball-wall force #Ball-wall   #
        #######################################################################################
        for i in range(particle.number):
            # Gap from the boundary in negative x direction:
            for j in range(wall.number):
                gap = self.get_ball_wall_gap(particle, wall, i, j)
                index_wall = -1
                tangOverTemp = vec(0.0, 0.0, 0.0)
                Ftotal = vec(0.0, 0.0, 0.0)
                cpos = vec(0.0, 0.0, 0.0)
                if gap < 0.0:
                    index_wall = j
                    pos1, pos2 = particle.get_pos(i), wall.get_pos(j)
                    rad1 = particle.get_radius(i)
                    vel1, vel2 = particle.get_vel(i), wall.get_vel(j)
                    w1 = particle.get_vel_rot(i)
                    normal = wall.get_normal(j)
                    m_eff = particle.get_mass(i)

                    kn, ks = self.stiffnessNormWall[0], self.stiffnessShearWall[0]
                    ndratio, sdratio = self.dampBallWallNorm[0], self.dampBallWallShear[0]
                    mu = self.frictionBallWall[0]

                    cpos = self.get_ball_wall_cpos(particle, wall, i, j)
                    v_rel = vel1 + w1.cross(cpos - pos1) - vel2
                    vn = v_rel.dot(normal)
                    vs = v_rel - v_rel.dot(normal) * normal

                    normal_contact_force = -kn * gap
                    # collision damping
                    normal_damping_force = -2.0 * ndratio * ti.math.sqrt(m_eff * kn) * vn
                    if dp_mode == 1:
                        normal_damping_force = ti.max(normal_damping_force, 0)
                    normal_force = (normal_contact_force + normal_damping_force) * normal

                    tangOverlapOld = self.get_ball_wall_tang_overlap_old(i, j)
                    tangOverlapRot = tangOverlapOld - tangOverlapOld.dot(normal) * normal
                    tangOverTemp = vs * self.dt[0] + tangOverlapOld.norm() * self.normalize(tangOverlapRot)
                    trial_ft = -ks * tangOverTemp
                    tang_damping_force = -2.0 * sdratio * ti.math.sqrt(m_eff * ks) * vs

                    fric = mu * ti.abs(normal_contact_force + normal_damping_force)
                    tangential_force = vec(0.0, 0.0, 0.0)
                    if trial_ft.norm() > fric:
                        tangential_force = fric * trial_ft.normalized()
                        tangOverTemp = - tangential_force / ks
                    else:
                        tangential_force = trial_ft + tang_damping_force

                    Ftotal = normal_force + tangential_force
                    resultant_moment = Ftotal.cross(pos1 - cpos)
                    particle.add_force_to_ball(i, Ftotal, resultant_moment)
                    wall.add_contact_force(j, -Ftotal)
                    wall.add_contact_stiffness(j, kn)

                self.record_ball_wall_shear_info(i, j, index_wall, tangOverTemp, Ftotal, cpos)

    @ti.kernel
    def resolve_ball_wall_force_hertz(self, particle: ti.template(), wall: ti.template(), dp_mode:ti.i32):
        #######################################################################################
        #  Ball-wall force # Ball-wall force # Ball-wall force # Ball-wall force #Ball-wall   #
        #######################################################################################
        for i in range(particle.number):
            # Gap from the boundary in negative x direction:
            for j in range(wall.number):
                gap = self.get_ball_wall_gap(particle, wall, i, j)
                index_wall = -1
                tangOverTemp = vec(0.0, 0.0, 0.0)
                Ftotal = vec(0.0, 0.0, 0.0)
                cpos = vec(0.0, 0.0, 0.0)
                if gap < 0.0:
                    index_wall = j
                    effective_E = 5.0e8
                    effective_G = 2.5e8
                    pos1, pos2 = particle.get_pos(i), wall.get_pos(j)
                    rad1 = particle.get_radius(i)
                    # vel1, vel2 = particle.get_vel(i), wall.get_vel(j)
                    vel1 = particle.get_vel(i)
                    # w1 = particle.get_vel_rot(i)
                    # normal = wall.get_normal(j)
                    # m_eff = particle.get_mass(i)
                    # rad_eff = rad1
                    # contactAreaRad = ti.math.sqrt(-gap * rad_eff)
                    # ndratio, sdratio = self.dampBallWallNorm[0], self.dampBallWallShear[0]
                    # mu = self.frictionBallWall[0]
                    # mu_dynamic = 0.99 * mu

                    # kn = 2 * effective_E * contactAreaRad
                    # ks = 8 * effective_G * contactAreaRad
                    # cpos = self.get_ball_wall_cpos(particle, wall, i, j)
                    # v_rel = vel1 + w1.cross(cpos - pos1) - vel2
                    # vn = v_rel.dot(normal)
                    # vs = v_rel - v_rel.dot(normal) * normal

                #     normal_contact_force = -2./3. * kn * gap
                #     # collision damping
                #     normal_damping_force = -1.8257 * ndratio * ti.math.sqrt(m_eff * kn) * vn
                #     if dp_mode == 1:
                #         normal_damping_force = ti.max(normal_damping_force, 0)
                #     normal_force = (normal_contact_force + normal_damping_force) * normal

                #     tangOverlapOld = self.get_ball_wall_tang_overlap_old(i, j)
                #     tangOverlapRot = tangOverlapOld - tangOverlapOld.dot(normal) * normal
                #     tangOverTemp = vs * self.dt[0] + tangOverlapOld.norm() * self.normalize(tangOverlapRot)
                #     trial_ft = -ks * tangOverTemp

                #     fric = mu * ti.abs(normal_contact_force + normal_damping_force)
                #     tangential_force = vec(0.0, 0.0, 0.0)
                #     if trial_ft.norm() > fric:
                #         tangential_force = mu_dynamic * ti.abs(normal_contact_force + normal_damping_force) * trial_ft.normalized()
                #         tangOverTemp = - tangential_force / ks
                #     else:
                #         tang_damping_force = -1.8257 * sdratio * ti.math.sqrt(m_eff * ks) * vs
                #         tangential_force = trial_ft + tang_damping_force
                #         if tangential_force.norm() > fric:
                #             tangential_force = mu * ti.abs(normal_contact_force + normal_damping_force) * tangential_force.normalized()

                #     Ftotal = normal_force + tangential_force
                #     resultant_moment = Ftotal.cross(pos1 - cpos)
                #     particle.add_force_to_ball(i, Ftotal, resultant_moment)
                #     wall.add_contact_force(j, -Ftotal)
                #     wall.add_contact_stiffness(j, kn)
                # self.record_ball_wall_shear_info(i, j, index_wall, tangOverTemp, Ftotal, cpos)

                
