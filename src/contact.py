import taichi as ti
import numpy as np

vec = ti.math.vec3


@ti.data_oriented
class ContactInfo(object):
    """
    # Allocate fields with fixed size for shear force information storage
    """

    def __init__(self, n, fric=0.5, stiff_n=5.0e7, stiff_s=1.0e7, ):
        self.n = n  # number of particles or rows for contact info storage
        self.fric = ti.field(dtype=ti.f64, shape=(1,))
        self.fric[0] = fric
        self.stiff_n = ti.field(dtype=ti.f64, shape=(1,))
        self.stiff_n[0] = stiff_n
        self.stiff_s = ti.field(dtype=ti.f64, shape=(1,))
        self.stiff_s[0] = stiff_s
        self.damp_bb_n = ti.field(dtype=ti.f64, shape=(1,))
        self.damp_bb_n[0] = 0.7
        self.damp_bb_s = ti.field(dtype=ti.f64, shape=(1,))
        self.damp_bb_s[0] = 0.2
        self.damp_wb = ti.field(dtype=ti.f64, shape=(1,))
        self.damp_wb[0] = 0.3
        self.con_rec_len = 64
        # id of particles in contact
        self.contacts = ti.field(dtype=ti.i32, shape=(self.n, self.con_rec_len),
                                 name="contacts")
        # id of particles in contact in the last cycle
        self.contacts_pre = ti.field(dtype=ti.i32, shape=(self.n, self.con_rec_len),
                                     name="contacts_pre")
        # contact number on one particle
        self.contact_count = ti.field(dtype=ti.i32, shape=(self.n,))
        self.contact_dist_x = ti.field(dtype=ti.f64, shape=(self.n, self.con_rec_len))
        self.contact_dist_y = ti.field(dtype=ti.f64, shape=(self.n, self.con_rec_len))
        self.contact_dist_z = ti.field(dtype=ti.f64, shape=(self.n, self.con_rec_len))
        # shear force components
        self.force_s_x = ti.field(dtype=ti.f64, shape=(self.n, self.con_rec_len))
        self.force_s_y = ti.field(dtype=ti.f64, shape=(self.n, self.con_rec_len))
        self.force_s_z = ti.field(dtype=ti.f64, shape=(self.n, self.con_rec_len))
        # normal force components
        self.force_n_x = ti.field(dtype=ti.f64, shape=(self.n, self.con_rec_len))
        self.force_n_y = ti.field(dtype=ti.f64, shape=(self.n, self.con_rec_len))
        self.force_n_z = ti.field(dtype=ti.f64, shape=(self.n, self.con_rec_len))
        # shear force component in the last cycle
        self.force_s_x_pre = ti.field(dtype=ti.f64, shape=(self.n, self.con_rec_len))
        self.force_s_y_pre = ti.field(dtype=ti.f64, shape=(self.n, self.con_rec_len))
        self.force_s_z_pre = ti.field(dtype=ti.f64, shape=(self.n, self.con_rec_len))
        # DEBUG Mode *********************************
        self.acc_disp = ti.field(dtype=ti.f64, shape=(1,))
        self.acc_force = ti.field(dtype=ti.f64, shape=(1,))
        # DEBUG Mode *********************************

    def init_contact(self, dt, gf, gd):
        self.contacts.fill(-1)
        self.contacts_pre.fill(-1)
        self.contact_count.fill(0)
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
            self.contacts_pre[i, k] = self.contacts[i, k]
        # Clear the current contact field (size: num_particle * record_len.
        # record_len=16 by default)
        self.contacts.fill(-1)  # Initialize in every cycle
        # Zero the counting list (size: num_particle * 1)
        self.contact_count.fill(0)  # Initialize

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
        for i in range(gf.num_ptc):
            grid_idx = ti.math.floor(vec((gf.pos[i, 0] + gd.domain_size / 2) / gd.size_grid,
                                         (gf.pos[i, 1] + gd.domain_size / 2) / gd.size_grid,
                                         (gf.pos[i, 2] + gd.domain_size / 2) / gd.size_grid),
                                     int)  # which grid it is located in. ((0, 1, 2) means
            # the 1st layer, 2nd row, and 3rd column)
            # DEBUG Mode ********************************************************************
            # if grid_idx[0] < 0:
            #     print(gf.pos[i, 0], gf.pos[i, 1], gf.pos[i, 2], i)
            # print(gf.pos[3, 0])
            # DEBUG Mode ********************************************************************
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
                                    self.get_contact(i, j)
                                else:
                                    pass
                            else:
                                pass

    @ti.func
    def get_gap(self, gf: ti.template(), i: ti.i32, j: ti.i32) -> ti.f64:
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
        # DEBUG Mode ******************************************
        # print(dist)
        # DEBUG Mode ******************************************
        gap = dist - gf.rad[i] - gf.rad[j]  # gap = d - 2 * r
        return gap

    @ti.func
    def get_contact(self, i: ti.i32, j: ti.i32):
        """
        Append the id of particles to the contact field
        :param i: id of the particle
        :param j: id of the other particle
        :return: None
        """
        # Obtain the index of current column in the contact list
        index_i = ti.atomic_add(self.contact_count[i], 1)  # Get the value and add 1 to it
        # Put the id of another particle to the corresponding position:
        self.contacts[i, index_i] = j
        # Index of the column for another particle
        index_j = ti.atomic_add(self.contact_count[j], 1)
        self.contacts[j, index_j] = i

    @ti.kernel
    def get_force_normal(self, gf: ti.template()):
        """
        Obtain the normal force based on the overlapping length (-gap)
        :param gf: grain field
        :return:
        """
        #######################################################################################
        # Normal force # Normal force # Normal force # Normal force # Normal force # Normal F #
        #######################################################################################
        # Obtain the relative position (vec3)
        for i in range(self.n):
            for k in range(self.con_rec_len):
                if self.contacts[i, k] != -1:
                    if i < self.contacts[i, k]:
                        j = self.contacts[i, k]
                        rel_pos = vec(gf.pos[j, 0] - gf.pos[i, 0],
                                      gf.pos[j, 1] - gf.pos[i, 1],
                                      gf.pos[j, 2] - gf.pos[i, 2])
                        # Obtain the relative velocity (vec3)
                        rel_vel = vec(gf.vel[i, 0] - gf.vel[j, 0],
                                      gf.vel[i, 1] - gf.vel[j, 1],
                                      gf.vel[i, 2] - gf.vel[j, 2])
                        # Distance between two particle
                        dist = ti.sqrt(rel_pos[0] ** 2 + rel_pos[1] ** 2 + rel_pos[2] ** 2)
                        gap = dist - gf.rad[i] - gf.rad[j]  # gap = d - 2 * r
                        # Normalize the direction
                        normal = rel_pos / dist
                        gf.force_n[i, 0] += normal[0] * gap * self.stiff_n[0]
                        gf.force_n[i, 1] += normal[1] * gap * self.stiff_n[0]
                        gf.force_n[i, 2] += normal[2] * gap * self.stiff_n[0]
                        gf.force_n[j, 0] -= normal[0] * gap * self.stiff_n[0]
                        gf.force_n[j, 1] -= normal[1] * gap * self.stiff_n[0]
                        gf.force_n[j, 2] -= normal[2] * gap * self.stiff_n[0]
                        # Damping force
                        # Only collision direction considered (no tension damping force)
                        # if ti.math.dot(rel_pos, rel_vel) > 0:
                        M = (gf.mass[i] * gf.mass[j]) / (gf.mass[i] + gf.mass[j])
                        K = self.stiff_n[0]
                        C = 2. * self.damp_bb_n[0] * ti.sqrt(K * M)
                        V = ti.math.dot(rel_vel, normal)
                        gf.force_n[i, 0] += -C * V * normal[0]
                        gf.force_n[i, 1] += -C * V * normal[1]
                        gf.force_n[i, 2] += -C * V * normal[2]
                        gf.force_n[j, 0] -= -C * V * normal[0]
                        gf.force_n[j, 1] -= -C * V * normal[1]
                        gf.force_n[j, 2] -= -C * V * normal[2]
                        # else:  # Moving apart results no damping force
                        #     pass
                    else:  # i is smaller than j.
                        pass
                else:  # the contact id after k should be -1
                    break

    @ti.func
    def get_index(self, i: ti.i32, j: ti.i32) -> ti.i32:
        """
        Obtain the position of particle j in the contact list of particle i
        :param i: id of the first particle
        :param j: id of the other particle
        :return: None
        """
        index_ini = -1
        for index in range(self.con_rec_len):
            if self.contacts[i, index] == j:
                index_ini = index
                break
            else:
                pass
        return index_ini

    @ti.kernel
    def get_force_shear(self, gf: ti.template()):
        """
        Transform shear force to the new contact plane
        :param gf: grain field
        :return: None
        """
        #######################################################################################
        # Shear force # Shear force # Shear force # Shear force # Shear force # Shear Force   #
        #######################################################################################
        for i in range(self.n):  # The i-th row
            for index_i in range(self.con_rec_len):  # The index_i-th column
                if self.contacts[i, index_i] != -1:  # Particle detected in the detection
                    if i < self.contacts[i, index_i]:
                        j = self.contacts[i, index_i]
                        index_j = self.get_index(j, i)
                        # Obtain the relative position (vec3)
                        rel_pos = vec(gf.pos[j, 0] - gf.pos[i, 0],
                                      gf.pos[j, 1] - gf.pos[i, 1],
                                      gf.pos[j, 2] - gf.pos[i, 2])
                        # Obtain the relative velocity (vec3)
                        rel_vel = vec(gf.vel[i, 0] - gf.vel[j, 0],
                                      gf.vel[i, 1] - gf.vel[j, 1],
                                      gf.vel[i, 2] - gf.vel[j, 2])
                        # Distance between two particle (Scalar)
                        dist = ti.sqrt(rel_pos[0] ** 2 + rel_pos[1] ** 2 + rel_pos[2] ** 2)
                        gap = dist - gf.rad[i] - gf.rad[j]  # gap = d - 2 * r
                        # Normalize the direction (Normalized vec3)
                        normal = rel_pos / dist
                        # Distance of contact point from particle centroid
                        self.contact_dist_x[i, index_i] = (gf.rad[i] + gap / 2.0) * normal[0]
                        self.contact_dist_y[i, index_i] = (gf.rad[i] + gap / 2.0) * normal[1]
                        self.contact_dist_z[i, index_i] = (gf.rad[i] + gap / 2.0) * normal[2]
                        self.contact_dist_x[j, index_j] = - (gf.rad[i] + gap / 2.0) * normal[0]
                        self.contact_dist_y[j, index_j] = - (gf.rad[i] + gap / 2.0) * normal[1]
                        self.contact_dist_z[j, index_j] = - (gf.rad[i] + gap / 2.0) * normal[2]

                        # Index of contact between particle i and j
                        # index_pre = -1 by default
                        # i = -1 after resolution indicates no contact at previous time step
                        # or an initial state.
                        index_pre = -1
                        gap_pre = (ti.math.sqrt((gf.pos_pre[i, 0] - gf.pos_pre[j, 0]) ** 2 +
                                                (gf.pos_pre[i, 1] - gf.pos_pre[j, 1]) ** 2 +
                                                (gf.pos_pre[i, 2] - gf.pos_pre[j, 2]) ** 2) -
                                   (gf.rad[i] + gf.rad[j]))

                        if gap_pre < 0.0:
                            for l in range(self.con_rec_len):
                                if self.contacts_pre[i, l] == j:
                                    index_pre = l
                                    break
                                else:
                                    pass
                            # Previous shear force transformation by using a quaternion
                            if index_pre != -1:  # The particles were not in contact at the la-
                                # st cycle
                                # print("before: ", self.force_s_x_pre[0, 0],
                                #       self.force_s_y_pre[0, 0],
                                #       self.force_s_z_pre[0, 0])
                                proj_pre = (self.force_s_x_pre[i, index_pre] * normal[0] +
                                            self.force_s_y_pre[i, index_pre] * normal[1] +
                                            self.force_s_z_pre[i, index_pre] * normal[2])

                                force_transformed = vec(self.force_s_x_pre[i, index_pre] -
                                                        normal[0] * proj_pre,
                                                        self.force_s_y_pre[i, index_pre] -
                                                        normal[1] * proj_pre,
                                                        self.force_s_z_pre[i, index_pre] -
                                                        normal[2] * proj_pre)

                                self.force_s_x[i, index_i] = force_transformed[0]
                                self.force_s_y[i, index_i] = force_transformed[1]
                                self.force_s_z[i, index_i] = force_transformed[2]
                                self.force_s_x[j, index_j] = - self.force_s_x[i, index_i]
                                self.force_s_y[j, index_j] = - self.force_s_y[i, index_i]
                                self.force_s_z[j, index_j] = - self.force_s_z[i, index_i]
                                # print("after: ", self.force_s_x[0, 0],
                                #       self.force_s_y[0, 0],
                                #       self.force_s_z[0, 0])

                            else:
                                # Initial state
                                self.force_s_x[i, index_i] = 0.0
                                self.force_s_y[i, index_i] = 0.0
                                self.force_s_z[i, index_i] = 0.0
                                self.force_s_x[j, index_j] = 0.0
                                self.force_s_y[j, index_j] = 0.0
                                self.force_s_z[j, index_j] = 0.0
                        else:
                            """
                            The contact is initiated and the shear disp increment should 
                            be derived from the previous velocity and rotational velocity
                            of thr two particles
                            """
                            rel_pos_pre = vec(gf.pos_pre[j, 0] - gf.pos_pre[i, 0],
                                              gf.pos_pre[j, 1] - gf.pos_pre[i, 1],
                                              gf.pos_pre[j, 2] - gf.pos_pre[i, 2])
                            # Obtain the relative velocity (vec3)
                            rel_vel_pre = vec(gf.vel_pre[i, 0] - gf.vel_pre[j, 0],
                                              gf.vel_pre[i, 1] - gf.vel_pre[j, 1],
                                              gf.vel_pre[i, 2] - gf.vel_pre[j, 2])
                            # Distance between two particle (Scalar)
                            dist_pre = ti.sqrt(rel_pos_pre[0] ** 2 +
                                               rel_pos_pre[1] ** 2 +
                                               rel_pos_pre[2] ** 2)
                            gap_pre = dist_pre - gf.rad[i] - gf.rad[j]  # gap = d - 2 * r
                            # Normalize the direction (Normalized vec3)
                            normal_pre = rel_pos_pre / dist_pre
                            contact_dist_x_pre_i = (gf.rad[i] + gap_pre / 2.0) * normal_pre[0]
                            contact_dist_y_pre_i = (gf.rad[i] + gap_pre / 2.0) * normal_pre[1]
                            contact_dist_z_pre_i = (gf.rad[i] + gap_pre / 2.0) * normal_pre[2]
                            contact_dist_x_pre_j = - (gf.rad[i] + gap_pre / 2.0) * \
                                                   normal_pre[0]
                            contact_dist_y_pre_j = - (gf.rad[i] + gap_pre / 2.0) * \
                                                   normal_pre[0]
                            contact_dist_z_pre_j = - (gf.rad[i] + gap_pre / 2.0) * \
                                                   normal_pre[0]
                            rel_disp_pre = rel_vel_pre * self.dt
                            disp_inc_lin_pre = rel_disp_pre - ti.math.dot(
                                rel_disp_pre, normal_pre) * normal_pre
                            # The first particle
                            disp_inc_rot_pre_i = vec(gf.vel_rot_pre[i, 1] *
                                                     contact_dist_z_pre_i -
                                                     gf.vel_rot_pre[i, 2] *
                                                     contact_dist_y_pre_i,
                                                     gf.vel_rot_pre[i, 2] *
                                                     contact_dist_x_pre_i -
                                                     gf.vel_rot_pre[i, 0] *
                                                     contact_dist_z_pre_i,
                                                     gf.vel_rot_pre[i, 0] *
                                                     contact_dist_y_pre_i -
                                                     gf.vel_rot_pre[i, 1] *
                                                     contact_dist_x_pre_i) * self.dt

                            # The second particle
                            disp_inc_rot_pre_j = vec(gf.vel_rot_pre[j, 1] *
                                                     contact_dist_z_pre_j -
                                                     gf.vel_rot_pre[j, 2] *
                                                     contact_dist_y_pre_j,
                                                     gf.vel_rot_pre[j, 2] *
                                                     contact_dist_x_pre_j -
                                                     gf.vel_rot_pre[j, 0] *
                                                     contact_dist_z_pre_j,
                                                     gf.vel_rot_pre[j, 0] *
                                                     contact_dist_y_pre_j -
                                                     gf.vel_rot_pre[j, 1] *
                                                     contact_dist_x_pre_j) * self.dt
                            disp_inc_rot_pre = disp_inc_rot_pre_i - disp_inc_rot_pre_j
                            disp_inc_pre = disp_inc_lin_pre + disp_inc_rot_pre
                            ratio = -gap / (-gap + gap_pre)
                            disp_inc_pre = disp_inc_pre * ratio
                            f_s_inc_pre = - disp_inc_pre * self.stiff_s[0]
                            proj_pre = (f_s_inc_pre[0] * normal[0] +
                                        f_s_inc_pre[1] * normal[1] +
                                        f_s_inc_pre[2] * normal[2])

                            force_transformed = vec(f_s_inc_pre[0] -
                                                    normal[0] * proj_pre,
                                                    f_s_inc_pre[1] -
                                                    normal[1] * proj_pre,
                                                    f_s_inc_pre[2] -
                                                    normal[2] * proj_pre)
                            self.force_s_x[i, index_i] = force_transformed[0]
                            self.force_s_y[i, index_i] = force_transformed[1]
                            self.force_s_z[i, index_i] = force_transformed[2]
                            self.force_s_x[j, index_j] = - force_transformed[0]
                            self.force_s_y[j, index_j] = - force_transformed[1]
                            self.force_s_z[j, index_j] = - force_transformed[2]
                        """
                        damping force
                        """
                        # Linear relative velocity induced part:
                        rel_disp = rel_vel * self.dt
                        disp_inc_lin = rel_disp - ti.math.dot(rel_disp, normal) * normal

                        # Rotational relative velocity induced part:
                        # Obtain the rotation induced displacement with cross product of rotat-
                        # ional velocity and contact distance
                        # The first particle
                        disp_inc_rot_i = vec(gf.vel_rot[i, 1] *
                                             self.contact_dist_z[i, index_i] -
                                             gf.vel_rot[i, 2] *
                                             self.contact_dist_y[i, index_i],
                                             gf.vel_rot[i, 2] *
                                             self.contact_dist_x[i, index_i] -
                                             gf.vel_rot[i, 0] *
                                             self.contact_dist_z[i, index_i],
                                             gf.vel_rot[i, 0] *
                                             self.contact_dist_y[i, index_i] -
                                             gf.vel_rot[i, 1] *
                                             self.contact_dist_x[i, index_i]) * self.dt

                        # The second particle
                        disp_inc_rot_j = vec(gf.vel_rot[j, 1] *
                                             self.contact_dist_z[j, index_j] -
                                             gf.vel_rot[j, 2] *
                                             self.contact_dist_y[j, index_j],
                                             gf.vel_rot[j, 2] *
                                             self.contact_dist_x[j, index_j] -
                                             gf.vel_rot[j, 0] *
                                             self.contact_dist_z[j, index_j],
                                             gf.vel_rot[j, 0] *
                                             self.contact_dist_y[j, index_j] -
                                             gf.vel_rot[j, 1] *
                                             self.contact_dist_x[j, index_j]) * self.dt

                        disp_inc_rot = disp_inc_rot_i - disp_inc_rot_j
                        disp_inc = disp_inc_lin + disp_inc_rot
                        # print("disp: ", disp_inc)
                        # DEBUG
                        # print(disp_inc)
                        # Damping shear force
                        rel_vel_s = disp_inc / self.dt
                        M = (gf.mass[i] * gf.mass[j]) / (gf.mass[i] + gf.mass[j])
                        K_s = self.stiff_s[0]
                        C_s = 2. * self.damp_bb_s[0] * ti.sqrt(K_s * M)
                        f_s_damp = -rel_vel_s * C_s

                        force_s_mod = ti.math.sqrt(self.force_s_x[i, index_i] ** 2 +
                                                   self.force_s_y[i, index_i] ** 2 +
                                                   self.force_s_z[i, index_i] ** 2)

                        force_n_mod = - gap * self.stiff_n[0]
                        force_s_mod_lim = force_n_mod * self.fric[0]  # Coulomb limit
                        # print(force_s_mod_lim/force_s_mod*force_s_mod)
                        ratio_f_s = 1.0
                        if force_s_mod > force_s_mod_lim:
                            ratio_f_s = force_s_mod_lim / force_s_mod
                        else:
                            pass
                        self.force_s_x[i, index_i] *= ratio_f_s
                        self.force_s_y[i, index_i] *= ratio_f_s
                        self.force_s_z[i, index_i] *= ratio_f_s
                        self.force_s_x[j, index_j] *= ratio_f_s
                        self.force_s_y[j, index_j] *= ratio_f_s
                        self.force_s_z[j, index_j] *= ratio_f_s

                        f_s = vec(self.force_s_x[i, index_i],
                                  self.force_s_y[i, index_i],
                                  self.force_s_z[i, index_i]) + f_s_damp
                        # print("id of ptc: ", i)
                        # print("disp_by_lin: ", disp_inc_lin)
                        # print("disp_by_rot: ", disp_inc_rot)
                        # print("damping force:", f_s_damp)
                        # Force and moment sum

                        gf.force_s[i, 0] += f_s[0]
                        gf.force_s[i, 1] += f_s[1]
                        gf.force_s[i, 2] += f_s[2]
                        gf.force_s[j, 0] -= f_s[0]
                        gf.force_s[j, 1] -= f_s[1]
                        gf.force_s[j, 2] -= f_s[2]
                        # DEBUG Mode **********************************************************
                        # print(gf.force_s[0, 1])
                        # DEBUG Mode **********************************************************
                        moment_i = vec(self.contact_dist_y[i, index_i] * f_s[2] -
                                       self.contact_dist_z[i, index_i] * f_s[1],
                                       self.contact_dist_z[i, index_i] * f_s[0] -
                                       self.contact_dist_x[i, index_i] * f_s[2],
                                       self.contact_dist_x[i, index_i] * f_s[1] -
                                       self.contact_dist_y[i, index_i] * f_s[0]
                                       )
                        gf.moment[i, 0] += moment_i[0]
                        gf.moment[i, 1] += moment_i[1]
                        gf.moment[i, 2] += moment_i[2]
                        # DEBUG Mode **********************************************************
                        # print(gf.f[0, 1] / gf.moment[0, 0])
                        # print(gf.moment[0, 0])
                        # DEBUG Mode **********************************************************
                        moment_j = vec(self.contact_dist_y[j, index_j] * (-f_s[2]) -
                                       self.contact_dist_z[j, index_j] * (-f_s[1]),
                                       self.contact_dist_z[j, index_j] * (-f_s[0]) -
                                       self.contact_dist_x[j, index_j] * (-f_s[2]),
                                       self.contact_dist_x[j, index_j] * (-f_s[1]) -
                                       self.contact_dist_y[j, index_j] * (-f_s[0])
                                       )
                        gf.moment[j, 0] += moment_j[0]
                        gf.moment[j, 1] += moment_j[1]
                        gf.moment[j, 2] += moment_j[2]

                    else:
                        pass  # i > j, pass to avoid double count
                else:
                    break  # -1 detected and there is no contact after this column

    @ti.kernel
    def get_force_shear_inc(self, gf: ti.template()):
        #######################################################################################
        # Shear force increment # Shear force increment # Shear force increment # Shear force #
        #######################################################################################
        for i in range(self.n):
            for index_i in range(self.con_rec_len):
                if self.contacts[i, index_i] != -1:
                    if i > self.contacts[i, index_i]:
                        j = self.contacts[i, index_i]
                        index_j = self.get_index(j, i)
                        # Obtain the relative position (vec3)
                        rel_pos = vec(gf.pos[j, 0] - gf.pos[i, 0],
                                      gf.pos[j, 1] - gf.pos[i, 1],
                                      gf.pos[j, 2] - gf.pos[i, 2])
                        # Obtain the relative velocity (vec3)
                        rel_vel = vec(gf.vel[i, 0] - gf.vel[j, 0],
                                      gf.vel[i, 1] - gf.vel[j, 1],
                                      gf.vel[i, 2] - gf.vel[j, 2])
                        # Distance between two particle (Scalar)
                        dist = ti.sqrt(rel_pos[0] ** 2 + rel_pos[1] ** 2 + rel_pos[2] ** 2)
                        gap = dist - gf.rad[i] - gf.rad[j]  # gap = d - 2 * r
                        # Normalize the direction (Normalized vec3)
                        normal = rel_pos / dist

                        """
                        [Increment of shear force]
                        The increment of shear force is induced by relative velocity and relat-
                        ive rotational velocity. 
                        """
                        # Linear relative velocity induced part:
                        rel_disp = rel_vel * self.dt
                        disp_inc_lin = rel_disp - ti.math.dot(rel_disp, normal) * normal

                        # Rotational relative velocity induced part:
                        # Obtain the rotation induced displacement with cross product of rotat-
                        # ional velocity and contact distance
                        # The first particle
                        # print(gf.vel_rot[0, 0], gf.vel_rot[0, 1], gf.vel_rot[0, 2])
                        disp_inc_rot_i = vec(gf.vel_rot[i, 1] *
                                             self.contact_dist_z[i, index_i] -
                                             gf.vel_rot[i, 2] *
                                             self.contact_dist_y[i, index_i],
                                             gf.vel_rot[i, 2] *
                                             self.contact_dist_x[i, index_i] -
                                             gf.vel_rot[i, 0] *
                                             self.contact_dist_z[i, index_i],
                                             gf.vel_rot[i, 0] *
                                             self.contact_dist_y[i, index_i] -
                                             gf.vel_rot[i, 1] *
                                             self.contact_dist_x[i, index_i]) * self.dt

                        # The second particle
                        disp_inc_rot_j = vec(gf.vel_rot[j, 1] *
                                             self.contact_dist_z[j, index_j] -
                                             gf.vel_rot[j, 2] *
                                             self.contact_dist_y[j, index_j],
                                             gf.vel_rot[j, 2] *
                                             self.contact_dist_x[j, index_j] -
                                             gf.vel_rot[j, 0] *
                                             self.contact_dist_z[j, index_j],
                                             gf.vel_rot[j, 0] *
                                             self.contact_dist_y[j, index_j] -
                                             gf.vel_rot[j, 1] *
                                             self.contact_dist_x[j, index_j]) * self.dt

                        disp_inc_rot = disp_inc_rot_i - disp_inc_rot_j
                        disp_inc = disp_inc_lin + disp_inc_rot
                        # # DEBUG Mode ********************************************************
                        # print(disp_inc[1])
                        # print(gf.v_rot[0, 0] * 0.095 * 2 * self.dt)
                        # self.acc_disp[0] += disp_inc[1]
                        # print(disp_inc[1]/gf.v_rot[i, 0]/self.dt/2.0)  # precision loss
                        # print(disp_inc_rot_i)
                        # print(disp_inc_rot_j)
                        # # DEBUG Mode ********************************************************
                        # if gap_pre > 0:
                        # #     disp_inc = -gap / (gap_pre-gap) * disp_inc
                        # else:
                        #     pass
                        f_s_inc = - disp_inc * self.stiff_s[0]
                        # DEBUG Mode **********************************************************
                        # print("f_s_inc: ", f_s_inc)
                        # print(disp_inc)
                        # print("disp_inc_rot: ", disp_inc_rot)
                        # print("disp_int_lin: ", disp_inc_lin)
                        # self.acc_force[0] += f_s_inc[1]
                        # print(gf.f[0, 1], self.acc_force[0])
                        # print(self.acc_force[0], self.force_s_y[0, 0])
                        # print(self.force_s_y[0, 0], self.force_s_y_post[0, 0])
                        # DEBUG Mode **********************************************************
                        self.force_s_x_pre[i, index_i] = self.force_s_x[i, index_i] + f_s_inc[0]
                        self.force_s_y_pre[i, index_i] = self.force_s_y[i, index_i] + f_s_inc[1]
                        self.force_s_z_pre[i, index_i] = self.force_s_z[i, index_i] + f_s_inc[2]
                        self.force_s_x_pre[j, index_j] = self.force_s_x[j, index_j] - f_s_inc[0]
                        self.force_s_y_pre[j, index_j] = self.force_s_y[j, index_j] - f_s_inc[1]
                        self.force_s_z_pre[j, index_j] = self.force_s_z[j, index_j] - f_s_inc[2]

                        # DEBUG Mode **********************************************************
                        # print(self.force_s_y_post[i, index_i] / self.acc_disp[0])
                        # DEBUG Mode **********************************************************

    @ti.kernel
    def apply_bc_comp(self, gf: ti.template(), ic: ti.template()):
        for i in range(gf.num_ptc):
            x = gf.pos[i, 0]
            y = gf.pos[i, 1]
            z = gf.pos[i, 2]
            # Gap from the boundary in negative x direction:
            gap_x_n = x - gf.rad[i] + ic.len[0] / 2.0
            gap_x_p = -x - gf.rad[i] + ic.len[0] / 2.0
            gap_y_n = y - gf.rad[i] + ic.len[1] / 2.0
            gap_y_p = -y - gf.rad[i] + ic.len[1] / 2.0
            gap_z_n = z - gf.rad[i] + ic.len[2] / 2.0
            gap_z_p = -z - gf.rad[i] + ic.len[2] / 2.0

            if gap_x_n < 0:
                gf.force_n[i, 0] -= gap_x_n * self.stiff_n[0]
                ic.force[0] -= gap_x_n * self.stiff_n[0]
                vel_rel_bw = gf.vel[i, 0] - ic.vel_tgt[0]
                if vel_rel_bw < 0:
                    M = gf.mass[i]
                    K = self.stiff_n[0]
                    C = 2. * self.damp_wb[0] * ti.sqrt(K * M)
                    V = vel_rel_bw
                    gf.force_n[i, 0] += -C * V
                    ic.force[0] -= C * V
                else:
                    pass
                ic.stiffness[0] += self.stiff_n[0]

            elif gap_x_p < 0:
                gf.force_n[i, 0] += gap_x_p * self.stiff_n[0]
                ic.force[0] -= gap_x_p * self.stiff_n[0]
                vel_rel_bw = gf.vel[i, 0] - ic.vel_tgt[0]
                if vel_rel_bw > 0:
                    M = gf.mass[i]
                    K = self.stiff_n[0]
                    C = 2. * self.damp_wb[0] * ti.sqrt(K * M)
                    V = vel_rel_bw
                    gf.force_n[i, 0] += -C * V
                    ic.force[0] += C * V
                else:
                    pass
                ic.stiffness[0] += self.stiff_n[0]

            if gap_y_n < 0:
                gf.force_n[i, 1] -= gap_y_n * self.stiff_n[0]
                ic.force[1] -= gap_y_n * self.stiff_n[0]
                vel_rel_bw = gf.vel[i, 1] - ic.vel_tgt[1]
                if vel_rel_bw < 0:
                    M = gf.mass[i]
                    K = self.stiff_n[0]
                    C = 2. * self.damp_wb[0] * ti.sqrt(K * M)
                    V = vel_rel_bw
                    gf.force_n[i, 1] += -C * V
                    ic.force[1] -= C * V
                else:
                    pass
                ic.stiffness[1] += self.stiff_n[0]

            elif gap_y_p < 0:
                gf.force_n[i, 1] += gap_y_p * self.stiff_n[0]
                ic.force[1] -= gap_y_p * self.stiff_n[0]
                vel_rel_bw = gf.vel[i, 1] - ic.vel_tgt[1]
                if vel_rel_bw > 0:
                    M = gf.mass[i]
                    K = self.stiff_n[0]
                    C = 2. * self.damp_wb[0] * ti.sqrt(K * M)
                    V = vel_rel_bw
                    gf.force_n[i, 1] += -C * V
                    ic.force[1] += C * V
                else:
                    pass
                ic.stiffness[1] += self.stiff_n[0]

            if gap_z_n < 0:
                gf.force_n[i, 2] -= gap_z_n * self.stiff_n[0]
                ic.force[2] -= gap_z_n * self.stiff_n[0]
                vel_rel_bw = gf.vel[i, 2] - ic.vel_tgt[2]
                if vel_rel_bw < 0:
                    M = gf.mass[i]
                    K = self.stiff_n[0]
                    C = 2. * self.damp_wb[0] * ti.sqrt(K * M)
                    V = vel_rel_bw
                    gf.force_n[i, 2] += -C * V
                    ic.force[2] -= C * V
                else:
                    pass
                ic.stiffness[2] += self.stiff_n[0]

            elif gap_z_p < 0:
                gf.force_n[i, 2] += gap_z_p * self.stiff_n[0]
                ic.force[2] -= gap_z_p * self.stiff_n[0]
                vel_rel_bw = gf.vel[i, 2] - ic.vel_tgt[2]
                if vel_rel_bw > 0:
                    M = gf.mass[i]
                    K = self.stiff_n[0]
                    C = 2. * self.damp_wb[0] * ti.sqrt(K * M)
                    V = vel_rel_bw
                    gf.force_n[i, 2] += -C * V
                    ic.force[2] += C * V
                else:
                    pass
                ic.stiffness[2] += self.stiff_n[0]
