import taichi as ti
import csv
import pandas as pd
import numpy as np

vec = ti.math.vec3
flt_dtype = ti.f32


@ti.data_oriented
class IsoComp(object):
    def __init__(self, gf, ci, gd, p=2.0e5, **params):
        self.gf = gf  # grain field
        self.ci = ci  # contact info
        self.gd = gd  # grid domain
        self.params = params
        if 'vt' in params:  # Visual mode on
            self.vt = params['vt']  # visualization tool
        else:
            pass  # Visual mode off
        self.time_duration = ti.field(dtype=flt_dtype ,shape=(1,))
        self.time_duration[0] = 0.0
        self.dt = ti.field(dtype=ti.f32, shape=(1,))
        self.p_tgt_0 = ti.field(dtype=flt_dtype, shape=(1,))
        self.p_tgt_u = ti.field(dtype=flt_dtype, shape=(1,))
        self.p_tgt_u[0] = p
        self.p_tgt_0[0] = 1.0 * 1e4
        self.p_tgt = ti.field(dtype=flt_dtype, shape=(1,))
        self.p_tgt[0] = self.p_tgt_0[0]
        self.bdr_min = ti.field(dtype=flt_dtype, shape=3)
        self.bdr_max = ti.field(dtype=flt_dtype, shape=3)
        self.bdr_min[0] = -self.gd.domain_size * 0.2
        self.bdr_min[1] = -self.gd.domain_size * 0.4
        self.bdr_min[2] = -self.gd.domain_size * 0.3
        self.bdr_max[0] = self.gd.domain_size * 0.2
        self.bdr_max[1] = self.gd.domain_size * 0.4
        self.bdr_max[2] = self.gd.domain_size * (-0.1)
        self.len = ti.field(dtype=flt_dtype, shape=3)
        self.len[0] = self.bdr_max[0] - self.bdr_min[0]
        self.len[1] = self.bdr_max[1] - self.bdr_min[1]
        self.len[2] = self.bdr_max[2] - self.bdr_min[2]
        self.disp = ti.field(dtype=flt_dtype, shape=3)
        self.disp_acc = ti.field(dtype=flt_dtype, shape=3)
        self.volume = self.len[0] * self.len[1] * self.len[2]
        self.e = ti.field(dtype=flt_dtype, shape=1)
        self.force = ti.field(dtype=flt_dtype, shape=3)
        self.area = ti.field(dtype=flt_dtype, shape=3)
        self.stress = ti.field(dtype=flt_dtype, shape=3)
        self.stiffness = ti.field(dtype=flt_dtype, shape=3)
        self.cn = ti.field(dtype=flt_dtype, shape=3)
        self.force_tgt = ti.field(dtype=flt_dtype, shape=3)
        self.vel_tgt = ti.field(dtype=flt_dtype, shape=3)
        self.ratio_stress = ti.field(dtype=flt_dtype, shape=3)
        self.cyc_num = ti.field(dtype=ti.i32, shape=1)
        self.vel_lmt = ti.field(dtype=flt_dtype, shape=3)

    def init(self,):
        self.dt[0] = 0.6 * ti.sqrt(ti.math.pi * 4 / 3 * self.gf.rad_min[0]
                                   ** 3 * self.gf.density[0] / self.ci.stiff_n[0])
        self.gf.init_particle(
            self.bdr_min[0]+self.len[0]*0.05, self.bdr_max[0]-self.len[0]*0.05,
            self.bdr_min[1]+self.len[1]*0.05, self.bdr_max[1]-self.len[1]*0.05,
            self.bdr_min[2]+self.len[2]*0.05, self.bdr_max[2]-self.len[2]*0.05)
        self.ci.init_contact(self.dt[0], self.gf, self.gd)
        # self.write_ic_info_title()

    def write_ic_info_title(self):
        with open('ic_info.csv', 'w') as file:
            writer = csv.writer(file)
            writer.writerow(['time', 'p', 'len_x',
                'len_y', 'len_z', 'void_ratio'])

    def write_ic_info(self):
        with open('ic_info.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([self.time_duration, self.p_tgt, self.len[0],
                             self.len[1], self.len[2], self.e[0]])

    def write_ball_info(self, save_n, gf):
        df = pd.DataFrame({'pos_x': gf.pos.to_numpy()[:, 0],
                           'pos_y': gf.pos.to_numpy()[:, 1],
                           'pos_z': gf.pos.to_numpy()[:, 2],
                           'rad': gf.rad.to_numpy()})
        df.to_csv('ball_info_{}.csv'.format(save_n), index=False)

    def get_area(self):
        self.area = vec(self.len[1] * self.len[2],
                        self.len[0] * self.len[2],
                        self.len[0] * self.len[1])

    def get_stress(self):
        for i in range(3):
            self.stress[i] = self.force[i] / 2.0 / self.area[i]

    def get_force_tgt(self):
        for i in range(3):
            self.force_tgt[i] = self.p_tgt[0] * self.area[i]

    def get_vel_tgt(self, servo_coef=0.02, stiff_min=1.0e7, ):
        for i in range(3):
            if self.stiffness[i] < stiff_min:
                self.stiffness[i] = stiff_min
            self.vel_tgt[i] = (self.force_tgt[i] - self.force[i] / 2) / (
                    self.dt[0] * self.stiffness[i]) * servo_coef
            if self.vel_tgt[i] > self.vel_lmt[i]:
                self.vel_tgt[i] = self.vel_lmt[i]
            elif self.vel_tgt[i] < -self.vel_lmt[i]:
                self.vel_tgt[i] = -self.vel_lmt[i]

    def get_disp(self):
        for i in range(3):
            self.disp[i] = - self.dt[0] * self.vel_tgt[i]
            self.disp_acc[i] += self.disp[i]

    def get_bdr_min(self):
        self.bdr_min[0] += self.disp[0]
        self.bdr_min[1] += self.disp[1]
        self.bdr_min[2] += self.disp[2]

    def get_bdr_max(self):
        self.bdr_max[0] -= self.disp[0]
        self.bdr_max[1] -= self.disp[1]
        self.bdr_max[2] -= self.disp[2]

    def get_len(self):
        self.len[0] = self.bdr_max[0] - self.bdr_min[0]
        self.len[1] = self.bdr_max[1] - self.bdr_min[1]
        self.len[2] = self.bdr_max[2] - self.bdr_min[2]

    def get_volume(self):
        self.volume = self.len[0] * self.len[1] * self.len[2]

    def get_e(self):
        self.e[0] = (self.volume - self.gf.volume_s[0]) / self.gf.volume_s[0]

    def get_stress_ratio(self):
        self.ratio_stress[0] = ti.abs((self.stress[0] - self.p_tgt[0]) / self.p_tgt[0])
        self.ratio_stress[1] = ti.abs((self.stress[1] - self.p_tgt[0]) / self.p_tgt[0])
        self.ratio_stress[2] = ti.abs((self.stress[2] - self.p_tgt[0]) / self.p_tgt[0])

    def update_time(self):
        self.time_duration[0] += self.dt[0]

    def update(self,):
        # law of motion
        self.gf.update_acc()
        self.gf.clear_force()
        self.gf.update_vel(self.dt[0])
        self.gf.update_pos(self.dt[0])

        # advance time
        self.update_time()

        # contact detection
        self.ci.clear_contact()
        self.ci.detect(self.gf, self.gd)

        # force-displacement law
        self.ci.get_force_normal(self.gf)
        # self.ci.get_force_shear_inc(self.gf)
        self.ci.get_force_shear(self.gf)
        self.ci.resolve_ball_wall_force(self.gf, self)

        # boundary
        self.get_disp()
        self.get_bdr_min()
        self.get_bdr_max()
        self.get_len()
        self.get_area()
        self.get_stress()
        self.get_force_tgt()
        self.get_vel_tgt()
        self.get_volume()
        self.get_e()
        self.get_stress_ratio()
        self.force[0] = 0.0
        self.force[1] = 0.0
        self.force[2] = 0.0
        self.stiffness[0] = 0.0
        self.stiffness[1] = 0.0
        self.stiffness[2] = 0.0
        

    def print_info(self):
        print("*" * 80)
        print("* particle number: ".ljust(25) + str(self.gf.num_ptc))
        print("* time duration (s): ".ljust(25) +
              (str(round(self.time_duration[0], 6))).ljust(15))
        print("* timestep (s): ".ljust(25) +
              ("%e" % self.dt[0]).ljust(15))
        print("* stress (kPa): ".ljust(25) +
              ("%e" % (self.stress[0] / 1.0e3)).ljust(15) +
              ("%e" % (self.stress[1] / 1.0e3)).ljust(15) +
              ("%e" % (self.stress[2] / 1.0e3)).ljust(15))
        print("* stiffness: (kN/m)".ljust(25) +
              ("%e" % (self.stiffness[0] / 1.0e3)).ljust(15) +
              ("%e" % (self.stiffness[1] / 1.0e3)).ljust(15) +
              ("%e" % (self.stiffness[2] / 1.0e3)).ljust(15))
        print("* vel (mm / s): ".ljust(25) +
              ("%e" % (self.vel_tgt[0] * 1000.0)).ljust(15) +
              ("%e" % (self.vel_tgt[1] * 1000.0)).ljust(15) +
              ("%e" % (self.vel_tgt[2] * 1000.0)).ljust(15))
        print("* cumulative disp (mm): ".ljust(25) +
              ("%e" % (self.disp_acc[0] * 1000.0)).ljust(15) +
              ("%e" % (self.disp_acc[1] * 1000.0)).ljust(15) +
              ("%e" % (self.disp_acc[2] * 1000.0)).ljust(15))
        print("* length (mm): ".ljust(25) +
              ("%e" % (self.len[0] * 1000.0)).ljust(15) +
              ("%e" % (self.len[1] * 1000.0)).ljust(15) +
              ("%e" % (self.len[2] * 1000.0)).ljust(15))

        print("* volume: ".ljust(25) +
              ("%e" % self.volume).ljust(15))
        print("* void ratio: ".ljust(25) +
              ("%e" % self.e[0]).ljust(15))
        print("* cycle num: ".ljust(25) +
              ("%d" % self.cyc_num[0]).ljust(15))
        print("*" * 80)


    def print_debug_info(self):
        # print("* frictional force (N): ".ljust(25) +
        #       ("%e" % self.gf.f[0, 1]).ljust(15))
        # print(self.ci.contacts[1, 0], self.ci.contacts[1, 1])
        # print(self.gf.vel_rot[0, 0], self.gf.vel_rot[1, 0])
        # print(self.gf.f[0, 2], self.gf.f[1, 2])
        # print(self.gf.vel[0, 2], self.gf.vel[1, 2])
        # print(self.ci.contacts[0, 0], self.ci.contacts[0, 1])
        print("vel rot of No.1: {}, {}, {}".format(self.gf.vel_rot[0, 0],
                                                   self.gf.vel_rot[0, 1],
                                                   self.gf.vel_rot[0, 2]))
        print("vel rot of No.2: {}, {}, {}".format(self.gf.vel_rot[1, 0],
                                                   self.gf.vel_rot[1, 1],
                                                   self.gf.vel_rot[1, 2]))

        print("shear force: {} {} {}".format(self.ci.force_s_x[0, 0],
                                             self.ci.force_s_y[0, 0],
                                             self.ci.force_s_z[0, 0]))
        # in_prod = (self.ci.force_s_x[0, 0] * self.ci.contact_dist_x[0, 0] +
        #            self.ci.force_s_y[0, 0] * self.ci.contact_dist_y[0, 0] +
        #            self.ci.force_s_z[0, 0] * self.ci.contact_dist_z[0, 0])
        # print("inner: {}".format(in_prod))
        # print("* stress (kPa): ".ljust(25) +
        #       ("%e" % (self.stress[0] / 1.0e3)).ljust(15) +
        #       ("%e" % (self.stress[1] / 1.0e3)).ljust(15) +
        #       ("%e" % (self.stress[2] / 1.0e3)).ljust(15))
        # print(self.ci.contacts[1, 0], self.ci.contacts[1, 1])
        # print(self.ci.force_s_y_post[0, 0])

    def compress(self,):
        self.substep_comp = 100
        indices = np.linspace(0, 1, 101)
        p_targets = self.p_tgt_0[0] * (self.p_tgt_u[0] /
                self.p_tgt_0[0]) ** indices
        save_n = 0
        self.update()  # initialize void ratio
        self.vel_lmt[0] = 0.0
        self.vel_lmt[1] = 0.0
        self.vel_lmt[2] = 0.0
        #  calm
        calm_time = 20
        sub_calm_time = 200
        for i in range(calm_time):
            for j in range(sub_calm_time):
                self.update()
            print("{} steps finished in calm phase".format(sub_calm_time))
            self.gf.calm()
        for p_tgt in p_targets:
            self.p_tgt[0] = p_tgt

            while True:
                if 'vt' in self.params:
                    self.vt.update_pos(self.gf)
                    self.vt.render(self.gf)
                e_pre = self.e[0]
                self.disp_acc.fill(0.0)
                for j in range(self.substep_comp):
                    self.update()
                self.cyc_num[0] += self.substep_comp
                self.print_info()
                e_cur = self.e[0]
                ratio_e = abs(e_cur - e_pre) / e_pre
                if (self.ratio_stress[0] < 1.0e-3 and
                        self.ratio_stress[1] < 1.0e-3 and
                        self.ratio_stress[2] < 1.0e-3 and
                        ratio_e < 5.0e-7):
                    break
            # self.write_ball_info(save_n, self.gf)
            # self.write_ic_info()
            save_n += 1

    def pour(self,):
        """pour the particles for demo"""
        self.vel_lmt[0] = 0.0
        self.vel_lmt[1] = 0.0
        self.vel_lmt[2] = 0.0
        self.substep_comp = 500
        #  calm
        calm_time = 10
        sub_calm_time = 1000
        rec_count = 0
        for i in range(calm_time):
            for j in range(sub_calm_time):
                self.update()
            print("{} steps finished in calm phase".format(sub_calm_time))
            self.gf.calm()
        while True:
            if 'vt' in self.params:
                self.vt.update_pos(self.gf)
                self.vt.render(self.gf)

            for j in range(self.substep_comp):
                self.update()
            self.cyc_num[0] += self.substep_comp
            self.print_info()
            # self.write_ball_info(rec_count, self.gf)
            rec_count += 1
            if self.cyc_num[0] >= 30000:
                break

        self.bdr_max[2] = -self.bdr_min[2]
        while True:
            if 'vt' in self.params:
                self.vt.update_pos(self.gf)
                self.vt.render(self.gf) 
            for j in range(self.substep_comp):
                self.update()
            self.cyc_num[0] += self.substep_comp
            self.print_info()
            # self.write_ball_info(rec_count, self.gf)
            rec_count += 1
            if self.cyc_num[0] >= 60000:
                break

    def debug(self):
        calm_time = 0
        sub_calm_time = 200
        for i in range(calm_time):
            for j in range(sub_calm_time):
                self.update()
            self.gf.calm()
        n = 0
        self.vel_lmt[0] = 0.0
        self.vel_lmt[1] = 0.0
        self.vel_lmt[2] = 0.0
        self.p_tgt[0] = 1.0e3
        while True:
            for g in range(1):
                self.update()
            if 'vt' in self.params:
                self.vt.update_pos(self.gf)
                self.vt.render(self.gf)
            n += 1
            self.print_debug_info()
            if n >= 1000000:
                break

    def comp_debug(self):
        self.substep_comp = 100
        indices = np.linspace(0, 1, 101)
        p_targets = self.p_tgt_0[0] * (self.p_tgt_u[0] /
                self.p_tgt_0[0]) ** indices
        self.vel_lmt[0] = 0.0
        self.vel_lmt[1] = 10.0
        self.vel_lmt[2] = 10.0
        save_n = 0
        self.update()  # initialize void ratio

        #  calm
        calm_time = 10
        sub_calm_time = 100
        for i in range(calm_time):
            for j in range(sub_calm_time):
                self.update()
            self.gf.calm()
        for p_tgt in p_targets:
            self.p_tgt[0] = p_tgt

            while True:
                if 'vt' in self.params:
                    self.vt.update_pos(self.gf)
                    self.vt.render(self.gf)
                e_pre = self.e[0]
                self.disp_acc.fill(0.0)
                for j in range(self.substep_comp):
                    self.update()
                self.cyc_num[0] += self.substep_comp
                self.print_info()
                e_cur = self.e[0]
                ratio_e = abs(e_cur - e_pre) / e_pre
                if (self.ratio_stress[1] < 1.0e-3 and
                        self.ratio_stress[2] < 1.0e-3 and
                        ratio_e < 5.0e-7):
                    break
            # self.write_ball_info(save_n, self.gf)
            # self.write_ic_info()
            save_n += 1
