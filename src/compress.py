import taichi as ti
import csv
import pandas as pd
import numpy as np
from fmt import flt_dtype
from wall import Wall
from visual import VisualTool

vec = ti.math.vec3

@ti.data_oriented
class IsoComp(object):
    def __init__(self, gf, ci, gd, vt_is_on, p=2.0e5):
        self.particle = gf  # grain field
        self.contact = ci  # contact info
        self.gd = gd  # grid domain
        self.vt_is_on = vt_is_on
        if self.vt_is_on:  # Visual mode on
            self.vt = VisualTool(n=gf.num_ptc)  # visualization tool
        else:
            pass
        self.time_duration = ti.field(dtype=flt_dtype ,shape=(1,))
        self.time_duration[0] = 0.0
        self.dt = ti.field(dtype=ti.f32, shape=(1,))
        self.p_tgt_0 = ti.field(dtype=flt_dtype, shape=(1,))
        self.p_tgt_u = ti.field(dtype=flt_dtype, shape=(1,))
        self.p_tgt_u[0] = p
        self.p_tgt_0[0] = 1.0 * 1e4
        self.p_tgt = ti.field(dtype=flt_dtype, shape=(1,))
        self.p_tgt[0] = self.p_tgt_0[0]
        self.wallPosMin = ti.field(dtype=flt_dtype, shape=3)
        self.wallPosMax = ti.field(dtype=flt_dtype, shape=3)
        self.wallPosMin[0] = -self.gd.domain_size * 0.2
        self.wallPosMin[1] = -self.gd.domain_size * 0.4
        self.wallPosMin[2] = -self.gd.domain_size * 0.3
        self.wallPosMax[0] = self.gd.domain_size * 0.2
        self.wallPosMax[1] = self.gd.domain_size * 0.4
        self.wallPosMax[2] = self.gd.domain_size * (-0.1)
        self.len = ti.field(dtype=flt_dtype, shape=3)
        self.len[0] = self.wallPosMax[0] - self.wallPosMin[0]
        self.len[1] = self.wallPosMax[1] - self.wallPosMin[1]
        self.len[2] = self.wallPosMax[2] - self.wallPosMin[2]
        self.wall = Wall(6, self.wallPosMin[0], self.wallPosMax[0],
                         self.wallPosMin[1], self.wallPosMax[1],
                         self.wallPosMin[2], self.wallPosMax[2])
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
        self.dt[0] = 0.6 * ti.sqrt(ti.math.pi * 4 / 3 * self.particle.rad_min[0]
                                   ** 3 * self.particle.density[0] / self.contact.stiff_n[0])
        self.particle.init_particle(
            self.wallPosMin[0] + self.len[0] * 0.05, self.wallPosMax[0] - self.len[0] * 0.05,
            self.wallPosMin[1] + self.len[1] * 0.05, self.wallPosMax[1] - self.len[1] * 0.05,
            self.wallPosMin[2] + self.len[2] * 0.05, self.wallPosMax[2] - self.len[2] * 0.05)
        self.contact.init_contact(self.dt[0], self.particle, self.gd)

        # contact detection
        self.contact.detect(self.particle, self.gd)

        # force-displacement law
        self.contact.get_force_normal(self.particle)
        self.contact.resolve_ball_ball_shear_force(self.particle)
        self.contact.resolve_ball_wall_force(self.particle, self, self.wall)

        self.contact.clear_contact()

        self.particle.update_acc()
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
        self.wall.set_velocity(0, vec(self.vel_tgt[0], 0.0, 0.0))
        self.wall.set_velocity(1, vec(-self.vel_tgt[0], 0.0, 0.0))
        self.wall.set_velocity(2, vec(0.0, self.vel_tgt[1], 0.0))
        self.wall.set_velocity(3, vec(0.0, -self.vel_tgt[1], 0.0))
        self.wall.set_velocity(4, vec(0.0, 0.0, self.vel_tgt[2]))
        self.wall.set_velocity(5, vec(0.0, 0.0, -self.vel_tgt[2]))

    def get_disp(self):
        for i in range(3):
            self.disp[i] = - self.dt[0] * self.vel_tgt[i]

    def get_len(self):
        self.len[0] = - self.wall.position[0, 0] + self.wall.position[1, 0]
        self.len[1] = - self.wall.position[2, 1] + self.wall.position[3, 1]
        self.len[2] = - self.wall.position[4, 2] + self.wall.position[5, 2]

    def get_volume(self):
        self.volume = self.len[0] * self.len[1] * self.len[2]

    def get_e(self):
        self.e[0] = (self.volume - self.particle.volume_s[0]) / self.particle.volume_s[0]

    def get_stress_ratio(self):
        self.ratio_stress[0] = ti.abs((self.stress[0] - self.p_tgt[0]) / self.p_tgt[0])
        self.ratio_stress[1] = ti.abs((self.stress[1] - self.p_tgt[0]) / self.p_tgt[0])
        self.ratio_stress[2] = ti.abs((self.stress[2] - self.p_tgt[0]) / self.p_tgt[0])

    def update_time(self):
        self.time_duration[0] += self.dt[0]

    def update(self,):
        # law of motion
        self.particle.update_pos(self.dt[0])

        # contact detection
        self.contact.detect(self.particle, self.gd)

        # force-displacement law
        self.contact.get_force_normal(self.particle)
        self.contact.resolve_ball_ball_shear_force(self.particle)
        self.contact.resolve_ball_wall_force(self.particle, self, self.wall)
        self.contact.clear_contact()

        self.particle.record_acc()
        self.particle.update_acc()
        self.particle.clear_force()
        self.particle.update_vel(self.dt[0])

        # advance time
        self.update_time()

        # boundary
        # wall
        self.wall.update_position(timestep=self.contact.dt)
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
        print("* particle number: ".ljust(25) + str(self.particle.num_ptc))
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

    def pour(self,):
        """pour the particles for demo"""
        self.vel_lmt[0] = 0.0
        self.vel_lmt[1] = 0.0
        self.vel_lmt[2] = 0.0
        self.substep_comp = 500
        #  calm
        calm_time = 10
        sub_calm_time = 3000
        rec_count = 0
        for i in range(calm_time):
            for j in range(sub_calm_time):
                self.update()
            print("{} steps finished in calm phase".format(sub_calm_time))
            self.particle.calm()
        while True:
            if self.vt_is_on:
                self.vt.update_pos(self.particle)
                self.vt.render(self.particle)

            for j in range(self.substep_comp):
                self.update()
            self.cyc_num[0] += self.substep_comp
            self.print_info()
            # self.write_ball_info(rec_count, self.gf)
            rec_count += 1
            if self.cyc_num[0] >= 50000:
                break

        self.wallPosMax[2] = -self.wallPosMin[2]
        self.wall.position[5, 2] = -self.wall.position[4, 2]
        while True:
            if self.vt_is_on:
                self.vt.update_pos(self.particle)
                self.vt.render(self.particle)
            for j in range(self.substep_comp):
                self.update()
            self.cyc_num[0] += self.substep_comp
            self.print_info()
            # self.write_ball_info(rec_count, self.gf)
            rec_count += 1
            if self.cyc_num[0] >= 200000:
                break
