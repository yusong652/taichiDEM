import taichi as ti
import sys
sys.path.append("../src")
from fmt import flt_dtype
from particle import Particle
from grid import Grid
from contact import Contact
from wall import Wall
from visual import VisualTool
import csv
import pandas as pd

vec = ti.math.vec3


@ti.data_oriented
class Slope(object):
    def __init__(self, number_particle, vt_is_on, log_is_on=False):
        self.substep = 10000
        self.particle = Particle(number_particle)  # grain field
        self.grid = Grid(num_ptc=self.particle.number, rad_max=self.particle.radMax[0])
        self.contact = Contact(self.particle.number, model="hertz")  # contact info
        self.vt_is_on = vt_is_on
        self.log_is_on = log_is_on
        if self.vt_is_on:  # Visual mode on
            self.vt = VisualTool(n=self.particle.number)  # visualization tool
        else:
            pass
        self.duration = ti.field(dtype=flt_dtype, shape=(1,))
        self.duration[0] = 0.0
        self.dt = ti.field(dtype=flt_dtype, shape=(1,))
        self.wallPosMin = ti.field(dtype=flt_dtype, shape=3)
        self.wallPosMax = ti.field(dtype=flt_dtype, shape=3)
        self.wallPosMin[0] = -self.grid.domain_size * 0.48
        self.wallPosMin[1] = -self.grid.domain_size * 0.48
        self.wallPosMin[2] = -self.grid.domain_size * 0.1
        self.wallPosMax[0] = -self.grid.domain_size * 0.28
        self.wallPosMax[1] = self.grid.domain_size * 0.25
        self.wallPosMax[2] = self.grid.domain_size * 0.1
        self.len = ti.field(dtype=flt_dtype, shape=3)
        self.len[0] = self.wallPosMax[0] - self.wallPosMin[0]
        self.len[1] = self.wallPosMax[1] - self.wallPosMin[1]
        self.len[2] = self.wallPosMax[2] - self.wallPosMin[2]
        self.wall = Wall(6, self.wallPosMin[0], self.wallPosMax[0],
                         self.wallPosMin[1], self.wallPosMax[1],
                         self.wallPosMin[2], self.wallPosMax[2])
        self.cyc_num = ti.field(dtype=ti.i32, shape=1)
        self.rec_num = ti.field(dtype=ti.i32, shape=1)
        self.gravity = ti.field(dtype=flt_dtype, shape=(3,))
        self.gravity[1] = -9.81 * 10.0

    def get_critical_timestep(self):
        rad_min = self.particle.radMin[0]
        mass_min = ti.math.pi * rad_min**3 * 4 / 3 * self.particle.density[0]
        coefficient = 0.2
        timestep = ti.sqrt(mass_min/(self.contact.stiffnessNorm[0]*2.0)) * 2.0 * coefficient
        return timestep

    def init(self,):
        self.dt[0] = self.get_critical_timestep()
        self.contact.init_contact(self.dt[0])
        self.contact.clear_contact()

    def write_wall_info_title(self):
        with open('ic_info.csv', 'w') as file:
            writer = csv.writer(file)
            writer.writerow(['duration',
                             'posXMin', 'posXMax', 'posYMin', 'posYMax', 'posZMin', 'posZMax'])

    def write_wall_info(self):
        with open('ic_info.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([self.duration,
                             self.wall.position[0, 0], self.wall.position[1, 0],
                             self.wall.position[2, 1], self.wall.position[3, 1],
                             self.wall.position[4, 2], self.wall.position[5, 2]])

    def write_ball_info(self, save_n):
        df = pd.DataFrame({'pos_x': self.particle.pos.to_numpy()[:, 0],
                           'pos_y': self.particle.pos.to_numpy()[:, 1],
                           'pos_z': self.particle.pos.to_numpy()[:, 2],
                           'vel_x': self.particle.vel.to_numpy()[:, 0],
                           'vel_y': self.particle.vel.to_numpy()[:, 1],
                           'vel_z': self.particle.vel.to_numpy()[:, 2],
                           'velRot_x': self.particle.velRot.to_numpy()[:, 0],
                           'velRot_y': self.particle.velRot.to_numpy()[:, 1],
                           'velRot_z': self.particle.velRot.to_numpy()[:, 2],
                           'acc_x': self.particle.acc.to_numpy()[:, 0],
                           'acc_y': self.particle.acc.to_numpy()[:, 1],
                           'acc_z': self.particle.acc.to_numpy()[:, 2],
                           'forceContact_x': self.particle.forceContact.to_numpy()[:, 0],
                           'forceContact_y': self.particle.forceContact.to_numpy()[:, 1],
                           'forceContact_z': self.particle.forceContact.to_numpy()[:, 2],
                           'rad': self.particle.rad.to_numpy()})
        df.to_csv('output/ball_info_{}.csv'.format(save_n), index=False)

    def set_wall_vel(self, vel_x, vel_y, vel_z):
        vel_tgt = vec(vel_x, vel_y, vel_z)
        self.wall.set_velocity(0, vec(vel_tgt[0], 0.0, 0.0))
        self.wall.set_velocity(1, vec(-vel_tgt[0], 0.0, 0.0))
        self.wall.set_velocity(2, vec(0.0, vel_tgt[1], 0.0))
        self.wall.set_velocity(3, vec(0.0, -vel_tgt[1], 0.0))
        self.wall.set_velocity(4, vec(0.0, 0.0, vel_tgt[2]))
        self.wall.set_velocity(5, vec(0.0, 0.0, -vel_tgt[2]))

    def get_gravity(self):
        return vec(self.gravity[0], self.gravity[1], self.gravity[2])

    def update_time(self):
        self.duration[0] += self.dt[0]

    def update(self,):
        self.particle.clear_force()
        self.contact.clear_contact()
        self.wall.clear_contact_force()
        self.wall.clear_contact_stiffness()
        # contact detection
        self.contact.detect(self.particle, self.grid)
        self.contact.resolve_ball_wall_force_hertz(self.particle, self.wall, 1)
        
        # particle update
        gravity = self.get_gravity()
        self.particle.update_pos_verlet(self.dt[0], gravity)
        # wall
        self.wall.update_position(timestep=self.dt[0])
        # advance time
        self.update_time()

    def generate(self):
        self.particle.init_particle(
            self.wallPosMin[0] + self.len[0] * 0.005, self.wallPosMax[0] - self.len[0] * 0.005,
            self.wallPosMin[1] + self.len[1] * 0.005, self.wallPosMax[1] - self.len[1] * 0.005,
            self.wallPosMin[2] + self.len[2] * 0.005, self.wallPosMax[2] - self.len[2] * 0.005)
        calm_time = 20
        sub_calm_time = 200
        for i in range(calm_time):
            for j in range(sub_calm_time):
                self.update()
            self.particle.calm()
            print("{} steps finished in calm phase".format(sub_calm_time))

    def aggregate_particles(self):
        while True:
            if self.vt_is_on:
                self.vt.update(self.particle)

            for j in range(self.substep):
                self.update()
            self.cyc_num[0] += self.substep
            self.rec_num[0] += 1
            self.print_info()
            if self.log_is_on:
                self.write_ball_info(self.rec_num[0])
            if self.cyc_num[0] >= 2000000:
                break

    def move_wall(self):
        self.wall.position[1, 0] = self.grid.domain_size * 0.49
        self.wallPosMax[0] = self.wall.position[1, 0]
        while True:
            if self.vt_is_on:
                self.vt.update(self.particle)
            for j in range(self.substep):
                self.update()
            self.cyc_num[0] += self.substep
            self.rec_num[0] += 1
            self.print_info()
            if self.log_is_on:
                self.write_ball_info(self.rec_num[0])
            if self.cyc_num[0] >= 10000000:
                break

    def print_info(self):
        print("*" * 80)
        print("* particle number: ".ljust(25) + str(self.particle.number))
        print("* dim_y: ".ljust(25) + str(-self.wall.position[2, 1]+self.wall.position[3, 1]))
        print("* time duration (s): ".ljust(25) +
              (str(round(self.duration[0], 6))).ljust(15))
        print("* timestep (s): ".ljust(25) +
              ("%e" % self.dt[0]).ljust(15))
        print("* cycle num: ".ljust(25) +
              ("%d" % self.cyc_num[0]).ljust(15))
        print("*" * 80)

    def run(self, ):
        """pour the particles for demo"""
        self.generate()
        self.aggregate_particles()
        self.move_wall()
