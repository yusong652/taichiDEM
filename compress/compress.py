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
class Compress(object):
    def __init__(self, number_particle, vt_is_on, log_is_on=False):
        self.substep = 100
        self.particle = Particle(number_particle)  # grain field
        self.grid = Grid(num_ptc=self.particle.number, rad_max=self.particle.radMax[0])
        self.contact = Contact(self.particle.number)  # contact info
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
        self.wallPosMin[0] = -self.grid.domain_size * 0.4
        self.wallPosMin[1] = -self.grid.domain_size * 0.4
        self.wallPosMin[2] = -self.grid.domain_size * 0.4
        self.wallPosMax[0] = self.grid.domain_size * 0.4
        self.wallPosMax[1] = self.grid.domain_size * 0.4
        self.wallPosMax[2] = self.grid.domain_size * 0.4
        self.wall = Wall(6, self.wallPosMin[0], self.wallPosMax[0],
                         self.wallPosMin[1], self.wallPosMax[1],
                         self.wallPosMin[2], self.wallPosMax[2])
        self.length = ti.field(dtype=flt_dtype, shape=3)
        self.volume = ti.field(dtype=flt_dtype, shape=1)
        self.voidRatio = ti.field(dtype=flt_dtype, shape=1)
        self.stress = ti.field(dtype=flt_dtype, shape=3)
        self.servoStress = ti.field(dtype=flt_dtype, shape=3)
        self.servoVelocity = ti.field(dtype=flt_dtype, shape=3)
        self.cyc_num = ti.field(dtype=ti.i32, shape=1)
        self.rec_num = ti.field(dtype=ti.i32, shape=1)
        self.gravity = ti.field(dtype=flt_dtype, shape=(3,))
        self.gravity[1] = -0

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

    def set_wall_servo_vel(self):
        vel_tgt = vec(self.servoVelocity[0], self.servoVelocity[1], self.servoVelocity[2])
        self.wall.set_velocity(0, vec(vel_tgt[0], 0.0, 0.0))
        self.wall.set_velocity(1, vec(-vel_tgt[0], 0.0, 0.0))
        self.wall.set_velocity(2, vec(0.0, vel_tgt[1], 0.0))
        self.wall.set_velocity(3, vec(0.0, -vel_tgt[1], 0.0))
        self.wall.set_velocity(4, vec(0.0, 0.0, vel_tgt[2]))
        self.wall.set_velocity(5, vec(0.0, 0.0, -vel_tgt[2]))

    def set_servo_stress(self, stress: vec):
        self.servoStress[0] = stress[0]
        self.servoStress[1] = stress[1]
        self.servoStress[2] = stress[2]

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
        self.contact.resolve_ball_wall_force(self.particle, self.wall)
        
        # particle update
        gravity = self.get_gravity()
        self.particle.update_pos_euler(self.dt[0], gravity)
        # wall
        self.compute_servo()
        self.wall.update_position(timestep=self.dt[0])
        # advance time
        self.update_time()

    def generate(self):
        self.particle.init_particle(
            self.wallPosMin[0] + 0.005, self.wallPosMax[0] - 0.005,
            self.wallPosMin[1] + 0.005, self.wallPosMax[1] - 0.005,
            self.wallPosMin[2] + 0.005, self.wallPosMax[2] - 0.005)
        calm_time = 20
        sub_calm_time = 200
        for i in range(calm_time):
            for j in range(sub_calm_time):
                self.update()
            self.particle.calm()
            print("{} steps finished in calm phase".format(sub_calm_time))

    def compute_stress(self):
        forceX = (self.wall.contactForce[1, 0] - self.wall.contactForce[0, 0]) * 0.5
        forceY = (self.wall.contactForce[3, 1] - self.wall.contactForce[2, 1]) * 0.5
        forceZ = (self.wall.contactForce[5, 2] - self.wall.contactForce[4, 2]) * 0.5
        self.stress[0] = forceX / (self.length[1] * self.length[2])
        self.stress[1] = forceY / (self.length[0] * self.length[2])
        self.stress[2] = forceZ / (self.length[0] * self.length[1])

    def aggregate_particles(self):
        self.set_servo_stress(vec(5000, 5000, 5000))
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

    def compute_length(self):
        self.length[0] = self.wall.position[1, 0] - self.wall.position[0, 0]
        self.length[1] = self.wall.position[3, 1] - self.wall.position[2, 1]
        self.length[2] = self.wall.position[5, 2] - self.wall.position[4, 2]

    def compute_volume(self):
        self.volume[0] = self.length[0] * self.length[1] * self.length[2]

    def compute_void_ratio(self):
        self.voidRatio[0] = (self.volume[0] - self.particle.volumeSolid[0]) / self.particle.volumeSolid[0]

    def compute_servo_velocity(self):
        servoFactor = 0.01
        forceCurX = self.stress[0] * self.length[1] * self.length[2]
        forceCurY = self.stress[1] * self.length[0] * self.length[2]
        forceCurZ = self.stress[2] * self.length[0] * self.length[1]
        forceTargetX = self.servoStress[0] * self.length[1] * self.length[2]
        forceTargetY = self.servoStress[1] * self.length[0] * self.length[2]
        forceTargetZ = self.servoStress[2] * self.length[0] * self.length[1]
        forceDifX = forceTargetX - forceCurX
        forceDifY = forceTargetY - forceCurY
        forceDifZ = forceTargetZ - forceCurZ
        stiffnessMin = 1.0e7
        stiffnessX = ti.max((self.wall.contactStiffness[0] + self.wall.contactStiffness[1]) * 0.5, stiffnessMin)
        stiffnessY = ti.max((self.wall.contactStiffness[2] + self.wall.contactStiffness[3]) * 0.5, stiffnessMin)
        stiffnessZ = ti.max((self.wall.contactStiffness[4] + self.wall.contactStiffness[5]) * 0.5, stiffnessMin)
        velocityMax = 0.5
        self.servoVelocity[0] = ti.min(forceDifX / stiffnessX / self.dt[0] * servoFactor, velocityMax)
        self.servoVelocity[1] = ti.min(forceDifY / stiffnessY / self.dt[0] * servoFactor, velocityMax)
        self.servoVelocity[2] = ti.min(forceDifZ / stiffnessZ / self.dt[0] * servoFactor, velocityMax)

    def compute_servo(self):
        self.compute_length()
        self.compute_volume()
        self.compute_void_ratio()
        self.compute_stress()
        self.compute_servo_velocity()
        self.set_wall_servo_vel()

    def print_info(self):
        print("*" * 80)
        print("* particle number: ".ljust(25) + str(self.particle.number))
        print("* time duration (s): ".ljust(25) +
              (str(round(self.duration[0], 6))).ljust(15))
        print("* stress(Pa): %.2f, %.2f, %.2f "%(self.stress[0], self.stress[1] ,self.stress[2]))
        print("* velocity(mm/s): %.6f, %.6f, %.6f "%(self.servoVelocity[0]*1.0e3, self.servoVelocity[1]*1.0e3 ,self.servoVelocity[2]*1.0e3))
        print("* length(mm): %.6f, %.6f, %.6f "%(self.length[0]*1.0e3, self.length[1]*1.0e3 ,self.length[2]*1.0e3))
        print("* void ratio: %.8f "%self.voidRatio[0])
        print("* timestep (s): ".ljust(25) +
              ("%e" % self.dt[0]).ljust(15))
        print("* cycle num: ".ljust(25) +
              ("%d" % self.cyc_num[0]).ljust(15))
        print("*" * 80)

    def run(self, ):
        """pour the particles for demo"""
        self.generate()
        self.aggregate_particles()
