import taichi as ti
import sys
sys.path.append("../src")
from fmt import flt_dtype
from particle import Particle
from grid import Grid
from contact import Contact
from wall import Wall
from visual import VisualTool
import numpy as np
import csv
import pandas as pd
import math

vec = ti.math.vec3


@ti.data_oriented
class CyclicShear(object):
    def __init__(self, number_particle, vt_is_on, log_is_on=False, csr=0.25, freq=16.0):
        self.substep = 100
        self.particle = Particle(number_particle, 0.01, 0.005)  # grain field
        self.grid = Grid(num_ptc=self.particle.number, rad_max=self.particle.radMax[0])
        self.contact = Contact(self.particle.number, fric=0.3, model="hertz")  # contact info
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
        bias = 0.
        shrink = 0.15
        self.wallPosMin[0] = -self.grid.domain_size * (0.5 - shrink) 
        self.wallPosMin[1] = -self.grid.domain_size * (0.5 - shrink) 
        self.wallPosMin[2] = -self.grid.domain_size * (0.5 - shrink) - bias
        self.wallPosMax[0] = self.grid.domain_size * (0.5 - shrink) 
        self.wallPosMax[1] = self.grid.domain_size * (0.5 - shrink) 
        self.wallPosMax[2] = self.grid.domain_size * (0.5 - shrink) - bias
        self.wall = Wall(6, self.wallPosMin[0], self.wallPosMax[0],
                         self.wallPosMin[1], self.wallPosMax[1],
                         self.wallPosMin[2], self.wallPosMax[2])
        self.length = ti.field(dtype=flt_dtype, shape=3)
        self.axialLengthIni = ti.field(dtype=flt_dtype, shape=1)
        self.durationCyclicIni = ti.field(dtype=flt_dtype, shape=(1,))
        self.contactStiffnessMin = ti.field(dtype=flt_dtype, shape=(1,))
        self.volume = ti.field(dtype=flt_dtype, shape=1)
        self.voidRatio = ti.field(dtype=flt_dtype, shape=1)
        self.stress = ti.field(dtype=flt_dtype, shape=3)
        self.stressP = ti.field(dtype=flt_dtype, shape=1)
        self.stressP0 = ti.field(dtype=flt_dtype, shape=1)
        self.stressDifRatio = ti.field(dtype=flt_dtype, shape=3)
        self.servoStress = ti.field(dtype=flt_dtype, shape=3)
        self.servoVelocity = ti.field(dtype=flt_dtype, shape=3)
        self.csr = csr
        self.freq = freq
        self.cyc_num = ti.field(dtype=ti.i32, shape=1)
        self.rec_num = ti.field(dtype=ti.i32, shape=1)
        self.gravity = ti.field(dtype=flt_dtype, shape=(3,))
        self.gravity[1] = -9.81 * 0.

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

    def write_compress_info_title(self):
        with open('output/comp_info/compress_info.csv', 'w') as file:
            writer = csv.writer(file)
            writer.writerow(['stress_x', 'stress_y', 'stress_z', 'void_ratio'])

    def write_cyclic_shear_info_title(self):
        with open('output/cyclic_shear_info/cyclic_shear_info.csv', 'w') as file:
            writer = csv.writer(file)
            writer.writerow(['stress_x', 'stress_y', 'stress_z',
             'length_x', 'length_y', 'length_z', 'void_ratio', 'duration'])

    def write_compress_info(self):
        with open('output/comp_info/compress_info.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([self.stress[0], self.stress[1], self.stress[2], self.voidRatio[0]])

    def write_cyclic_shear_info(self):
        with open('output/cyclic_shear_info/cyclic_shear_info.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([self.stress[0], self.stress[1], self.stress[2],
            self.length[0], self.length[1], self.length[2], self.voidRatio[0], self.duration[0]])      

    def write_ball_info(self, save_n, path='output/comp_info/ball_info_'):
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
        df.to_csv(path + '{}.csv'.format(save_n), index=False)

    def write_contact_info(self, index, path='output/comp_info/compress_contact_'):
        forceX = []
        forceY = []
        forceZ = []
        positionX = []
        positionY = []
        positionZ = []
        contactType = [] # type of 0 and 1 indicates ball-ball and ball-wall contacts
        end1 = []
        end2 = []
        for i in range(self.particle.number):
            for index_i in range(self.contact.lenContactBallBallRecord):
                j = self.contact.contacts[i, index_i]
                if j == -1:
                    break
                if i > j:
                    continue
                forceX.append(self.contact.forceX[i, index_i])
                forceY.append(self.contact.forceY[i, index_i])
                forceZ.append(self.contact.forceZ[i, index_i])
                positionX.append(self.contact.positionX[i, index_i])
                positionY.append(self.contact.positionY[i, index_i])
                positionZ.append(self.contact.positionZ[i, index_i])
                contactType.append(0)
                end1.append(i)
                end2.append(j)
        for i in range(self.particle.number):
            for index_i in range(self.contact.lenContactBallWallRecord):
                j = self.contact.contactsBallWall[i, index_i]
                if j == -1:
                    continue
                forceX.append(self.contact.forceBallWallX[i, index_i])
                forceY.append(self.contact.forceBallWallY[i, index_i])
                forceZ.append(self.contact.forceBallWallZ[i, index_i])
                positionX.append(self.contact.positionBallWallX[i, index_i])
                positionY.append(self.contact.positionBallWallY[i, index_i])
                positionZ.append(self.contact.positionBallWallZ[i, index_i])
                contactType.append(1)
                end1.append(i)
                end2.append(j)
        forceX = np.array(forceX)
        forceY = np.array(forceY)
        forceZ = np.array(forceZ)
        positionX = np.array(positionX)
        positionY = np.array(positionY)
        positionZ = np.array(positionZ)
        contactType = np.array(contactType)
        end1 = np.array(end1)
        end2 = np.array(end2)
        df = pd.DataFrame({'pos_x': positionX,
                           'pos_y': positionY,
                           'pos_z': positionZ,
                           'force_x': forceX,
                           'force_y': forceY,
                           'force_z': forceZ,
                           'contact_type': contactType,
                           'end1': end1,
                           'end2': end2})
        df.to_csv(path + '{}.csv'.format(index), index=False)

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
        if self.contact.model == "linear":
            self.contact.resolve_ball_wall_force(self.particle, self.wall)
        elif self.contact.model == "hertz":
            self.contact.resolve_ball_wall_force_hertz(self.particle, self.wall, 1)
        
        # particle update
        gravity = self.get_gravity()
        self.particle.update_pos_verlet(self.dt[0], gravity)
        # wall
        self.wall.update_position(timestep=self.dt[0])
        # advance time
        self.update_time()
        self.cyc_num[0] += 1

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
        self.stressP[0] = (self.stress[0] + self.stress[1] + self.stress[2]) / 3.

    def compute_stress_dif_ratio(self):
        self.stressDifRatio[0] = abs(self.stress[0] - self.servoStress[0])/self.servoStress[0]
        self.stressDifRatio[1] = abs(self.stress[1] - self.servoStress[1])/self.servoStress[1]
        self.stressDifRatio[2] = abs(self.stress[2] - self.servoStress[2])/self.servoStress[2]

    def is_stress_stable(self, tolerance=1.0e-2):
        return self.stressDifRatio[0] < tolerance and self.stressDifRatio[1] < tolerance and self.stressDifRatio[2] < tolerance

    def aggregate_particles(self):
        self.write_compress_info_title()
        tgt_p_0 = 200.0e3
        tgt_p_1 = 200.0e3
        ratio_p = tgt_p_1 / tgt_p_0
        record_count = 0
        for index_ratio in np.linspace(0, 1, 10):
            tgt_p = tgt_p_0 * (ratio_p) ** index_ratio
            self.set_servo_stress(vec(tgt_p, tgt_p, tgt_p))
            while True:
                e0 = self.voidRatio[0]
                if self.vt_is_on:
                    self.vt.update(self.particle)
                for _ in range(self.substep):
                    self.compute_servo()
                    self.update()
                self.print_info()
                e1 = self.voidRatio[0]
                isStable = self.is_stress_stable() and abs(e1 - e0)/e0 < 1.0e-5
                if isStable:
                    self.write_ball_info(record_count)
                    self.write_contact_info(record_count)
                    record_count += 1
                    self.write_compress_info()
                    break
        self.contact.frictionBallBall[0] = 0.5
        while True:
            e0 = self.voidRatio[0]
            if self.vt_is_on:
                self.vt.update(self.particle)
            for _ in range(self.substep):
                self.compute_servo()
                self.update()
            self.print_info()
            e1 = self.voidRatio[0]
            isStable = self.is_stress_stable() and abs(e1 - e0)/e0 < 5.0e-6
            if isStable:
                self.write_ball_info(record_count)
                self.write_contact_info(record_count)
                record_count += 1
                self.write_compress_info()
                break

    def cyclic_shear(self):
        self.write_cyclic_shear_info_title()
        self.axialLengthIni[0] = self.length[1]
        record_count = 0
        self.duration[0] = 0.0
        self.durationCyclicIni[0] = self.duration[0]
        self.contactStiffnessMin[0] = (self.wall.contactStiffness[0] + self.wall.contactStiffness[1]) * 0.5 * 0.4
        self.stressP0[0] = self.stressP[0]
        time_tgts = np.linspace(0.0, 10, 1001)
        time_tgt = time_tgts[record_count]
        while self.duration[0] < 10.0:
            if self.vt_is_on:
                self.vt.update(self.particle)  
            for _ in range(self.substep):
                self.compute_servo_cyclic_shear()
                self.update()

            self.print_info()
            self.write_cyclic_shear_info()
            if self.duration[0] > time_tgt:
                self.write_ball_info(record_count, path='output/cyclic_shear_info/cyclic_shear_ball_')
                self.write_contact_info(record_count, path='output/cyclic_shear_info/cyclic_shear_contact_')
                record_count += 1
                time_tgt = time_tgts[record_count]

    def settle(self):
        for _ in range(500):
            for _ in range(100):
                self.update()
            if self.vt_is_on:
                self.vt.update(self.particle)
            self.print_info()

    def move_wall(self):
        self.wall.position[5, 2] = self.grid.domain_size * 0.49
        self.wallPosMax[2] = self.wall.position[5, 2]
        while True:
            if self.vt_is_on:
                self.vt.update(self.particle)
            for j in range(self.substep):
                self.update()
            self.rec_num[0] += 1
            self.print_info()
            if self.log_is_on:
                self.write_ball_info(self.rec_num[0])
            if self.cyc_num[0] >= 1000000:
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
        servoFactor = 0.02
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
        velocityMax = 5
        velocityMin = -5
        servoVelocity = ti.min(vec(
            forceDifX / stiffnessX / self.dt[0] * servoFactor,
            forceDifY / stiffnessY / self.dt[0] * servoFactor,
            forceDifZ / stiffnessZ / self.dt[0] * servoFactor), velocityMax)
        servoVelocity = ti.max(servoVelocity, velocityMin)
        self.servoVelocity[0] = servoVelocity[0]
        self.servoVelocity[1] = servoVelocity[1]
        self.servoVelocity[2] = servoVelocity[2]

    def compute_servo_velocity_cyclic_shear(self):
        t = self.duration[0] - self.durationCyclicIni[0]
        target_q = self.stressP0[0] * ti.math.sin(ti.math.pi * 2 * t * self.freq) * self.csr
        current_q = self.stress[1] - (self.stress[0] + self.stress[2]) * 0.5
        dif_q = target_q - current_q
        area = vec(self.length[1] * self.length[2],
                   self.length[0] * self.length[2],
                   self.length[0] * self.length[1])
        dif_f = dif_q * area[1]
        stiff = ti.max((self.wall.contactStiffness[2] + self.wall.contactStiffness[3]) * 0.5, self.contactStiffnessMin[0])
        servoFactor = 0.08
        vel_max = 2.0
        vel_min = -2.0
        axial_vel = dif_f / (stiff * self.dt[0]) * servoFactor
        axial_vel = ti.min(axial_vel, vel_max)
        axial_vel = ti.max(axial_vel, vel_min)
        volume_increment = axial_vel * area[1]
        volume_frac_x = self.length[2] / (self.length[0] + self.length[2])
        volume_frac_z = 1. - volume_frac_x
        servoVelocity = vec(- volume_increment * volume_frac_x / area[0],
                            axial_vel,
                            - volume_increment * volume_frac_z / area[2])
        self.servoVelocity[0] = servoVelocity[0]
        self.servoVelocity[1] = servoVelocity[1]
        self.servoVelocity[2] = servoVelocity[2]

    def compute_servo(self):
        self.compute_length()
        self.compute_volume()
        self.compute_void_ratio()
        self.compute_stress()
        self.compute_servo_velocity()
        self.set_wall_servo_vel()
        self.compute_stress_dif_ratio()

    def compute_servo_cyclic_shear(self):
        self.compute_length()
        self.compute_volume()
        self.compute_void_ratio()
        self.compute_stress()
        self.compute_servo_velocity_cyclic_shear()
        self.set_wall_servo_vel()

    def print_info(self):
        print("*" * 80)
        print("* particle number: ".ljust(25) + str(self.particle.number))
        print("* time duration (s): ".ljust(25) +
              (str(round(self.duration[0], 6))).ljust(15))
        print("* stress(kPa): %.6e, %.6e, %.6e "%(self.stress[0]/1.0e3, self.stress[1]/1.0e3 ,self.stress[2]/1.0e3))
        print("* stiffness(N/m): %.4e, %.4e, %.4e "%(
            (self.wall.contactStiffness[0] + self.wall.contactStiffness[1]) * 0.5,
            (self.wall.contactStiffness[2] + self.wall.contactStiffness[3]) * 0.5,
            (self.wall.contactStiffness[4] + self.wall.contactStiffness[5]) * 0.5))
        print("* velocity(mm/s): %.6f, %.6f, %.6f "%(self.servoVelocity[0]*1.0e3, self.servoVelocity[1]*1.0e3 ,self.servoVelocity[2]*1.0e3))
        print("* length(mm): %.6e, %.6e, %.6e "%(self.length[0]*1.0e3, self.length[1]*1.0e3 ,self.length[2]*1.0e3))
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
        self.cyclic_shear()

