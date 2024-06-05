import taichi as ti
import numpy as np

vel = 1.0
acc_disp = 0.0
f_n = 0
f_d_factor = 0.1
mass = 1.0
stiff = 1.0e6
dt = 1.0e-4
disp = 0.0
cycle = 10000
n = 0

while n <= cycle:
    vel += f_n / mass * dt
    disp_inc = vel * dt
    disp += disp_inc
    f_n = (-disp * stiff) - f_d_factor * vel
    n += 1
    print(vel)
