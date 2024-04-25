import taichi as ti
import numpy as np

vel_rot = 1.0
rad = 1.0
acc_disp = 0.0
f_s = 0
I = 1.0
stiff = 1.0e6
dt = 1.0e-4
disp_inc = 0.0
cycle = 10000
n = 0

while n <= cycle:
    disp_inc = vel_rot * rad * dt
    f_s += (-disp_inc * stiff)
    vel_rot += f_s * rad / I * dt
    n += 1
    print(vel_rot)
