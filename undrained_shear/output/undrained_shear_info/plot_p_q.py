import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


df = pd.read_csv("undrained_shear_info.csv")
stress_x = df['stress_x']
stress_y = df['stress_y']
stress_z = df['stress_z']
void_ratio = df['void_ratio']
stress_p = (stress_x + stress_y + stress_z) / 3.0 / 1.0e3
stress_q = (stress_y - (stress_x + stress_z) / 2.0) / 1.0e3

length_y = df['length_y']
strain_y = (-length_y + length_y[0]) / length_y[0] * 100.0

fig = plt.figure()
ax = plt.gca()
ax.plot(stress_p[::5], stress_q[::5])

ax.set_xscale('linear')
ax.set_xlim(0.0, 300.0)
ax.set_ylim(-10.0, 220.0)

plt.show()