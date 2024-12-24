import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df = pd.read_csv("cyclic_shear_info.csv")
stress_x = df['stress_x']
stress_y = df['stress_y']
stress_z = df['stress_z']
void_ratio = df['void_ratio']
stress_p = (stress_x + stress_y + stress_z) / 3.0 / 1.0e3
stress_q = (stress_y - (stress_x + stress_z) / 2.0) / 1.0e3

length_y = df['length_y']
length_x = df['length_x']
length_z = df['length_z']
strain_y = (-length_y + length_y[0]) / length_y[0] * 100.0
strain_x = (-length_x + length_x[0]) / length_x[0] * 100.0
strain_z = (-length_z + length_z[0]) / length_z[0] * 100.0
strain_q = strain_y - (strain_x + strain_z) * 0.5
fig = plt.figure()
ax = plt.gca()
ax.plot(strain_q[::5], stress_q[::5])

ax.set_xscale('linear')
ax.set_xlim(-5.0, 5.0)
ax.set_xlabel(r"$Deviator\ strain\ \varepsilon_q\ (\%)$")
ax.set_ylabel(r"$Deviator\ stress\ q$")
# ax.set_ylim(-10.0, 100.0)

plt.show()