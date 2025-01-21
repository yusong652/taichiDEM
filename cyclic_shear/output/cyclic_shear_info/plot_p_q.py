import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time 

df = pd.read_csv("cyclic_shear_info.csv")
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
<<<<<<< HEAD
ax.plot(stress_p[::16], stress_q[::16])

ax.set_xscale('linear')
ax.set_xlim(0.0, 220.0)
=======
line1, = ax.plot(stress_p[::1], stress_q[::1])

ax.set_xscale('linear')
ax.set_xlim(0.0, 250.0)
>>>>>>> resolving-taichi-error
ax.set_ylim(-80.0, 80.0)
ax.set_xlabel("Mean effective stress p'")
ax.set_ylabel("Deviator stress q")
plt.show()
