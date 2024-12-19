import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df = pd.read_csv("compress_info.csv")
stress_x = df['stress_x']
stress_y = df['stress_y']
stress_z = df['stress_z']
void_ratio = df['void_ratio']
stress_p = (stress_x + stress_y + stress_z) / 3.0 / 1.0e3

fig = plt.figure()
ax = plt.gca()
ax.plot(stress_p, void_ratio)
ax.set_xscale('log')
ax.set_xlim(5.0, 5.0e2)

plt.show()