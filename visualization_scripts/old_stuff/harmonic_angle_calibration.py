import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import os


def get_csv_column(csv_path, col_name, sort_by=None):
    df = pd.read_csv(csv_path, delimiter=';')
    col = df[col_name].tolist()
    col = np.array(col)
    if sort_by is not None:
        sort_val = get_csv_column(csv_path, sort_by)
        sort_idxs = np.argsort(sort_val)
        col = col[sort_idxs]
    return col



csv1 = '/share/wandell/data/reith/2_class_MTF_angle_experiment/frequency_1/results.csv'
fname = 'harmonic_angle_calibration'

oo = get_csv_column(csv1, 'optimal_observer_d_index', sort_by='angle')
nn = get_csv_column(csv1, 'nn_dprime', sort_by='angle')
shifts = get_csv_column(csv1, 'angle', sort_by='angle')

fig = plt.figure()
# plt.grid(which='both')
plt.xscale('log')
plt.xlabel('angle in pi')
plt.ylabel('dprime')
plt.title('Frequency 1 harmonic - dprime for various angle values')

plt.plot(shifts/np.pi, oo, label='Ideal Observer')
plt.plot(shifts/np.pi, nn, label='ResNet18')
plt.legend(frameon=True)

out_path = os.path.dirname(csv1)
fig.savefig(os.path.join(out_path, f'{fname}.png'), dpi=200)
# fig.show()
print('done!')
