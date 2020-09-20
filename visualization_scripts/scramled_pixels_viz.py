import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import os


def get_csv_column(csv_path, col_name, sort_by=None, exclude_from=None):
    df = pd.read_csv(csv_path, delimiter=';')
    col = df[col_name].tolist()
    col = np.array(col)
    if sort_by is not None:
        sort_val = get_csv_column(csv_path, sort_by)
        sort_idxs = np.argsort(sort_val)
        col = col[sort_idxs]
    if exclude_from is not None:
        sort_val = sort_val[sort_idxs]
        col = col[sort_val >= exclude_from]
    return col


include_oo = True
include_nn = True
include_svm = False

shift = False
angle = True

if shift:
    metric = 'shift'
elif angle:
    metric = 'angle'
else:
    metric = 'contrast'


folder_paths = ['/share/wandell/data/reith/redo_experiments/sensor_harmonic_rotation/', '/share/wandell/data/reith/redo_experiments/shuffled_pixels/sensor_harmonic_rotation/']
fname = f'harmonic_curve_detection_{metric}_nn'

fig = plt.figure()
# plt.grid(which='both')
plt.xscale('log')
plt.xlabel(metric + " in pi")
plt.ylabel('dprime')
plt.title(f"Sensor data - harmonic curve pixel location vs randomized pixel location")
plt.grid(which='both')
for i, folder in enumerate(folder_paths):
    if i == 0:
        appendix = ''
    elif i == 1:
        appendix = ' randomized pixel location'
        include_oo = False
    csv1 = os.path.join(folder, 'results.csv')
    csv_svm = os.path.join(folder, 'svm_results.csv')
    oo = get_csv_column(csv1, 'optimal_observer_d_index', sort_by=metric)
    nn = get_csv_column(csv1, 'nn_dprime', sort_by=metric)
    contrasts = get_csv_column(csv1, metric, sort_by=metric)

    if include_oo:
        plt.plot(contrasts, oo, label='Ideal Observer'+appendix)
    if include_nn:
        plt.plot(contrasts, nn, label='ResNet18'+appendix)
    epsilon = 0.001
    if include_svm:
        svm = get_csv_column(csv_svm, 'dprime_accuracy', sort_by=metric)
        svm[svm >= (svm.max()-epsilon)] = oo.max()
        plt.plot(contrasts, svm, label='Support Vector Machine'+appendix)
    out_path = folder

plt.legend(frameon=True, loc='upper left', fontsize='xx-small')
fig.savefig(os.path.join(out_path, f'{fname}.png'), dpi=200)
# fig.show()
print('done!')
