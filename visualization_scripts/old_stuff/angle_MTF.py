import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import os
import bisect


def get_csv_column(csv_path, col_name, sort_by=None, max_vals=20):
    df = pd.read_csv(csv_path, delimiter=';')
    col = df[col_name].tolist()
    if len(col) > max_vals:
        col = col[:max_vals]
    col = np.array(col)
    if sort_by is not None:
        sort_val = get_csv_column(csv_path, sort_by)
        sort_idxs = np.argsort(sort_val)
        col = col[sort_idxs]
    return col


MTF_path = '/share/wandell/data/reith/2_class_MTF_angle_experiment/'
fname = 'Modular_transfer_function_frequencies_log_svm'
include_svm = True
target_d = 2

frequency_paths = [f.path for f in os.scandir(MTF_path) if f.is_dir()]

freqs =[]
nn_dprimes = []
oo_dprimes = []

for p in frequency_paths:
    freq = int(p.split('_')[-1])
    freqs.append(freq)
    nn_dprimes.append(get_csv_column(os.path.join(p, 'results.csv'), 'nn_dprime', sort_by='angle'))
    oo_dprimes.append(get_csv_column(os.path.join(p, 'results.csv'), 'optimal_observer_d_index', sort_by='angle'))

sort_idxs = np.argsort(freqs)
freqs, nn_dprimes, oo_dprimes = np.array(freqs), np.array(nn_dprimes), np.array(oo_dprimes)
freqs = freqs[sort_idxs]
nn_dprimes = nn_dprimes[sort_idxs]
oo_dprimes = oo_dprimes[sort_idxs]

angles = get_csv_column(os.path.join(frequency_paths[0], 'results.csv'), 'angle', sort_by='angle')
nn_bilinear_targets = []
oo_bilinear_targets = []

for dprimes in nn_dprimes:
    right_target = bisect.bisect(dprimes, target_d)
    left_target = right_target -1
    p_val = (target_d - dprimes[left_target])/(dprimes[right_target]-dprimes[left_target])
    interpolated_val = (1-p_val) * angles[left_target] + p_val * angles[right_target]
    nn_bilinear_targets.append(interpolated_val)

for dprimes in oo_dprimes:
    right_target = bisect.bisect(dprimes, target_d)
    left_target = right_target -1
    p_val = (target_d - dprimes[left_target])/(dprimes[right_target]-dprimes[left_target])
    print(p_val, angles[left_target])
    interpolated_val = (1-p_val) * angles[left_target] + p_val * angles[right_target]
    oo_bilinear_targets.append(interpolated_val)

nn_bilinear_targets = np.array(nn_bilinear_targets)
oo_bilinear_targets = np.array(oo_bilinear_targets)

fig = plt.figure()
plt.yscale('log')
plt.xscale('log')
plt.xlabel('frequency')
plt.ylabel('1 over angle')
plt.title('Modular Transfer Function - Harmonic curve angles with various frequencies')

plt.plot(freqs, 1/nn_bilinear_targets, label='ResNet')
plt.plot(freqs, 1/oo_bilinear_targets, label='Optimal Observer')
############SVM SUPPORT#######################################
if include_svm:
    svm_bilinear_targets = []
    svm_dprimes = []
    svm_freqs = []
    for p in frequency_paths:
        freq = int(p.split('_')[-1])
        svm_freqs.append(freq)
        svm_dprimes.append(get_csv_column(os.path.join(p, 'svm_results.csv'), 'dprime_accuracy', sort_by='angle'))
    svm_dprimes, svm_freqs = np.array(svm_dprimes), np.array(svm_freqs)
    sort_idxs = np.argsort(svm_freqs)
    svm_dprimes = svm_dprimes[sort_idxs]
    for dprimes in svm_dprimes:
        right_target = bisect.bisect(dprimes, target_d)
        left_target = right_target -1
        p_val = (target_d - dprimes[left_target])/(dprimes[right_target]-dprimes[left_target])
        interpolated_val = (1-p_val) * angles[left_target] + p_val * angles[right_target]
        svm_bilinear_targets.append(interpolated_val)
    svm_bilinear_targets = np.array(svm_bilinear_targets)
    plt.plot(freqs, 1 / svm_bilinear_targets, label='Support Vector Machine')
################################################################
plt.legend(frameon=True)

fig.savefig(os.path.join(MTF_path, f'{fname}.png'), dpi=200)
# fig.show()
print('done!')
