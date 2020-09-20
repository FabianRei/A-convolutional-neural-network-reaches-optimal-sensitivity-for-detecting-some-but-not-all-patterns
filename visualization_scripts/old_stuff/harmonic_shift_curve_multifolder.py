import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import os


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


super_folder = '/share/wandell/data/reith/2_class_MTF_shift_experiment/'
include_svm = True

folder_paths = [f.path for f in os.scandir(super_folder) if f.is_dir()]

for p in folder_paths:
    csv1 = os.path.join(p, 'results.csv')
    csv_svm = os.path.join(p, 'svm_results.csv')
    fname = 'harmonic_shift_curve_svm'

    oo = get_csv_column(csv1, 'optimal_observer_d_index', sort_by='shift')
    nn = get_csv_column(csv1, 'nn_dprime', sort_by='shift')
    shifts = get_csv_column(csv1, 'shift', sort_by='shift')

    fig = plt.figure()
    # plt.grid(which='both')
    plt.xscale('log')
    plt.xlabel('shift in pi')
    plt.ylabel('dprime')
    plt.title(f"Frequency {p.split('_')[-1]} harmonic - dprime for various shift values")
    freq = int(p.split('_')[-1])
    plt.plot(shifts/np.pi/freq, oo, label='Ideal Observer')
    plt.plot(shifts/np.pi/freq, nn, label='ResNet18')
    if include_svm:
        svm = get_csv_column(csv_svm, 'dprime_accuracy', sort_by='shift')
        svm[svm == svm.max()] = oo.max()
        plt.plot(shifts/np.pi/freq, svm, label='Support Vector Machine')
    plt.legend(frameon=True)

    out_path = os.path.dirname(csv1)
    fig.savefig(os.path.join(out_path, f'{fname}.png'), dpi=200)
    # fig.show()
    print('done!')
