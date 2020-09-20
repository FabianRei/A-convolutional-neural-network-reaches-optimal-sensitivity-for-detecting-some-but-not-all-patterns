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


super_folder = '/share/wandell/data/reith/2_class_MTF_freq_experiment/'
include_svm = True

folder_paths = [f.path for f in os.scandir(super_folder) if f.is_dir()]

for p in folder_paths:
    csv1 = os.path.join(p, 'results.csv')
    csv_svm = os.path.join(p, 'svm_results.csv')
    fname = 'harmonic_contrast_curve_svm'

    oo = get_csv_column(csv1, 'optimal_observer_d_index', sort_by='contrast')
    nn = get_csv_column(csv1, 'nn_dprime', sort_by='contrast')
    contrasts = get_csv_column(csv1, 'contrast', sort_by='contrast')

    fig = plt.figure()
    # plt.grid(which='both')
    plt.xscale('log')
    plt.xlabel('contrast')
    plt.ylabel('dprime')
    plt.title(f"Frequency {p.split('_')[-1]} harmonic - dprime for various contrast values")

    plt.plot(contrasts, oo, label='Ideal Observer')
    plt.plot(contrasts, nn, label='ResNet18')
    if include_svm:
        svm = get_csv_column(csv_svm, 'dprime_accuracy', sort_by='contrast')
        svm[svm == svm.max()] = oo.max()
        plt.plot(contrasts, svm, label='Support Vector Machine')
    plt.legend(frameon=True)

    out_path = p
    fig.savefig(os.path.join(out_path, f'{fname}.png'), dpi=200)
    # fig.show()
    print('done!')
