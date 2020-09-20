from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import os
from deepLearning.src.analysis.weibull_alphas import get_alphas_fixed_beta


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





folder = r'C:\Users\Fabian\Documents\data\rsync\redo_experiments\sensor_harmonic_contrasts'
metric = 'contrast'
csv1 = os.path.join(folder, 'results.csv')
csv_svm = os.path.join(folder, 'svm_results.csv')
oo = get_csv_column(csv1, 'optimal_observer_d_index', sort_by=metric)
nn = get_csv_column(csv1, 'nn_dprime', sort_by=metric)
svm = get_csv_column(csv_svm, 'dprime_accuracy', sort_by=metric)
contrasts = get_csv_column(csv1, metric, sort_by=metric)
oo_nn = get_alphas_fixed_beta(contrasts, oo, nn)
oo_svm = get_alphas_fixed_beta(contrasts, oo, svm)
plt.plot(contrasts, oo)
print('nice')
