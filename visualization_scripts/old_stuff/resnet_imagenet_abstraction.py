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


csv0 = '/share/wandell/data/reith/experiment_freq_1_log_contrasts30_frozen_until_0/results.csv'
csv1 = '/share/wandell/data/reith/experiment_freq_1_log_contrasts30_frozen_until_1/results.csv'
csv2 = '/share/wandell/data/reith/experiment_freq_1_log_contrasts30_frozen_until_2/results.csv'
csv3 = '/share/wandell/data/reith/experiment_freq_1_log_contrasts30_frozen_until_3/results.csv'
csv4 = '/share/wandell/data/reith/experiment_freq_1_log_contrasts30_frozen_until_4/results.csv'
fname = 'network_imagenet_abstractions_dprime_curve'

oo = get_csv_column(csv0, 'optimal_observer_d_index', sort_by='contrast')
nn_0 = get_csv_column(csv0, 'nn_dprime', sort_by='contrast')
nn_1 = get_csv_column(csv1, 'nn_dprime', sort_by='contrast')
nn_2 = get_csv_column(csv2, 'nn_dprime', sort_by='contrast')
nn_3 = get_csv_column(csv3, 'nn_dprime', sort_by='contrast')
nn_4 = get_csv_column(csv4, 'nn_dprime', sort_by='contrast')
contrasts = get_csv_column(csv4, 'contrast', sort_by='contrast')


fig = plt.figure()
plt.xscale('log')
plt.xlabel('contrast')
plt.ylabel('dprime')
plt.title('Frequency 1 harmonic - dprime for various shifts')

plt.plot(contrasts, oo, label='Ideal Observer')
plt.plot(contrasts, nn_0, label='ResNet18 - all randomly initialized')
plt.plot(contrasts, nn_1, label='ResNet18 - first quarter ImageNet pretrained and frozen')
plt.plot(contrasts, nn_2, label='ResNet18 - first half ImageNet pretrained and frozen')
plt.plot(contrasts, nn_3, label='ResNet18 - first three quarters Imagenet pretrained and frozen')
plt.plot(contrasts, nn_4, label='ResNet18 - all layers except final fully connected layer ImageNet pretrained and frozen')

plt.legend(frameon=True, fontsize='x-small', loc='upper left')

out_path = os.path.dirname(csv0)
fig.savefig(os.path.join(out_path, f'{fname}.png'), dpi=500)
# fig.show()
print('done!')
