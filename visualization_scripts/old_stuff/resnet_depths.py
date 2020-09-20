import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import os


def get_csv_column(csv_path, col_name):
    df = pd.read_csv(csv_path, delimiter=';')
    col = df[col_name].tolist()
    return np.array(col)



csv1 = '/share/wandell/data/reith/experiment_freq_1_log_contrasts30_resnet18/results.csv'
csv2 = '/share/wandell/data/reith/experiment_freq_1_log_contrasts30_resnet101/results.csv'
fname = 'network_depths_dprime_curve'

oo = get_csv_column(csv1, 'optimal_observer_d_index')
nn_18 = get_csv_column(csv1, 'nn_dprime')
nn_101 = get_csv_column(csv2, "nn_dprime")
contrasts = get_csv_column(csv2, 'contrast')

sort_idxs = np.argsort(contrasts)
contrasts = contrasts[sort_idxs]
oo = oo[sort_idxs]
nn_18 = nn_18[sort_idxs]
nn_101 = nn_101[sort_idxs]

fig = plt.figure()
plt.xscale('log')
plt.xlabel('contrast')
plt.ylabel('dprime')
plt.title('Frequency 1 harmonic - dprime for various shifts')

plt.plot(contrasts, oo, label='Ideal Observer')
plt.plot(contrasts, nn_18, label='ResNet18')
plt.plot(contrasts, nn_101, label='ResNet101')
plt.legend(frameon=True)

out_path = os.path.dirname(csv1)
fig.savefig(os.path.join(out_path, f'{fname}.png'), dpi=200)
# fig.show()
print('done!')
