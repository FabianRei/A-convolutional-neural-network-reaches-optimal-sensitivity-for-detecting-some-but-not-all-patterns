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



csv1 = '/share/wandell/data/reith/2_class_MTF_freq_experiment/frequency_6/results.csv'
csv2 =  '/share/wandell/data/reith/svm_num_samples_freq1_contrast/svm_results.csv'
fname = 'harmonic_contrast_svm_iterations_curve'

oo = get_csv_column(csv1, 'optimal_observer_d_index', sort_by='contrast')
nn = get_csv_column(csv1, 'nn_dprime', sort_by='contrast')
contrasts = get_csv_column(csv1, 'contrast', sort_by='contrast')

svms = get_csv_column(csv2, 'dprime_accuracy')
iterations = get_csv_column(csv2, 'samples_used')
svm_contrasts = get_csv_column(csv2, 'contrast')

sort_idxs = np.argsort(iterations)
svms, svm_contrasts, iterations = svms[sort_idxs], svm_contrasts[sort_idxs], iterations[sort_idxs]

svm_curves = []
it_samples = np.unique(iterations, return_counts=True)[1][0]
max_its = max(np.unique(iterations))
for it in range(len(np.unique(iterations))):
    start = it*it_samples
    end = (it+1)*it_samples
    svm_curves.append([svms[start:end], iterations[start:end], svm_contrasts[start:end], it])
    print(it, start, end, it)
fig = plt.figure()
# plt.grid(which='both')
plt.xscale('log')
plt.xlabel('contrast')
plt.ylabel('dprime')
plt.title('Frequency 1 harmonic - dprime for various contrast values')
plt.grid()

for svm, its, cont, i_it in svm_curves:
    srt = np.argsort(cont)
    svm = svm[srt]
    svm[((svm >= svm.max()) | (svm >= oo.max()))] = oo.max()
    svm[((svm <= svm.min()) | (svm <= oo.min()))] = 0
    cont = cont[srt]
    plt.plot(cont, svm, label=f'SVM on {int(its[0]*0.8)} samples', alpha=0.33)

plt.plot(contrasts, oo, label='Ideal Observer')
plt.plot(contrasts, nn, label='ResNet18')
plt.legend(frameon=True)

out_path = os.path.dirname(csv2)
fig.savefig(os.path.join(out_path, f'{fname}.png'), dpi=200)
# fig.show()
print('done!')
