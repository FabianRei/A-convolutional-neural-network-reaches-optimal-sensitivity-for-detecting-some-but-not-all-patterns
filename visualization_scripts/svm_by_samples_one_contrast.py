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
include_svm = True

shift = False
angle = False

if shift:
    metric = 'shift'
elif angle:
    metric = 'angle'
else:
    metric = 'contrast'


folder = r'C:\Users\Fabian\Documents\data\rsync\sample_number_contrast\svm'
folder2 = r'C:\Users\Fabian\Documents\data\rsync\sample_number_contrast\svm_1_sample'
# contrast to select is 1.25892541e-03
contrast_selector = 1.25892541e-03

fname = f'svm_accuracies_one_contrast_{metric}'
epsilon = 0.001

csv1 = os.path.join(folder, 'results.csv')
csv_svm = os.path.join(folder, 'svm_results.csv')
csv_svm2 = os.path.join(folder2, 'svm_results.csv')
oo = get_csv_column(csv1, 'optimal_observer_d_index', sort_by=metric)
nn = get_csv_column(csv1, 'nn_dprime', sort_by=metric)
contrasts = get_csv_column(csv1, metric, sort_by=metric)

fig = plt.figure()
plt.grid(which='both')
svm = get_csv_column(csv_svm, 'dprime_accuracy', sort_by=metric)
num_samples = get_csv_column(csv_svm, 'samples_used', sort_by=metric)
svm_contrasts = get_csv_column(csv_svm, metric, sort_by=metric)
svm2 = get_csv_column(csv_svm2, 'dprime_accuracy', sort_by=metric)
num_samples2 = get_csv_column(csv_svm2, 'samples_used', sort_by=metric)
svm_contrasts2 = get_csv_column(csv_svm2, metric, sort_by=metric)
svm[svm >= (svm.max() - epsilon)] = oo.max()

num_samples_contrast = np.concatenate((num_samples[np.isclose(svm_contrasts, contrast_selector)],
                                       num_samples2[np.isclose(svm_contrasts2, contrast_selector)]))
svm_samples_contrast = np.concatenate(
    (svm[np.isclose(svm_contrasts, contrast_selector)], svm2[np.isclose(svm_contrasts2, contrast_selector)]))
sort_idxs = np.argsort(num_samples_contrast)
num_samples_contrast = num_samples_contrast[sort_idxs]
svm_samples_contrast = svm_samples_contrast[sort_idxs]
num_samples_contrast = num_samples_contrast-5000


if include_oo:
    plt.plot(num_samples_contrast, np.repeat(oo[np.isclose(contrasts, contrast_selector)], 10), label='Ideal Observer for reference')
if include_nn:
    plt.plot(num_samples_contrast, np.repeat(nn[np.isclose(contrasts, contrast_selector)], 10), label='ResNet18 for reference')
if include_svm:
    plt.plot(num_samples_contrast, svm_samples_contrast, label=f'SVM performance', alpha=0.3)
plt.xscale('log')
plt.yscale('log')
plt.xlabel(f"Number of samples used to train the SVM")
plt.ylabel('dprime')
if metric != 'shift':
    plt.title(f'Frequency 1 harmonic - dprime for contrast of 0.0012589')
else:
    plt.title(f'Frequency 1 harmonic - dprime for various {metric} values')
plt.legend(frameon=True)
out_path = os.path.dirname(csv1)
fig.savefig(os.path.join(out_path, f'{fname}.png'), dpi=200)
# fig.show()
print('done!')
