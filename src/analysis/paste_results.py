from deepLearning.src.models.optimal_observer import get_optimal_observer_acc, calculate_dprime
import numpy as np
import pickle
from glob import glob
import os
import csv
import pandas as pd


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

def get_contrast(path):
    part2 = path.split('_')[-8]
    part1 = path.split('_')[-9]
    contrast = part1 + '.' + part2
    contrast = float(contrast)
    return contrast


def sort_paths(paths):
    paths.sort(key=lambda x: int(x.split('_')[-8]))
    return paths


def write_csv_row(resultCSV, testAcc, accOptimal, d1, d2, dataContrast, nn_dprime):
    file_exists = os.path.isfile(resultCSV)
    with open(resultCSV, 'a') as csvfile:
        headers = ['ResNet_accuracy', 'optimal_observer_accuracy', 'theoretical_d_index', 'optimal_observer_d_index',
                   'contrast', 'nn_dprime']
        writer = csv.DictWriter(csvfile, delimiter=';', lineterminator='\n', fieldnames=headers)

        if not file_exists:
            writer.writeheader()  # file doesn't exist yet, write a header

        writer.writerow({'ResNet_accuracy': testAcc, 'optimal_observer_accuracy': accOptimal, 'theoretical_d_index': d1,
                         'optimal_observer_d_index': d2, 'contrast': dataContrast,
                         'nn_dprime': nn_dprime})


def write_csv_svm(resultCSV, svm_accuracy, dprime_accuracy, contrast, samples_used=1000):
    file_exists = os.path.isfile(resultCSV)
    with open(resultCSV, 'a') as csvfile:
        headers = ['svm_accuracy', 'dprime_accuracy', 'contrast', 'samples_used']
        writer = csv.DictWriter(csvfile, delimiter=';', lineterminator='\n', fieldnames=headers)

        if not file_exists:
            writer.writeheader()  # file doesn't exist yet, write a header

        writer.writerow(
            {'svm_accuracy': svm_accuracy, 'dprime_accuracy': dprime_accuracy, 'contrast': contrast, 'samples_used': samples_used})


def calculate_values(fp1, fp2):
    metric = 'contrast'
    oo = get_csv_column(fp1, 'optimal_observer_d_index', sort_by=metric)
    nn = get_csv_column(fp2, 'nn_dprime', sort_by=metric)
    oo_acc = get_csv_column(fp1, 'optimal_observer_accuracy', sort_by=metric)
    nn_acc = get_csv_column(fp2, 'ResNet_accuracy', sort_by=metric)
    contrast = get_csv_column(fp1, metric, sort_by=metric)
    csv_path = fp2
    if os.path.exists(csv_path):
        os.replace(csv_path, os.path.join(os.path.dirname(fp2), 'results_old.csv'))
    for o, n, o_acc, n_acc, cont in zip(oo, nn, oo_acc, nn_acc, contrast):
        write_csv_row(csv_path, n_acc, o_acc, -1, o, cont, n)


def transfer_column(fp1, fp2):
    metric = 'contrast'
    nn_test = get_csv_column(fp1, 'nn_dprime', sort_by=metric)
    if nn_test[0] != -1:
        return
    oo = get_csv_column(fp1, 'optimal_observer_d_index', sort_by=metric)
    nn = get_csv_column(fp2, 'nn_dprime', sort_by=metric)
    oo_acc = get_csv_column(fp2, 'optimal_observer_accuracy', sort_by=metric)
    nn_acc = get_csv_column(fp2, 'ResNet_accuracy', sort_by=metric)
    contrast = get_csv_column(fp2, metric, sort_by=metric)
    csv_path = fp2
    if os.path.exists(csv_path):
        os.replace(csv_path, os.path.join(os.path.dirname(fp2), 'results_old.csv'))
    for o, n, o_acc, n_acc, cont in zip(oo, nn, oo_acc, nn_acc, contrast):
        write_csv_row(csv_path, n_acc, o_acc, -1, o, cont, n)



# create result csv files based on pickle prediction-labels
f1 = r'C:\Users\Fabian\Documents\data\rsync\oo\more_nn_oo'
f2 = r'C:\Users\Fabian\Documents\data\rsync\oo\more_nn_2'
fpaths1 = glob(f"{f1}\\**\\results.csv", recursive=True)
fpaths2 = glob(f"{f2}\\**\\results.csv", recursive=True)
p1s = np.array([fp1.split('more_nn_oo')[-1] for fp1 in fpaths1])
fpaths1 = np.array(fpaths1)
for fp2 in fpaths2:
    p2 = fp2.split('more_nn_2')[-1]
    if p2 in p1s:
        fp1 = fpaths1[p1s == p2][0]
        transfer_column(fp1, fp2)
        print('nice')
    # fp1 = glob(f'{fp1}\\results.csv')[0]
    # fp2 = glob(f'{fp2}\\results.csv')[0]
    # calculate_values(fp1, fp2)
print('done')


r'''
# create result csv files based on pickle prediction-labels
f1 = r'C:\Users\Fabian\Documents\data\rsync\redo_experiments\multiple_locations\multiple_locations_experiment_ideal_observer_adjusted_oo'
f2 = r'C:\Users\Fabian\Documents\data\rsync\redo_experiments\multiple_locations\multiple_locations_experiment_modified_updated'
fpaths1 = glob(f"{f1}\\**\\results.csv", recursive=True)
fpaths2 = glob(f"{f2}\\**\\results.csv", recursive=True)
p1s = np.array([fp1.split('ideal_observer_adjusted_oo')[-1] for fp1 in fpaths1])
fpaths1 = np.array(fpaths1)
for fp2 in fpaths2:
    p2 = fp2.split('modified_updated')[-1]
    if p2 in p1s:
        fp1 = fpaths1[p1s == p2][0]
        transfer_column(fp1, fp2)
        print('nice')
    # fp1 = glob(f'{fp1}\\results.csv')[0]
    # fp2 = glob(f'{fp2}\\results.csv')[0]
    # calculate_values(fp1, fp2)
print('done')

###############################################################
# create result csv files based on pickle prediction-labels
f1 = r'C:\Users\Fabian\Documents\data\rsync\redo_experiments\multiple_locations\multiple_locations_experiment_ideal_observer_adjusted'
f2 = r'C:\Users\Fabian\Documents\data\rsync\redo_experiments\multiple_locations\multiple_locations_experiment'
fpaths1 = glob(rf"{f1}\*signal*")
fpaths2 = glob(rf"{f2}\*signal*")

for fp1, fp2 in zip(fpaths1, fpaths2):
    fp1 = glob(f'{fp1}\\results.csv')[0]
    fp2 = glob(f'{fp2}\\results.csv')[0]
    calculate_values(fp1, fp2)
print('done')

'''