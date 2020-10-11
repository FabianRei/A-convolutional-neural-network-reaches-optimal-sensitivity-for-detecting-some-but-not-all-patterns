import pandas as pd
import numpy as np
import os
import bisect
from glob import glob
import collections
import csv

def nested_dict():
    return collections.defaultdict(nested_dict)


def get_csv_column(csv_path, col_name, sort_by=None, exclude_from=None):
    try:
        df = pd.read_csv(csv_path, delimiter=';')
        col = df[col_name].tolist()
    except:
        df = pd.read_csv(csv_path, delimiter=',')
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


def get_interpolation(dprimes, target_d, metric_values):
    """"
    Interpolate contrast sensitivity via linear interpolation
    """
    right_target = bisect.bisect(dprimes, target_d)
    left_target = right_target - 1
    p_val = (target_d - dprimes[left_target]) / (dprimes[right_target] - dprimes[left_target])
    interpolated_val = (1 - p_val) * metric_values[left_target] + p_val * metric_values[right_target]
    return 1/interpolated_val


def get_contrast_sensitivity(fpath, target_d=1.5):
    """"
    Get contrast sensitivity from result csv files within "fpath" folder
    """

    metric = 'contrast'
    nn_dprimes = get_csv_column(os.path.join(fpath, 'results.csv'), 'nn_dprime', sort_by=metric)
    oo_dprimes = get_csv_column(os.path.join(fpath, 'results.csv'), 'optimal_observer_d_index', sort_by=metric)
    metric_values = get_csv_column(os.path.join(fpath, 'results.csv'), metric, sort_by=metric)


    nn_contrast_sensitivity = get_interpolation(nn_dprimes, target_d, metric_values)
    oo_contrast_sensitivity = get_interpolation(oo_dprimes, target_d, metric_values)

    try:
        svm_dprimes = get_csv_column(os.path.join(fpath, 'svm_results_seeded.csv'), 'dprime_accuracy', sort_by=metric)
    except:
        svm_dprimes = get_csv_column(os.path.join(fpath, 'svm_results.csv'), 'dprime_accuracy', sort_by=metric)

    svm_contrast_sensitivity = get_interpolation(svm_dprimes, target_d, metric_values)
    return oo_contrast_sensitivity, nn_contrast_sensitivity, svm_contrast_sensitivity


def write_results_to_csv(results, super_path):
    """"""
    csv_name = f'contrast_sensitivities_{os.path.basename(super_path)}.csv'
    csv_path = os.path.join(super_path, csv_name)
    with open(csv_path, 'w', newline='') as f:
        cols = list(results[list(results.keys())[0]].keys()) + ['mean', 'std']
        io_rows = [f'io_{r}' for r in cols]
        nn_rows = [f'nn_{r}' for r in cols]
        svm_rows = [f'svm_{r}' for r in cols]
        fieldnames = ['mode'] + io_rows + nn_rows + svm_rows
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for mode in results.keys():
            res_row = {}
            res_row['mode'] = mode
            mode_results = results[mode]
            res_io = []
            res_nn = []
            res_svm = []
            for k in mode_results.keys():
                res_io.append(mode_results[k]['ideal observer'])
                res_nn.append(mode_results[k]['resnet'])
                res_svm.append(mode_results[k]['svm'])
            res_io.extend([np.mean(res_io), np.std(res_io)])
            res_nn.extend([np.mean(res_nn), np.std(res_nn)])
            res_svm.extend([np.mean(res_svm), np.std(res_svm)])
            for key, val in zip(io_rows, res_io):
                res_row[key] = val
            for key, val in zip(nn_rows, res_nn):
                res_row[key] = val
            for key, val in zip(svm_rows, res_svm):
                res_row[key] = val
            writer.writerow(res_row)
    return 0



if __name__ == '__main__':
    super_path = r'C:\Users\Fabian\Documents\rsync_csv\redo_experiments\sd_experiment'
    seed_paths = glob(os.path.join(super_path, 'sd_seed_4[3-6]'))
    results = nested_dict()
    for seed_path in seed_paths:
        seed = seed_path.split('_')[-1]
        multiloc_paths = glob(os.path.join(seed_path, '*multiloc*'))
        multiloc_paths = sorted(multiloc_paths, key=lambda x: int(x.split('_')[-1]))
        for multiloc_path in multiloc_paths:
            num_locations = multiloc_path.split('_')[-1]
            oo, nn, svm = get_contrast_sensitivity(multiloc_path, 1.5)
            results[num_locations][seed]['ideal observer'] = oo
            results[num_locations][seed]['resnet'] = nn
            results[num_locations][seed]['svm'] = svm
    write_results_to_csv(results, super_path)
    fpath = r'C:\Users\Fabian\Documents\rsync_csv\redo_experiments\sd_experiment\sd_seed_42\multiloc_16'
    get_contrast_sensitivity(fpath, target_d=1.5)
    print('done')
