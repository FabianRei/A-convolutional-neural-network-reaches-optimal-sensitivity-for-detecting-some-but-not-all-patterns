import numpy as np
import os
import bisect
import pandas as pd

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


def get_contrast_sensitivity(experiment, target_d=1.5, shift=False, angle=False, disks=False, calc_faces=False, calc_automata=False, calc_random=False):
    # functionality for other metrics (contrast is standard)
    if shift:
        metric = 'shift'
    elif angle:
        metric = 'angle'
    else:
        metric = 'contrast'

    # load values from csv
    nn_dprimes = get_csv_column(os.path.join(experiment, 'results.csv'), 'nn_dprime', sort_by=metric)
    oo_dprimes = get_csv_column(os.path.join(experiment, 'results.csv'), 'optimal_observer_d_index', sort_by=metric)
    try:
        svm_dprimes = get_csv_column(os.path.join(experiment, 'svm_results_seeded.csv'), 'dprime_accuracy', sort_by=metric)
    except:
        svm_dprimes = get_csv_column(os.path.join(experiment, 'svm_results.csv'), 'dprime_accuracy', sort_by=metric)
    metric_values = get_csv_column(os.path.join(experiment, 'results.csv'), metric, sort_by=metric)
    if shift:
        metric_values /= (np.pi*2)
    if angle:
        metric_values /= 2

    # contrast sensitivity nn
    right_target = bisect.bisect(nn_dprimes, target_d)
    left_target = right_target - 1
    p_val = (target_d - nn_dprimes[left_target]) / (nn_dprimes[right_target] - nn_dprimes[left_target])
    interpolated_val = (1 - p_val) * metric_values[left_target] + p_val * metric_values[right_target]
    nn_bilinear_target = interpolated_val

    # contrast sensitivity oo
    right_target = bisect.bisect(oo_dprimes, target_d)
    left_target = right_target - 1
    p_val = (target_d - oo_dprimes[left_target]) / (oo_dprimes[right_target] - oo_dprimes[left_target])
    interpolated_val = (1 - p_val) * metric_values[left_target] + p_val * metric_values[right_target]
    oo_bilinear_target = interpolated_val

    # contrast sensitivity svm
    right_target = bisect.bisect(svm_dprimes, target_d)
    left_target = right_target - 1
    p_val = (target_d - svm_dprimes[left_target]) / (svm_dprimes[right_target] - svm_dprimes[left_target])
    interpolated_val = (1 - p_val) * metric_values[left_target] + p_val * metric_values[right_target]
    svm_bilinear_target = interpolated_val

    # return one over target contrast
    return 1/nn_bilinear_target, 1/oo_bilinear_target, 1/svm_bilinear_target


if __name__ == "__main__":
    folder_path = r'C:\Users\Fabian\Documents\data\rsync\redo_experiments\sd_experiment\sd_seed_45'
    output = os.path.join(folder_path, f'{os.path.basename(folder_path)}_contrast_sensitivity.csv')
    # delete csv file, if it already exists
    if os.path.exists(output):
        os.remove(output)

    experiment_paths = [f.path for f in os.scandir(folder_path) if f.is_dir()]
    experiment_paths.sort()
    experiments = []
    nns = []
    oos = []
    svms = []
    for experiment in experiment_paths:
        try:
            nn, oo, svm = get_contrast_sensitivity(experiment)
        except:
            continue
        experiments.append(os.path.basename(experiment))
        nns.append(nn)
        oos.append(oo)
        svms.append(svm)

    # write to csv file
    df = pd.DataFrame({'experiment': experiments, 'oo': oos, 'nn': nns, 'svm': svms})
    df.to_csv(output)
    print('done')

