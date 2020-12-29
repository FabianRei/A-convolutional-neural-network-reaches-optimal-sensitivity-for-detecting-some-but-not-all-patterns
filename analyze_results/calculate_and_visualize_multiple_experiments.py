import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import os
import bisect
import types
from matplotlib.ticker import ScalarFormatter
import itertools
from src.analysis. weibull_alphas import ScaledWeibull, get_weibull_interpolation
import csv
from on_root_path import on_root_path


def line_styler(offset_default=2, style=(2, 2)):
    """
    set line style of visualization when lines than good colors.
    :param offset_default:
    :param style:
    :return:
    """
    offset = offset_default
    yield '-'
    while True:
        if offset == 0:
            offset = offset_default
        else:
            offset = 0
        yield (offset, style)


def get_csv_column(csv_path, col_name, sort_by=None, exclude_from=None):
    '''
    Extract column from CSV table file by column header name.
    :param csv_path:
    :param col_name:
    :param sort_by:
    :param exclude_from:
    :return:
    '''
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


def visualize_pixel_blocks(block_folder, shift=False, angle=False, include_oo=True, include_nn=True,
                           include_svm=True, fname='default', plot_style='default', use_legend=True):
    if shift:
        metric = 'shift'
    elif angle:
        metric = 'angle'
    else:
        metric = 'contrast'
    if fname == 'default':
        if not use_legend:
            fname = f'harmonic_curve_detection_{metric}_comparison_no_legend'
        else:
            fname = f'harmonic_curve_detection_{metric}_comparison'

    if plot_style == 'default':
        line_style = line_styler()
    else:
        line_style=plot_style
    fig = plt.figure()
    # plt.grid(which='both')
    plt.xscale('log')
    if shift:
        plt.xlabel('Shift in radians')
    elif angle:
        plt.xlabel('Angle in radians')
    else:
        plt.xlabel('Contrast')
    plt.ylabel('d-prime')
    num = block_folder.split('_')[-1]
    plt.title(f"Harmonic frequency of {num} performance for various {metric} values")
    plt.grid(which='both')
    folder_paths = [block_folder]
    for i, folder in enumerate(folder_paths):
        if i == 0:
            appendix = ''
        elif i == 1:
            appendix = f' {num} random pixels were shuffled'
            include_oo = False
        csv1 = os.path.join(folder, 'results.csv')
        csv_svm = os.path.join(folder, 'svm_results_seeded.csv')
        oo = get_csv_column(csv1, 'optimal_observer_d_index', sort_by=metric)
        nn = get_csv_column(csv1, 'nn_dprime', sort_by=metric)
        contrasts = get_csv_column(csv1, metric, sort_by=metric)
        if shift:
            contrasts /= (np.pi*2)
        if angle:
            contrasts /= 2
        if isinstance(line_style, types.GeneratorType):
            if include_oo:
                plt.plot(contrasts, oo, label='Ideal Observer'+appendix, linestyle=next(line_style))
            if include_nn:
                plt.plot(contrasts, nn, label='ResNet-18'+appendix, linestyle=next(line_style))
            epsilon = 0.001
            if include_svm:
                try:
                    svm = get_csv_column(csv_svm, 'dprime_accuracy', sort_by=metric)
                except:
                    csv_svm = os.path.join(folder, 'svm_results.csv')
                    svm = get_csv_column(csv_svm, 'dprime_accuracy', sort_by=metric)
                if (svm>oo.max()-epsilon).any():
                    svm[svm >= (svm.max()-epsilon)] = oo.max()
                plt.plot(contrasts, svm, label='Support Vector Machine'+appendix, linestyle=line_style)
        else:
            if include_oo:
                plt.plot(contrasts, oo, label='Ideal Observer'+appendix, linestyle=line_style)
            if include_nn:
                plt.plot(contrasts, nn, label='ResNet-18'+appendix, linestyle=line_style)
            epsilon = 0.001
            if include_svm:
                try:
                    svm = get_csv_column(csv_svm, 'dprime_accuracy', sort_by=metric)
                except:
                    csv_svm = os.path.join(folder, 'svm_results.csv')
                    svm = get_csv_column(csv_svm, 'dprime_accuracy', sort_by=metric)
                if (svm>oo.max()-epsilon).any():
                    svm[svm >= (svm.max()-epsilon)] = oo.max()
                plt.plot(contrasts, svm, label='Support Vector Machine'+appendix, linestyle=line_style)

    out_path = block_folder
    if use_legend:
        plt.legend(frameon=True, loc='upper left', fontsize='small', framealpha=None)
    fig.savefig(os.path.join(out_path, f'{fname}.png'), dpi=200)
    # fig.show()
    print('done!')


def mtf_calc(mtf_paths, target_d=2., shift=False, angle=False, disks=False, include_oo=True, include_nn=True,
             include_svm=True, plot_style='default', calc_faces=False, calc_automata=False, calc_random=False,
             weibull_interpol=False):
    if plot_style == 'default':
        line_style = line_styler()
    else:
        line_style = plot_style
    out_path = os.path.dirname(mtf_paths[0])
    freqs = []
    nn_dprimes = []
    oo_dprimes = []
    if shift:
        metric = 'shift'
    elif angle:
        metric = 'angle'
    else:
        metric = 'contrast'


    fname = f'Modulation_transfer_function_{metric}_values_frequencies_target_d_{target_d}_new'
    if weibull_interpol:
        fname += '_weibull'
    counter = itertools.count(1)
    for p in mtf_paths:
        # freq is disk radius for disks. Just a number for faces
        if calc_faces:
            freq = next(counter)
        elif calc_automata:
            freq = int(p.split('_')[-2])
        elif calc_random:
            freq = int(p.split('x')[-1])
        else:
            freq = int(p.split('_')[-1])
        freqs.append(freq)
        nn_dprimes.append(get_csv_column(os.path.join(p, 'results.csv'), 'nn_dprime', sort_by=metric))
        oo_dprimes.append(
            get_csv_column(os.path.join(p, 'results.csv'), 'optimal_observer_d_index', sort_by=metric))

    sort_idxs = np.argsort(freqs)
    freqs, nn_dprimes, oo_dprimes = np.array(freqs), np.array(nn_dprimes), np.array(oo_dprimes)
    freqs = freqs[sort_idxs]
    nn_freqs = np.copy(freqs)
    oo_freqs = np.copy(freqs)
    nn_dprimes = nn_dprimes[sort_idxs]
    oo_dprimes = oo_dprimes[sort_idxs]

    metric_values = get_csv_column(os.path.join(mtf_paths[0], 'results.csv'), metric, sort_by=metric)
    if shift:
        metric_values /= (np.pi*2)
    if angle:
        metric_values /= 2
    nn_bilinear_targets = []
    oo_bilinear_targets = []

    for i, dprimes in enumerate(nn_dprimes):
        if weibull_interpol:
            nn_bilinear_targets.append(get_weibull_interpolation(metric_values, dprimes, target_d))
        else:
            right_target = bisect.bisect(dprimes, target_d)
            if right_target >= len(dprimes):
                nn_freqs = np.delete(nn_freqs, i)
                continue
            left_target = right_target - 1
            p_val = (target_d - dprimes[left_target]) / (dprimes[right_target] - dprimes[left_target])
            interpolated_val = (1 - p_val) * metric_values[left_target] + p_val * metric_values[right_target]
            nn_bilinear_targets.append(interpolated_val)

    for i, dprimes in enumerate(oo_dprimes):
        if weibull_interpol:
            oo_bilinear_targets.append(get_weibull_interpolation(metric_values,dprimes, target_d))
        else:
            right_target = bisect.bisect(dprimes, target_d)
            if right_target >= len(dprimes):
                oo_freqs = np.delete(oo_freqs, i)
                continue
            left_target = right_target - 1
            p_val = (target_d - dprimes[left_target]) / (dprimes[right_target] - dprimes[left_target])
            print(p_val, metric_values[left_target])
            interpolated_val = (1 - p_val) * metric_values[left_target] + p_val * metric_values[right_target]
            oo_bilinear_targets.append(interpolated_val)

    nn_bilinear_targets = np.array(nn_bilinear_targets)
    oo_bilinear_targets = np.array(oo_bilinear_targets)

    fig, ax = plt.subplots()
    for axis in [ax.yaxis]:
        axis.set_major_formatter(ScalarFormatter())
    plt.grid(which='both')
    # plt.yscale('log')
    plt.xscale('log')
    plt.yscale('log')
    if disks:
        plt.xlabel('Radius of disk signal')
    elif calc_random:
        plt.xlabel('Block size')
    else:
        plt.xlabel('Frequency (cycles/image)')

    if shift:
        plt.ylabel("Phase shift sensitivity")
    elif angle:
        plt.ylabel("Angle sensitivity")
    else:
        plt.ylabel('Contrast sensitivity')
    plt.title(f'Modulation Transfer Function - target dprime is {target_d}')
    if isinstance(line_style, types.GeneratorType):
        plt.plot(oo_freqs, 1 / oo_bilinear_targets, label='Optimal Observer', linestyle=next(line_style))
        plt.plot(nn_freqs, 1 / nn_bilinear_targets, label='ResNet-18', linestyle=next(line_style))
    else:
        plt.plot(oo_freqs, 1 / oo_bilinear_targets, label='Optimal Observer', linestyle=line_style)
        plt.plot(nn_freqs, 1 / nn_bilinear_targets, label='ResNet-18', linestyle=line_style)
    ############SVM SUPPORT#######################################
    if include_svm:
        svm_bilinear_targets = []
        svm_dprimes = []
        svm_freqs = []
        counter2 = itertools.count(1)
        for p in mtf_paths:
            if calc_faces:
                freq= next(counter2)
            elif calc_automata:
                freq = int(p.split('_')[-2])
            elif calc_random:
                freq = int(p.split('x')[-1])
            else:
                try:
                    freq = int(p.split('_')[-1])
                except:
                    freq = int(p.split('x')[-1])
            svm_freqs.append(freq)
            try:
                svm_dprimes.append(
                    get_csv_column(os.path.join(p, 'svm_results_seeded.csv'), 'dprime_accuracy', sort_by=metric))
            except:
                svm_dprimes.append(
                    get_csv_column(os.path.join(p, 'svm_results.csv'), 'dprime_accuracy', sort_by=metric))
        svm_dprimes, svm_freqs = np.array(svm_dprimes), np.array(svm_freqs)
        sort_idxs = np.argsort(svm_freqs)
        svm_dprimes = svm_dprimes[sort_idxs]
        svm_freqs = svm_freqs[sort_idxs]
        for i, dprimes in enumerate(svm_dprimes):
            if weibull_interpol:
                svm_bilinear_targets.append(get_weibull_interpolation(metric_values, dprimes, target_d))
            else:
                right_target = bisect.bisect(dprimes, target_d)
                if right_target >= len(dprimes):
                    svm_freqs = np.delete(svm_freqs, i)
                    continue
                left_target = right_target - 1
                p_val = (target_d - dprimes[left_target]) / (dprimes[right_target] - dprimes[left_target])
                interpolated_val = (1 - p_val) * metric_values[left_target] + p_val * metric_values[right_target]
                svm_bilinear_targets.append(interpolated_val)
        svm_bilinear_targets = np.array(svm_bilinear_targets)
        if isinstance(line_style, types.GeneratorType):
            plt.plot(svm_freqs, 1 / svm_bilinear_targets, label='Support Vector Machine', linestyle=next(line_style))
        else:
            plt.plot(svm_freqs, 1 / svm_bilinear_targets, label='Support Vector Machine', linestyle=line_style)
    ################################################################
    plt.legend(frameon=True, loc='best', fontsize='small')
    fig.savefig(os.path.join(out_path, f'{fname}.png'), dpi=200)
    ###############
    csv_dict = {'svm': 1 / svm_bilinear_targets, 'io': 1 / oo_bilinear_targets, 'resnet': 1 / nn_bilinear_targets}
    with open(os.path.join(out_path, f'{fname}.csv'), 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_dict.keys()))
        writer.writeheader()
        for i in range(len(nn_bilinear_targets)):
            temp_dict = {}
            for k, v in csv_dict.items():
                temp_dict[k] = v[i]
            writer.writerow(temp_dict)



if __name__ == "__main__":
    mtf_paths = [f.path for f in os.scandir(os.path.join(on_root_path(), 'local', 'experiment', 'faces')) if f.is_dir()]
    for scope_folder in mtf_paths:
        visualize_pixel_blocks(scope_folder)

    mtf_calc(mtf_paths, target_d=1.5)
    mtf_calc(mtf_paths, target_d=2)
    mtf_calc(mtf_paths, target_d=1)
    mtf_calc(mtf_paths, target_d=3)
