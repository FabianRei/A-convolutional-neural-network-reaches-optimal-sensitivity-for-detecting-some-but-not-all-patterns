import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import os
import types
from on_root_path import on_root_path

'''
Visualize results of ResNet performance based on number of samples used for training
'''


def line_styler(offset_default=2, style=(2, 2)):
    offset = offset_default
    yield '-'
    while True:
        if offset == 0:
            offset = offset_default
        else:
            offset = 0
        yield (offset, style)


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

def get_svm_vals(contrast_selector = 1.25892541e-03, sample_num=-1):
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


    folder = r'C:\Users\Fabian\Documents\data\rsync\redo_experiments\sample_number_contrast\svm'
    folder2 = r'C:\Users\Fabian\Documents\data\rsync\redo_experiments\sample_number_contrast\svm_1_sample'
    # contrast to select is 1.25892541e-03
    if not contrast_selector == 1.25892541e-03 and not contrast_selector ==  0.00031622776601683794:
        folder2 = folder2 + '_lower_contrast'
    elif contrast_selector ==  0.00031622776601683794:
        folder2 = folder2 + '_lowerer_contrast'

    fname = f'svm_accuracies_one_contrast_{metric}'
    epsilon = 0.001

    csv1 = os.path.join(folder, 'results.csv')
    csv_svm = os.path.join(folder, 'svm_results.csv')
    csv_svm2 = os.path.join(folder2, 'svm_results.csv')
    oo = get_csv_column(csv1, 'optimal_observer_d_index', sort_by=metric)
    nn = get_csv_column(csv1, 'nn_dprime', sort_by=metric)
    contrasts = get_csv_column(csv1, metric, sort_by=metric)

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
    if not sample_num == -1:
        svm_samples_num = np.concatenate(
        (svm[np.isclose(num_samples-5000, sample_num)], svm2[np.isclose(num_samples2-5000, sample_num)]))
        return svm_samples_num

    return svm_samples_contrast


def visualize_pixel_blocks(comparison_folder, sample_folder, shift=False, angle=False, include_oo=True, include_nn=True,
                           include_svm=True, fname='default'):
    if shift:
        metric = 'shift'
    elif angle:
        metric = 'angle'
    else:
        metric = 'contrast'
    if fname == 'default':
        fname = f'harmonic_curve_detection_{metric}_train_samples'
    line_style = line_styler()
    fig = plt.figure()
    # plt.grid(which='both')
    plt.xscale('log')
    plt.xlabel(metric)
    plt.ylabel('dprime')
    train_size = int(sample_folder.split('_')[-1])
    plt.title(f"ReseNet18 training in comparison with limited training set size training")
    plt.grid(which='both')
    folder_paths = [comparison_folder, sample_folder]
    for i, folder in enumerate(folder_paths):
        if i == 0:
            appendix = ' 300000 training samples'
        elif i == 1:
            appendix = f' {train_size} training samples'
        csv1 = os.path.join(folder, 'results.csv')
        csv_svm = os.path.join(folder, 'svm_results.csv')
        oo = get_csv_column(csv1, 'optimal_observer_d_index', sort_by=metric)
        nn = get_csv_column(csv1, 'nn_dprime', sort_by=metric)
        contrasts = get_csv_column(csv1, metric, sort_by=metric)
        if include_oo:
            plt.plot(contrasts, oo, label='Ideal Observer', linestyle=next(line_style))
            include_oo = False
        if include_nn:
            plt.plot(contrasts, nn, label='ResNet18'+appendix, linestyle=next(line_style))
        epsilon = 0.001
    if include_svm:
        svm = get_csv_column(csv_svm, 'dprime_accuracy', sort_by=metric)
        svm[svm >= (svm.max()-epsilon)] = oo.max()
        svm = get_svm_vals(sample_num = train_size)
        plt.plot(contrasts, svm, label='Support Vector Machine', linestyle=next(line_style))
    if include_oo:
        plt.plot(contrasts, oo, label='Ideal Observer', linestyle=next(line_style))

    out_path = sample_folder
    plt.legend(frameon=True, loc='upper left', fontsize='xx-small', markerscale=0.33, scatterpoints=5, scatteryoffsets=[0.5])
    fig.savefig(os.path.join(out_path, f'{fname}.png'), dpi=200)
    # fig.show()
    print('done!')


def pixel_blocks_specific_contrast(comparison_folder, sample_folders, selected_contrast, shift=False, angle=False, include_oo=True, include_nn=True,
                                   include_svm=True, fname='default', use_svm_train_sizes=False, plot_style='default'):
    if shift:
        metric = 'shift'
    elif angle:
        metric = 'angle'
    else:
        metric = 'contrast'
    if fname == 'default':
        fname = f'{selected_contrast}_selected_contrast_training_sample_comparison_nolegend_lw'
    if plot_style == 'default':
        line_style = line_styler()
    else:
        line_style = plot_style
    fig = plt.figure()
    plt.grid(which='both')
    plt.xscale('log')
    # plt.yscale('log')
    plt.xlabel('Training samples used')
    plt.ylabel('d-prime')
    train_size = []
    resnet_vals = []
    svm_vals = []
    oo_vals = []
    for bf in sample_folders + [comparison_folder]:
        try:
            # if int() fails, we know that we did something wrong.. :)
            train_size.append(int(bf.split('_')[-1]))
            # if train_size[-1] not in [1000, 1794, 3221, 5781]:
            #     train_size.pop(-1)
            #     continue
        except:
            # no small blocks for comparison
            train_size.append(300000)
        bf_csv  = os.path.join(bf, 'results.csv')
        bf_svm_csv = os.path.join(bf, 'svm_results.csv')
        c_vals = get_csv_column(bf_csv, metric, sort_by=metric)
        nn_vals = get_csv_column(bf_csv, 'nn_dprime', sort_by=metric)
        svm_col_vals = get_csv_column(bf_svm_csv, 'dprime_accuracy', sort_by=metric)
        oo_col_vals = get_csv_column(bf_csv, 'optimal_observer_d_index', sort_by=metric)
        nn_val = nn_vals[np.isclose(c_vals, selected_contrast)]
        svm_val = svm_col_vals[np.isclose(c_vals, selected_contrast)]
        oo_val = oo_col_vals[np.isclose(c_vals, selected_contrast)]
        print(oo_val, train_size[-1])
        resnet_vals.append(nn_val)
        svm_vals.append(svm_val)
        oo_vals.append(oo_val)
    sort_idxs = np.argsort(train_size)
    train_size = np.array(train_size)[sort_idxs]
    svm_vals = np.array(svm_vals)[sort_idxs]
    resnet_vals = np.array(resnet_vals)[sort_idxs]
    oo_vals = np.array(oo_vals)[sort_idxs]
    lw = 2.5
    #plt.title(f"ResNet18 performance for varying training set sizes")
    if include_oo:
        if isinstance(line_style, types.GeneratorType):
            plt.plot(train_size, oo_vals, label='Ideal Observer for reference', linestyle=next(line_style))
        else:
            plt.plot(train_size, oo_vals, label='Ideal Observer for reference', linestyle=line_style, linewidth=lw)
    if include_nn:
        if isinstance(line_style, types.GeneratorType):
            plt.plot(train_size, resnet_vals, label='ResNet18', linestyle=next(line_style))
        else:
            plt.plot(train_size, resnet_vals, label='ResNet18', linestyle=line_style, linewidth=lw)
    if include_svm:
        svm_vals = get_svm_vals(selected_contrast)
        svm_vals = [[val] for val in svm_vals]
        svm_vals = np.array(svm_vals)
        if isinstance(line_style, types.GeneratorType):
            plt.plot(train_size, svm_vals, label='Support Vector Machine', linestyle=next(line_style))
        else:
            plt.plot(train_size, svm_vals, label='Support Vector Machine', linestyle=line_style, linewidth=lw)

    out_path = os.path.dirname(sample_folders[0])
    # plt.legend(frameon=True, loc='upper left', fontsize='xx-small')
    # plt.legend(frameon=True, loc='best', fontsize='small', framealpha=0.8)
    fig.savefig(os.path.join(out_path, f'{fname}.png'), dpi=200)
    # fig.show()
    print('done!')


if __name__ == "__main__":
    selected_contrast = 6.30957344e-04
    block_folders = [f.path for f in os.scandir(os.path.join(on_root_path(), 'local', 'experiment', 'resnet')) if f.is_dir()]
    comparison_folder = os.path.join(on_root_path(), 'local', 'experiment', 'experiment_patches_238x238')
    pixel_blocks_specific_contrast(comparison_folder, block_folders, selected_contrast, plot_style='-')
    selected_contrast = 1.25892541e-03
    pixel_blocks_specific_contrast(comparison_folder, block_folders, selected_contrast, plot_style='-')
    selected_contrast = 0.00031622776601683794
    pixel_blocks_specific_contrast(comparison_folder, block_folders, selected_contrast, plot_style='-')
