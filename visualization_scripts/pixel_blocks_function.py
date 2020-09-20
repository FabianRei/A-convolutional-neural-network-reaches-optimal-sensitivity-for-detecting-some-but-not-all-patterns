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


def visualize_pixel_blocks(comparison_folder, block_folder, shift=False, angle=False, include_oo=True, include_nn=True,
                           include_svm=True, fname='default'):
    if shift:
        metric = 'shift'
    elif angle:
        metric = 'angle'
    else:
        metric = 'contrast'
    if fname == 'default':
        fname = f'harmonic_curve_detection_{metric}_block_comparison'

    fig = plt.figure()
    # plt.grid(which='both')
    plt.xscale('log')
    plt.xlabel(metric)
    plt.ylabel('dprime')
    num = block_folder.split('x')[-1]
    plt.title(f"Standard harmonic curve versus shuffled {num}x{num} blocks")
    plt.grid(which='both')
    folder_paths = [comparison_folder, block_folder]
    for i, folder in enumerate(folder_paths):
        if i == 0:
            appendix = ''
        elif i == 1:
            appendix = f' randomized {num}x{num} pixel blocks'
            include_oo = False
        csv1 = os.path.join(folder, 'results.csv')
        csv_svm = os.path.join(folder, 'svm_results.csv')
        oo = get_csv_column(csv1, 'optimal_observer_d_index', sort_by=metric)
        nn = get_csv_column(csv1, 'nn_dprime', sort_by=metric)
        contrasts = get_csv_column(csv1, metric, sort_by=metric)

        if include_oo:
            plt.plot(contrasts, oo, label='Ideal Observer'+appendix)
        if include_nn:
            plt.plot(contrasts, nn, label='ResNet18'+appendix)
        epsilon = 0.001
        if include_svm:
            svm = get_csv_column(csv_svm, 'dprime_accuracy', sort_by=metric)
            svm[svm >= (svm.max()-epsilon)] = oo.max()
            plt.plot(contrasts, svm, label='Support Vector Machine'+appendix)

    out_path = block_folder
    plt.legend(frameon=True, loc='upper left', fontsize='xx-small')
    fig.savefig(os.path.join(out_path, f'{fname}.png'), dpi=200)
    # fig.show()
    print('done!')


def pixel_blocks_specific_contrast(comparison_folder, block_folders, selected_contrast, shift=False, angle=False, include_oo=True, include_nn=True,
                           include_svm=True, fname='default'):
    if shift:
        metric = 'shift'
    elif angle:
        metric = 'angle'
    else:
        metric = 'contrast'
    if fname == 'default':
        fname = f'{selected_contrast}_{metric}_block_comparison'

    fig = plt.figure()
    plt.grid(which='both')
    plt.xscale('log')
    # plt.yscale('log')
    plt.xlabel('Block size')
    plt.ylabel('dprime')
    block_sizes = []
    resnet_vals = []
    svm_vals = []
    oo_vals = []
    for bf in block_folders + [comparison_folder]:
        try:
            # if int() fails, we know that we did something wrong.. :)
            block_sizes.append(int(bf.split('x')[-1]))
        except:
            # no small blocks for comparison
            block_sizes.append(238)
        bf_csv  = os.path.join(bf, 'results.csv')
        bf_svm_csv = os.path.join(bf, 'svm_results.csv')
        c_vals = get_csv_column(bf_csv, metric, sort_by=metric)
        nn_vals = get_csv_column(bf_csv, 'nn_dprime', sort_by=metric)
        svm_col_vals = get_csv_column(bf_svm_csv, 'dprime_accuracy', sort_by=metric)
        oo_col_vals = get_csv_column(bf_csv, 'optimal_observer_d_index', sort_by=metric)
        nn_val = nn_vals[np.isclose(c_vals, selected_contrast)]
        svm_val = svm_col_vals[np.isclose(c_vals, selected_contrast)]
        oo_val = oo_col_vals[np.isclose(c_vals, selected_contrast)]
        print(oo_val, block_sizes[-1])
        resnet_vals.append(nn_val)
        svm_vals.append(svm_val)
        oo_vals.append(oo_val)
    sort_idxs = np.argsort(block_sizes)
    block_sizes = np.array(block_sizes)[sort_idxs]
    svm_vals = np.array(svm_vals)[sort_idxs]
    resnet_vals = np.array(resnet_vals)[sort_idxs]
    oo_vals = np.array(oo_vals)[sort_idxs]

    plt.title(f"ResNet18, Ideal Observer and SVM performance for various block sizes")
    if include_oo:
        plt.plot(block_sizes, oo_vals, label='Ideal Observer')
    if include_nn:
        plt.plot(block_sizes, resnet_vals, label='ResNet18')
    if include_svm:
        plt.plot(block_sizes, svm_vals, label='Support Vector Machine')

    out_path = os.path.dirname(block_folders[0])
    plt.legend(frameon=True, loc='upper left', fontsize='xx-small')
    fig.savefig(os.path.join(out_path, f'{fname}.png'), dpi=200)
    # fig.show()
    print('done!')


if __name__ == "__main__":
    selected_contrast = 6.30957344e-04
    block_folders = [f.path for f in os.scandir(r'C:\Users\Fabian\Documents\data\rsync\redo_experiments\shuffled_pixels\different_patch_sizes') if f.is_dir()]
    comparison_folder = r'C:\Users\Fabian\Documents\data\rsync\redo_experiments\shuffled_pixels\experiment_patches_238x238'
    pixel_blocks_specific_contrast(comparison_folder, block_folders, selected_contrast)
    for block_folder in block_folders:
        visualize_pixel_blocks(comparison_folder, block_folder)
