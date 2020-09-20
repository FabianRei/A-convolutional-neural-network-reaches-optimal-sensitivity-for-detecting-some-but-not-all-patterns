import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import os
import bisect


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


def visualize_face(block_folder, shift=False, angle=False, include_oo=True, include_nn=True,
                   include_svm=True, fname='default'):
    if shift:
        metric = 'shift'
    elif angle:
        metric = 'angle'
    else:
        metric = 'contrast'
    if fname == 'default':
        fname = f'Face_detection {metric}'
    line_style = line_styler()
    fig = plt.figure()
    # plt.grid(which='both')
    plt.xscale('log')
    plt.xlabel(metric)
    plt.ylabel('dprime')
    num = block_folder.split('_')[-1]
    plt.title(f"Face detection performance for various contrasts")
    plt.grid(which='both')
    folder_paths = [block_folder]
    for i, folder in enumerate(folder_paths):
        if i == 0:
            appendix = ''
        elif i == 1:
            appendix = f' {num} random pixels were shuffled'
            include_oo = False
        csv1 = os.path.join(folder, 'results.csv')
        csv_svm = os.path.join(folder, 'svm_results.csv')
        oo = get_csv_column(csv1, 'optimal_observer_d_index', sort_by=metric)
        nn = get_csv_column(csv1, 'nn_dprime', sort_by=metric)
        contrasts = get_csv_column(csv1, metric, sort_by=metric)

        if include_oo:
            plt.plot(contrasts, oo, label='Ideal Observer'+appendix, linestyle=next(line_style))
        if include_nn:
            plt.plot(contrasts, nn, label='ResNet18'+appendix, linestyle=next(line_style))
        epsilon = 0.001
        if include_svm:
            svm = get_csv_column(csv_svm, 'dprime_accuracy', sort_by=metric)
            if (svm>oo.max()-epsilon).any():
                svm[svm >= (svm.max()-epsilon)] = oo.max()
            plt.plot(contrasts, svm, label='Support Vector Machine'+appendix, linestyle=next(line_style))

    out_path = block_folder
    plt.legend(frameon=True, loc='upper left', fontsize='xx-small')
    fig.savefig(os.path.join(out_path, f'{fname}.png'), dpi=200)
    # fig.show()
    print('done!')


if __name__ == "__main__":
    mtf_paths = [f.path for f in os.scandir(r'C:\Users\Fabian\Documents\data\rsync\redo_experiments\face_experiment') if f.is_dir()]
    for scope_folder in mtf_paths:
        visualize_face(scope_folder)