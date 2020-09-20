import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import os
import bisect


def line_styler(offset_default=2, style=(2, 2), stay_standard=False, standard_style='-'):
    offset = offset_default
    if stay_standard:
        while True:
            yield standard_style
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


def visualize_result_data(folder_path, shift=False, angle=False, include_oo=True, include_nn=True,
                          include_svm=True, fname='default', title='default', frequency=1, line_style='default'):
    if shift:
        metric = 'shift'
    elif angle:
        metric = 'angle'
    else:
        metric = 'contrast'
    if fname == 'default':
        fname = f'harmonic_curve_detection_{metric}_comparison'
    if line_style == 'default':
        line_style = line_styler()
    else:
        line_style = line_styler(stay_standard=True, standard_style=line_style)
    fig = plt.figure()
    # plt.grid(which='both')
    plt.xscale('log')
    if shift:
        plt.xlabel('Shift in \u03C0')
    elif angle:
        plt.xlabel('Angle in \u03C0')
    else:
        plt.xlabel('Contrast')
    plt.ylabel('d-prime')
    # num = folder_path.split('_')[-1]
    if title == 'default':
        title = f"Harmonic frequency of {frequency} performance for various {metric} values"
    plt.title(title)
    plt.grid(which='both')
    folder_paths = [folder_path]
    for i, folder in enumerate(folder_paths):
        if i == 0:
            appendix = ''
        elif i == 1:
            appendix = f' {frequency} random pixels were shuffled'
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
        epsilon = 0.3
        if include_svm:
            svm = get_csv_column(csv_svm, 'dprime_accuracy', sort_by=metric)
            if (svm>oo.max()-epsilon).any():
                svm[svm >= (svm.max())] = oo.max()
            plt.plot(contrasts, svm, label='Support Vector Machine'+appendix, linestyle=next(line_style))

    out_path = folder_path
    plt.legend(frameon=True, loc='upper left', fontsize='small')
    fig.savefig(os.path.join(out_path, f'{fname}.png'), dpi=200)
    # fig.show()
    print('done!')


if __name__ == '__main__':

    paths = [r'C:\Users\Fabian\Documents\data\rsync\redo_experiments\shuffled_pixels\face_signal\faces_shuff_columns',
             r'C:\Users\Fabian\Documents\data\rsync\redo_experiments\shuffled_pixels\face_signal\faces_shuff_rows',
             r'C:\Users\Fabian\Documents\data\rsync\redo_experiments\shuffled_pixels\face_signal\faces_shuff_pixels']
    subfolders = []
    for pa in paths:
        folders = [p.path for p in os.scandir(pa) if p.is_dir()]
        subfolders.extend(folders)

    for p in paths:
        visualize_result_data(p, line_style='-', fname=f"{os.path.basename(p)}_performance")


r'''
Past runs:
#############################################################
if __name__ == '__main__':

    paths = [r'C:\Users\Fabian\Documents\data\rsync\redo_experiments\cellular_automaton\class_2',
             r'C:\Users\Fabian\Documents\data\rsync\redo_experiments\cellular_automaton\class_3']
    subfolders = []
    for pa in paths:
        folders = [p.path for p in os.scandir(pa) if p.is_dir()]
        subfolders.extend(folders)

    for p in subfolders:
        visualize_result_data(p, line_style='-', fname=f"{os.path.basename(p)}_performance")
##############################################################
if __name__ == '__main__':
    paths = [r'C:\Users\Fabian\Documents\data\rsync\redo_experiments\shuffled_pixels\shuffled_columns']
    for p in paths:
        visualize_result_data(p, line_style='-', fname='shuffled_columns_harmonic_curve_freq1')
##################################################################
if __name__ == '__main__':
    paths = [p.path for p in os.scandir(r'C:\Users\Fabian\Documents\data\rsync\redo_experiments\cellular_automaton') if p.is_dir()]
    # f = r'C:\Users\Fabian\Documents\data\rsync\redo_experiments\sensor_harmonic_contrasts'
    # f = r'C:\Users\Fabian\Documents\data\rsync\redo_experiments\sensor_harmonic_phase_shift'
    # f = r'C:\Users\Fabian\Documents\data\rsync\redo_experiments\sensor_harmonic_rotation'
    # f = r'C:\Users\Fabian\Documents\data\rsync\redo_experiments\face_experiment\multi_face_result'
    for p in paths:
        visualize_result_data(p, line_style='-')
###############################################################
# f = r'C:\Users\Fabian\Documents\data\rsync\redo_experiments\sensor_harmonic_contrasts'
# f = r'C:\Users\Fabian\Documents\data\rsync\redo_experiments\sensor_harmonic_phase_shift'
# f = r'C:\Users\Fabian\Documents\data\rsync\redo_experiments\sensor_harmonic_rotation'
# f = r'C:\Users\Fabian\Documents\data\rsync\redo_experiments\face_experiment\multi_face_result'
#################################################################
'''