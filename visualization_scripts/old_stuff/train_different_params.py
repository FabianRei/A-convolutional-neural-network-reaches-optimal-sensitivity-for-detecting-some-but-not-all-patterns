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

def sloppy_get_unique_contrasts(super_folder):
    mode = 'train'
    path_folders = [p.path for p in os.scandir(super_folder) if p.is_dir()]
    for path_folder in path_folders:
        path_csv = os.path.join(path_folder, f'{mode}_results.csv')
        contrast_im = get_csv_column(path_csv, 'contrast', sort_by=None)
        contrast_unique = np.unique(contrast_im)
        return contrast_unique

def visualize_training_specific_contrast(modes, super_folder, contrast=0.00041461695597968157):
    path_folders = [p.path for p in os.scandir(super_folder) if p.is_dir()]
    for mode in modes:
        fname = os.path.basename(f'{os.path.basename(super_folder)}_{mode}')
        fig = plt.figure()
        # plt.grid(which='both')
        plt.xscale('linear')
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.title(f'{fname} - {mode} contrast is {contrast:4f}')
        plt.grid()
        for path_folder in path_folders:
            path_csv = os.path.join(path_folder, f'{mode}_results.csv')
            acc_im = get_csv_column(path_csv, 'accuracy', sort_by=None)
            epoch_im = get_csv_column(path_csv, 'epoch', sort_by=None)
            contrast_im = get_csv_column(path_csv, 'contrast', sort_by=None)
            contrast_unique = np.unique(contrast_im)

            acc_ep_con_im = []

            for c in contrast_unique:
                if not c == contrast:
                    continue
                tmp_epoch = epoch_im[contrast_im==c]
                tmp_acc = acc_im[contrast_im==c]
                tmp_contrast = contrast_im[contrast_im==c]
                srt_idx =np.argsort(tmp_epoch)
                acc_ep_con_im.append([tmp_acc, tmp_epoch, tmp_contrast])

            for ac, ep, con in acc_ep_con_im:
                plt.plot(list(range(1, 1+len(ep))), ac, label=f'Acc on {os.path.basename(path_folder)})', alpha=0.33)

        plt.legend(frameon=True, framealpha=0.2, prop={'size': 5}, loc='upper center')
        fig.savefig(os.path.join(super_folder, f'{fname}_progression_for_contrast_{contrast:.5f}.png'), dpi=400)

        # fig.show()
        print('done!')


def visualize_training(modes, path_folder):
    for mode in modes:
        path_csv = os.path.join(path_folder, f'{mode}_results.csv')
        acc_im = get_csv_column(path_csv, 'accuracy', sort_by=None)
        epoch_im = get_csv_column(path_csv, 'epoch', sort_by=None)
        contrast_im = get_csv_column(path_csv, 'contrast', sort_by=None)
        contrast_unique = np.unique(contrast_im)

        acc_ep_con_im = []

        for c in contrast_unique:
            tmp_epoch = epoch_im[contrast_im==c]
            tmp_acc = acc_im[contrast_im==c]
            tmp_contrast = contrast_im[contrast_im==c]
            srt_idx =np.argsort(tmp_epoch)
            acc_ep_con_im.append([tmp_acc, tmp_epoch, tmp_contrast])

        fname = os.path.basename(path_folder)
        fig = plt.figure()
        # plt.grid(which='both')
        plt.xscale('linear')
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.title(f'{fname} - {mode}')
        plt.grid()

        for ac, ep, con in acc_ep_con_im:
            plt.plot(list(range(1, 1+len(ep))), ac, label=f'Acc on contrast of {con[1]}', alpha=0.33)

        plt.legend(frameon=True, framealpha=0.2, prop={'size': 5}, loc='upper center')

        fig.savefig(os.path.join(path_folder, f'{fname}_{mode}_progression.png'), dpi=400)

        # fig.show()
        print('done!')


if __name__ == '__main__':
    super_folder = '/share/wandell/data/reith/imagenet_training/different_training_params'
    contrasts_unique = sloppy_get_unique_contrasts(super_folder)
    for co in contrasts_unique:
        visualize_training_specific_contrast(['train', 'test'], super_folder, co)


"""
Older params:
#########################################
if __name__ == '__main__':
    super_folder = '/share/wandell/data/reith/imagenet_training/different_training_params/'
    path_folders = [p.path for p in os.scandir(super_folder) if p.is_dir()]
    for path_folder in path_folders:
        visualize_training(['train', 'test'], path_folder)
##########################################
"""
