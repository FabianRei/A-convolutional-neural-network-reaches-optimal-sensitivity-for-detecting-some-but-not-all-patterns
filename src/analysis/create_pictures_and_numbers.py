from glob import glob
from src.data.mat_data import get_h5mean_data
from matplotlib import pyplot as plt
import scipy.misc
import os
from imageio import imsave
from glob import glob
import numpy as np


def show(arr):
    plt.imshow(arr, cmap='gray')
    plt.show()


def get_outpath(h5):
    p = os.path.dirname(h5)
    split = p.split('redo_experiments')
    outpath = os.path.join(split[0], 'redo_experiments', 'signal_images', split[1][1:])
    outpath = "\\\\?\\" + outpath
    return outpath

def get_block_size(h5):
    try:
        block_size = int(h5.split('_patches_')[1].split('x')[0])
    except:
        block_size = False
    return block_size


def create_txt(f_folder, fname, signal):
    out_txt = os.path.join(f_folder, f'txt_{fname}.txt')
    text = {}
    text['s_mean'] = np.mean(signal)
    text['s_var']= np.var(signal)
    text['s_max'] = np.max(signal)
    text['s_min'] = np.min(signal)
    text['s_shape'] = str(signal.shape)
    with open(out_txt, 'w') as t:
        for key, val in text.items():
            t.write(f'{key} is {val} \n')

exp_path = r'C:\Users\Fabian\Documents\data\rsync\redo_experiments'
all_h5 = glob(f'{exp_path}\\**\\**.h5', recursive=True)
all_h5 = glob(f'{exp_path}\\mtf*\\**\\**.h5', recursive=True)
shift_path = r'C:\Users\Fabian\Documents\data\rsync\redo_experiments\mtf_experiments\mtf_angle_new_freq'
shift_path = r'C:\Users\Fabian\Documents\data\rsync\redo_experiments\disks_mtf_experiment'
shift_path = r'C:\Users\Fabian\Documents\data\rsync\redo_experiments\redo_automaton\plain_automata'
shift_path = r'C:\Users\Fabian\Documents\data\rsync\redo_experiments\more_nn'

all_h5 = glob(f'{shift_path}\\**\\**.h5', recursive=True)

unique_h5 = []
used_h5 = []
for h5 in all_h5:
    f = os.path.basename(h5)
    if not f in used_h5:
        used_h5.append(f)
        unique_h5.append(h5)

print('nice"')

for h5 in all_h5:
    shuffled_pixels = get_block_size(h5)
    try:
        data = get_h5mean_data(h5, shuffled_pixels=shuffled_pixels)
    except:
        continue
    signal = data[0][1]
    # signal = scipy.misc.imresize(signal, 4.)
    out_path = get_outpath(h5)
    fname = os.path.basename(h5)[:-3]
    f_folder = os.path.join(out_path, fname)
    os.makedirs(f_folder, exist_ok=True)
    imsave(os.path.join(f_folder, f'{fname}.png'), signal)
    imsave(os.path.join(f_folder, f'{fname}_nosignal.png'), data[0][0])
    signal_p = np.random.poisson(signal)
    imsave(os.path.join(f_folder, f'{fname}_poisson.png'), signal_p)
    signal_p = np.random.poisson(signal) + signal
    imsave(os.path.join(f_folder, f'{fname}_mixed1.png'), signal_p)
    create_txt(f_folder, fname, signal)
    create_txt(f_folder, fname + 'no_signal', data[0][0])
    create_txt(f_folder, fname + '_poisson', np.random.poisson(signal))
    create_txt(f_folder, fname + 'no_signal_poisson', np.random.poisson(data[0][0]))
    create_txt(f_folder, fname + 'signal_part', signal-data[0][0])

print('nice!')
