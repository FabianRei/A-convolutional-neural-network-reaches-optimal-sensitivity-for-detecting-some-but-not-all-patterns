from src.data.mat_data import get_h5mean_data
from matplotlib import pyplot as plt
import scipy.misc
import os
from imageio import imsave
from glob import glob


def show(arr):
    plt.imshow(arr, cmap='gray')
    plt.show()

folderp = r'C:\Users\Fabian\Documents\data\rsync\redo_experiments\redo_automaton\matlab_contrasts'
sub_folders = [f.path for f in os.scandir(folderp) if f.is_dir()]
for sub in sub_folders:
    fp = glob(f"{sub}\\*0_019952623150*.h5")
    fp = fp[0]
    data = get_h5mean_data(fp)
    signal = data[0][1]
    signal = scipy.misc.imresize(signal, 4.)
    out_path = os.path.dirname(fp)
    imsave(os.path.join(out_path, 'mean_signal.png'), signal)
    # scipy.misc.imsave()
    print("nice")