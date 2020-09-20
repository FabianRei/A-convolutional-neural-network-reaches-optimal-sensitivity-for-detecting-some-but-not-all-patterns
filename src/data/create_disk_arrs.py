import numpy as np
import h5py
import cv2
from matplotlib import pyplot as plt
from skimage import draw
import os



def create_h5_circle(rad, out_folder):
    circle_arr = np.zeros((512, 512))
    rr, cc = draw.circle(255, 255, rad*2)
    circle_arr[rr, cc] = 1
    save_dir = os.path.join(out_folder, f'circle_with_radius_{rad}.h5')
    with h5py.File(save_dir, 'w') as f:
        dset = f.create_dataset(name='face_mat', data=circle_arr)


if __name__ == '__main__':
    rads = np.logspace(np.log10(1), np.log10(100), 8).round().astype(np.int)
    out_folder = r'C:\Users\Fabian\Documents\data\windows2rsync\windows_data\disks'
    os.makedirs(out_folder, exist_ok=True)
    for rad in rads:
        create_h5_circle(rad, out_folder)
