import csv
from glob import glob
import numpy as np

fp = r'C:\Users\Fabian\Documents\data\rsync\redo_experiments\mtf_experiments\mtf_contrast_new_freq_less_contrast'
fp = r'C:\Users\Fabian\Documents\data\rsync\redo_experiments\mtf_experiments\mtf_angle_new_freq_better_csv'
fp = r'C:\Users\Fabian\Documents\data\rsync\redo_experiments\face_experiment_reanalyze_csv'
fp = r'C:\Users\Fabian\Documents\data\rsync\redo_experiments\shuffled_pixels\redo_shuffle_blocks_csv'

csv_files = glob(f'{fp}\\**\\*results.csv', recursive=True)

for file in csv_files:
    with open(file, 'r') as f:
        lines=f.readlines()
        lines2 = lines[1:]
        lines2.sort(key=lambda x: float(x.split(';')[-2]))
        # lines2 = lines2[:-2]
        new_lines2 = []
        for l in lines2:
            l = l.split(';')
            shift = float(l[-2])
            # shift /= np.pi
            l[-2] = str(shift)
            new_lines2.append(';'.join(l))
        lines2 = new_lines2
        lines1 = lines[:1]
        lines1.extend(lines2)
        lines = lines1
        lines = [line.replace(';', ',') for line in lines]
    with open (file, 'w') as f:
        f.writelines(lines)
print('done')