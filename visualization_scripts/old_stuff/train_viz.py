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


mode = 'test'
csv1 = f'/share/wandell/data/reith/imagenet_training/low_lr/freq1_harmonic_pretrained_lowerlr/{mode}_results.csv'
csv2 =  f'/share/wandell/data/reith/imagenet_training/low_lr/freq1_harmonic_random_lowerlr/{mode}_results.csv'
fname = f'{mode}ing_and_accuracy lower lr'

acc_im = get_csv_column(csv1, 'accuracy', sort_by=None)
epoch_im = get_csv_column(csv1, 'epoch', sort_by=None)
contrast_im = get_csv_column(csv1, 'contrast', sort_by=None)
contrast_unique = np.unique(contrast_im)

acc_ep_con_im = []

for c in contrast_unique:
    tmp_epoch = epoch_im[contrast_im==c]
    tmp_acc = acc_im[contrast_im==c]
    tmp_contrast = contrast_im[contrast_im==c]
    srt_idx =np.argsort(tmp_epoch)
    acc_ep_con_im.append([tmp_acc, tmp_epoch, tmp_contrast])


acc_ran = get_csv_column(csv2, 'accuracy', sort_by=None)
epoch_ran = get_csv_column(csv2, 'epoch', sort_by=None)
contrast_ran = get_csv_column(csv2, 'contrast', sort_by=None)

acc_ep_con_ran = []

for c in contrast_unique:
    tmp_epoch = epoch_ran[contrast_ran==c]
    tmp_acc = acc_ran[contrast_ran==c]
    tmp_contrast = contrast_ran[contrast_ran==c]
    acc_ep_con_ran.append([tmp_acc, tmp_epoch, tmp_contrast])


fig = plt.figure()
# plt.grid(which='both')
plt.xscale('linear')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.title('Pretrained Imagenet')
plt.grid()

for ac, ep, con in acc_ep_con_im:
    plt.plot(list(range(1, 1+len(ep))), ac, label=f'Acc on contrast of {con[1]}', alpha=0.33)

plt.legend(frameon=True, framealpha=0.2, prop={'size': 5}, loc='upper center')

fig2 = plt.figure()
# plt.grid(which='both')
plt.xscale('linear')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.title('Randomly initialized')
plt.grid()
for ac, ep, con in acc_ep_con_ran:
    plt.plot(list(range(1, 1+len(ep))), ac, label=f'Acc on contrast of {con[1]}', alpha=0.33)

plt.legend(frameon=True, framealpha=0.2, prop={'size': 5}, loc='upper center')

out_path = os.path.dirname(csv2)
fig.savefig(os.path.join(out_path, f'{fname}_pretrained_imagenet.png'), dpi=400)

fig2.savefig(os.path.join(out_path, f'{fname}_randomly_initialized.png'), dpi=400)

# fig.show()
print('done!')


"""
Older params:
#########################################
mode = 'train'
path_csv = f'/share/wandell/data/reith/imagenet_training/freq1_harmonic_pretrained/{mode}_results.csv'
csv2 =  f'/share/wandell/data/reith/imagenet_training/freq1_harmonic_random/{mode}_results.csv'
fname = f'{mode}ing_and_accuracy'
##########################################
"""
