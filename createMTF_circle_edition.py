import os
import numpy as np
import h5py
import pickle
from scipy.stats import norm
import bisect
import matplotlib.pyplot as plt

super_folder = '/share/wandell/data/reith/circles_experiment_v3/'
target_d = 2


sub_folders = [os.path.join(super_folder, f) for f in os.listdir(super_folder)]

nn_blin_target_d_points = []
oo_blin_target_d_points = []
radii = []
for archivePath in sub_folders:
    circle_radius = int(archivePath.split("_")[-1])
    h5_name = f'2_samplesPerClass_freq_4_contrast_x_image_white_circle_rad_{circle_radius}.h5'
    h5_file = h5py.File(os.path.join(archivePath, h5_name))
    contrasts = np.array(h5_file['noNoiseImgContrast']).astype(np.float)

    nnPicklePath = os.path.join(archivePath, 'nnPredictionLabels.p')
    nnPredictionLabel = pickle.load(open(nnPicklePath, 'rb'))
    nnPredictions = nnPredictionLabel[:,0].astype(np.int)
    nnLabels = nnPredictionLabel[:,1].astype(np.int)

    ooPicklePath =  os.path.join(archivePath, 'optimalOpredictionLabel.p')
    ooPredictionLabel = pickle.load(open(ooPicklePath, 'rb'))
    ooPredictions = ooPredictionLabel[:,0]
    ooLabels = ooPredictionLabel[:,1]

    prediction_classes = np.unique(ooLabels)
    nnD = []
    ooD = []
    for i in prediction_classes:
        selector = np.where(nnPredictions==i)[0]
        hit = np.sum(nnLabels[selector] == i)/np.sum(nnLabels == i)
        false_alarm = np.sum(nnLabels[selector] != i)/np.sum(nnLabels != i)
        d = norm.ppf(hit)-norm.ppf(false_alarm)
        # print(f"d' for {seconds[i]:.2f} seconds is: {d:.3f}. Hit rate is: {hit*100:.2f}% and miss rate is {false_alarm*100:.2f}%. N is {len(selector)}")
        nnD.append(d)

    print(f"processing {archivePath}..")
    for i in prediction_classes:
        selector = np.where(ooPredictions==i)[0]
        hit = np.sum(nnLabels[selector] == i)/np.sum(nnLabels == i)
        false_alarm = np.sum(nnLabels[selector] != i)/np.sum(nnLabels != i)
        d = norm.ppf(hit)-norm.ppf(false_alarm)
        # print(f"d' for {seconds[i]:.2f} seconds is: {d:.3f}. Hit rate is: {hit*100:.2f}% and miss rate is {false_alarm*100:.2f}%. N is {len(selector)}")
        ooD.append(d)

    nn_target_right = bisect.bisect(nnD, target_d)
    nn_target_left = nn_target_right -1
    nn_p_val = (target_d - nnD[nn_target_right])/(nnD[nn_target_left] - nnD[nn_target_right])
    nn_blin_target_d_point = nn_p_val * contrasts[nn_target_left] + (1-nn_p_val)* contrasts[nn_target_right]
    nn_blin_target_d_points.append(nn_blin_target_d_point)
    oo_target_right = bisect.bisect(ooD, target_d)
    oo_target_left = oo_target_right -1
    oo_p_val = (target_d - ooD[oo_target_right])/(ooD[oo_target_left] - ooD[oo_target_right])
    oo_blin_target_d_point = oo_p_val * contrasts[oo_target_left] + (1-oo_p_val)* contrasts[oo_target_right]
    oo_blin_target_d_points.append(oo_blin_target_d_point)
    radii.append(circle_radius)

radii = np.array(radii)
nn_blin_target_d_points = np.array(nn_blin_target_d_points)
oo_blin_target_d_points = np.array(oo_blin_target_d_points)

sort_indices = np.argsort(radii)
radii = radii[sort_indices]
nn_blin_target_d_points = nn_blin_target_d_points[sort_indices]
oo_blin_target_d_points = oo_blin_target_d_points[sort_indices]

plt.figure()
plt.xscale('linear')
plt.yscale('log')
plt.xlabel('circle radius')
plt.ylabel('bilinear interpolated  contrast values, where d prime reaches 2')
plt.title(f'Modulation Transfer Function for {os.path.basename(os.path.dirname(super_folder))}')

# goodValsnnD = np.where(~(np.isnan(nnD) | np.isinf(nnD)))[0]
# goodValsooD = np.where(~(np.isnan(ooD) | np.isinf(ooD)))[0]
plt.plot(radii, 1/nn_blin_target_d_points, label="neural network")
plt.plot(radii, 1/oo_blin_target_d_points, label="optimal observer")
plt.legend()
fig = plt.gcf()
fig.set_size_inches(6,6)
fig.savefig(os.path.join(super_folder, f'MTF_{os.path.basename(os.path.dirname(super_folder))}.png'), dpi=200)
print("nice")
