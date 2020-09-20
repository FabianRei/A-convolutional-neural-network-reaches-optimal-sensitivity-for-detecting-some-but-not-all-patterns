from glob import glob
import pandas as pd
import os
import numpy as np
import re

# get csv file paths
data_folder = r'C:\Users\Fabian\Documents\data\rsync\redo_experiments\sd_experiment'
cs_files = glob(os.path.join(data_folder, rf'**\*sd_seed*contrast_sensitivity.csv'))

# read csv files
data = []
seeds = []
for f in cs_files:
    seeds.append(re.findall(r'\d{2}', f)[0])
    data.append(pd.read_csv(f))

# remove csv files that aren't complete
cleaned_data = []
max_len = -1
for d in data:
    if len(d) > max_len:
        max_len = len(d)
for d in data:
    if len(d) == max_len:
        cleaned_data.append(d)
data = cleaned_data

oos = []
nns = []
svms = []
exps = []
result = {}
for i, (d, seed) in enumerate(zip(data, seeds)):
    exp = d['experiment'].values
    oo = d['oo'].values
    nn = d['nn'].values
    svm = d['svm'].values
    sort_idxs = np.argsort(exp)
    exp = exp[sort_idxs]
    oo = oo[sort_idxs]
    nn = nn[sort_idxs]
    svm = svm[sort_idxs]
    nns.append(nn)
    svms.append(svm)
    oos.append(oo)
    if i == 0:
        result['experiments'] = exp
    result[f'ideal_observer_{seed}'] = oo
    result[f'resnet18_{seed}'] = nn
    result[f'svm_{seed}'] = svm

oos = np.array(oos); nns = np.array(nns); svms = np.array(svms)
result['ideal_observer_mean'] = oos.mean(axis=0)
result['ideal_observer_std'] = oos.std(axis=0)
result['resnet18_mean'] = nns.mean(axis=0)
result['resnet18_std'] = nns.std(axis=0)
result['svm_mean'] = svms.mean(axis=0)
result['svm_std'] = svms.std(axis=0)

df = pd.DataFrame(result)
df.to_csv(os.path.join(data_folder, 'sd_contrast_sensitivity_analysis.csv'))
print('done')