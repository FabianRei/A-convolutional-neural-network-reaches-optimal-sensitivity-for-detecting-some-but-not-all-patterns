from deepLearning.src.models.resnet_train_test import train, test
from deepLearning.src.models.GrayResNet import GrayResnet18
from deepLearning.src.models.optimal_observer import get_optimal_observer_acc
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import pickle
import numpy as np
from PIL import Image
import pickle
from scipy.stats import lognorm
import torchvision.models as models

from deepLearning.src.data.mat_data import get_mat_data, get_h5data

# relevant variables
test_interval = 2
batchSize = 128
pathMat = "/black/localhome/reith/Desktop/projects/WLDiscriminationNetwork/deepLearning/data/mat_files/1000_samplesPerClass_freq_1-2-3-4_contrast_0_001.h5"
networkWeights = "trained_ResNet_with_noise_0_001.torch"

data, labels, meanData, meanDataLabels = get_h5data(pathMat, shuffle=True)
# data = torch.from_numpy(data).type(torch.float32)
# pickle.dump([data, labels, dataNoNoise], open('mat1PercentNoNoiseData.p', 'wb'))
# data, labels, dataNoNoise = pickle.load(open("mat1PercentData.p", 'rb'))
# Image.fromarray(data[4]*(255/20)).show()

dimIn = data[0].shape[1]
dimOut = len(meanData)
labels = torch.from_numpy(labels.astype(np.long))
data = torch.from_numpy(data).type(torch.float32)
Net = GrayResnet18(dimOut)
Net.cuda()
# print(Net)
Net.load_state_dict(torch.load(networkWeights))

accOptimal = get_optimal_observer_acc(data, labels, meanData)
print(f"Optimal observer accuracy is {accOptimal*100:.2f}%")


# Test the network
testAcc = test(batchSize, data, labels, Net, dimIn)
print(f"Resnet accuracy is {testAcc*100:.2f}%")


print('\nDone!')
