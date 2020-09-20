from deepLearning.src.models.resnet_train_test import train, train_poisson
from deepLearning.src.models.GrayResNet import GrayResnet18
from deepLearning.src.models.optimal_observer import get_optimal_observer_acc, calculate_discriminability_index, get_optimal_observer_hit_false_alarm
import torch
from torch.autograd import Variable
from deepLearning.src.data.logger import Logger
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import pickle
import numpy as np
from PIL import Image
import pickle
from scipy.stats import lognorm
import torchvision.models as models


from deepLearning.src.data.mat_data import get_mat_data, get_h5data, get_h5mean_data, poisson_noise_loader

# relevant variables
test_interval = 2
batchSize = 128
numSamplesEpoch = 10000
pathMat = "/black/localhome/reith/Desktop/projects/WLDiscriminationNetwork/deepLearning/data/experiment_shift_contrasts/5_samplesPerClass_freq_1_contrast_0_10_shift_1_00_pi_per_300000.h5"

meanData, meanDataLabels = get_h5mean_data(pathMat)
# data = torch.from_numpy(data).type(torch.float32)
# pickle.dump([data, labels, dataNoNoise], open('mat1PercentNoNoiseData.p', 'wb'))
# data, labels, dataNoNoise = pickle.load(open("mat1PercentData.p", 'rb'))
# Image.fromarray(data[4]*(255/20)).show()

testData, testLabels = poisson_noise_loader(meanData, size=1000, numpyData=True)
# you gotta normalize stuff, bro
# Variance is taken from testData, as it's a good enough representation. Mean is taken from mean meanData
mean = meanData.mean()
var = testData.std()

accOptimal = get_optimal_observer_acc(testData, testLabels, meanData)
print(f"Optimal observer accuracy on all data is {accOptimal*100:.2f}%")

d1 = calculate_discriminability_index(meanData)
print(f"Theoretical d index is {d1}")

d2 = get_optimal_observer_hit_false_alarm(testData, testLabels, meanData)
print(f"Optimal observer d index is {d2}")

dimIn = testData[0].shape[1]
dimOut = len(meanData)


Net = GrayResnet18(dimOut)
Net.cuda()
# print(Net)
# Net.load_state_dict(torch.load('trained_RobustNet_denoised.torch'))
criterion = nn.NLLLoss()
bestTestAcc = 0

# Test the network
# testAcc = test(batchSize, testData, testLabels, Net, dimIn)
# Train the network
epochs = 4
learning_rate = 0.001
optimizer = optim.Adam(Net.parameters(), lr=learning_rate)
testLabels = torch.from_numpy(testLabels.astype(np.long))
testData = torch.from_numpy(testData).type(torch.float32)
Net, bestTestAccStep = train_poisson(epochs, numSamplesEpoch, batchSize, meanData, testData, testLabels, Net, test_interval, optimizer, criterion, dimIn)
bestTestAcc = max(bestTestAcc, bestTestAccStep)

print(f"Best accuracy to date is {bestTestAcc*100:.2f} percent")

# Train the network more
epochs = 4
learning_rate = 0.0001
optimizer = optim.Adam(Net.parameters(), lr=learning_rate)

Net, bestTestAccStep = train_poisson(epochs, numSamplesEpoch, batchSize, meanData, testData, testLabels, Net, test_interval, optimizer, criterion, dimIn)
bestTestAcc = max(bestTestAcc, bestTestAccStep)

# Train the network more
epochs = 4
learning_rate = 0.00001
optimizer = optim.Adam(Net.parameters(), lr=learning_rate)

Net, bestTestAccStep = train_poisson(epochs, numSamplesEpoch, batchSize, meanData, testData, testLabels, Net, test_interval, optimizer, criterion, dimIn)
bestTestAcc = max(bestTestAcc, bestTestAccStep)


print(f"Best ResNet accuracy is {bestTestAcc*100:.2f}%")
print(f"Optimal observer accuracy is {accOptimal*100:.2f}%")
print(f"Optimal observer d index is {d2}")
print(f"Theoretical d index is {d1}")


torch.save(Net.state_dict(), "trained_ResNet_with_noise_0_002_2class.torch")

print('\nDone!')
