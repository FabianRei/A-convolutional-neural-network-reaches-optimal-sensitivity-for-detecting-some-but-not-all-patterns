from deepLearning.src.models.resnet_train_test import train, test
from deepLearning.src.models.GrayResNet import GrayResnet18
from deepLearning.src.models.optimal_observer import get_optimal_observer_acc, calculate_discriminability_index, get_optimal_observer_hit_false_alarm
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
pathMat = "/black/localhome/reith/Desktop/projects/WLDiscriminationNetwork/deepLearning/data/mat_files/10000_samplesPerClass_freq_1_contrast_0_001.h5"

data, labels, meanData, meanDataLabels = get_h5data(pathMat, shuffle=True)
# data = torch.from_numpy(data).type(torch.float32)
# pickle.dump([data, labels, dataNoNoise], open('mat1PercentNoNoiseData.p', 'wb'))
# data, labels, dataNoNoise = pickle.load(open("mat1PercentData.p", 'rb'))
# Image.fromarray(data[4]*(255/20)).show()

accOptimal = get_optimal_observer_acc(data, labels, meanData)
print(f"Optimal observer accuracy on all data is {accOptimal*100:.2f}%")

d1 = calculate_discriminability_index(meanData)
print(f"Theoretical d index is {d1}")

d2 = get_optimal_observer_hit_false_alarm(data, labels, meanData)
print(f"Optimal observer d index is {d2}")

dimIn = data[0].shape[1]
dimOut = len(meanData)
labels = torch.from_numpy(labels.astype(np.long))
data = torch.from_numpy(data).type(torch.float32)
testData = data[:int(len(data)*0.2)]
testLabels = labels[:int(len(data)*0.2)]
trainData = data[int(len(data)*0.2):]
trainLabels = labels[int(len(data)*0.2):]

accOptimal = get_optimal_observer_acc(testData, testLabels, meanData)
print(f"Optimal observer accuracy is {accOptimal*100:.2f}%")



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

Net, bestTestAccStep = train(epochs, batchSize, trainData, trainLabels, testData, testLabels, Net, test_interval, optimizer, criterion, dimIn)
bestTestAcc = max(bestTestAcc, bestTestAccStep)

print(f"Best accuracy to date is {bestTestAcc*100:.2f} percent")

# Train the network more
epochs = 4
learning_rate = 0.0001
optimizer = optim.Adam(Net.parameters(), lr=learning_rate)

Net, bestTestAccStep = train(epochs, batchSize, trainData, trainLabels, testData, testLabels, Net, test_interval, optimizer, criterion, dimIn)
bestTestAcc = max(bestTestAcc, bestTestAccStep)

print(f"Best ResNet accuracy is {bestTestAcc*100:.2f}%")
print(f"Optimal observer accuracy is {accOptimal*100:.2f}%")
print(f"Optimal observer d index is {d2}")
print(f"Theoretical d index is {d1}")


torch.save(Net.state_dict(), "trained_ResNet_with_noise_0_002_2class.torch")

print('\nDone!')
