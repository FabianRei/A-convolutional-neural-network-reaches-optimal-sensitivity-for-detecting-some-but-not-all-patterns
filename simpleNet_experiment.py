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

from src.data.mat_data import getMatData, matDataLoader
from src.models.simple_net import RobustNet


def test():
    allAccuracy =[]
    allWrongs = []
    for batch_idx, (data, target) in enumerate(matDataLoader(testData, testLabels, batchSize, shuffle=False)):
        data_temp = np.copy(data)
        data, target = Variable(data), Variable(target)
        data, target = data.cuda(), target.cuda()
        data = data.view(-1, dimIn)
        Net.eval()
        net_out = Net(data)
        prediction = net_out.max(1)[1]
        selector = (prediction != target).cpu().numpy().astype(np.bool)
        wrongs = data_temp[selector]
        testAcc = list((prediction == target).cpu().numpy())
        if not sum(testAcc) == len(target) and False:
            print(prediction.cpu().numpy()[testAcc == 0])
            print(target.cpu().numpy()[testAcc==0])
        allAccuracy.extend(testAcc)
        allWrongs.extend(wrongs)
    print(f"Test accuracy is {np.mean(allAccuracy)}")


def train():
    for epoch in range(epochs):
        epochAcc = []
        lossArr = []
        for batch_idx, (data, target) in enumerate(matDataLoader(trainData, trainLabels, batchSize, shuffle=True)):
            data, target = Variable(data), Variable(target)
            data, target = data.cuda(), target.cuda()
            data = data.view(-1, dimIn)
            optimizer.zero_grad()
            Net.train()
            net_out = Net(data)
            prediction = net_out.max(1)[1]
            loss = criterion(net_out, target)
            loss.backward()
            optimizer.step()
            currAcc = (prediction == target).cpu().numpy()
            epochAcc.extend(list(currAcc))
            lossArr.append(loss.data.item())
        print(f"Train epoch: {epoch}, loss is {np.mean(lossArr)}, accuracy is {np.mean(epochAcc)}")
        if epoch % test_interval == 0:
            test()


# relevant variables
dimIn = 249*249
dimOut = 2
learning_rate = 0.000001
epochs = 200
log_interval = 2
test_interval = 2
batchSize = 800
pathMat = "data/processed/freq_8_contrast_001_sample.mat"

data, labels = getMatData(pathMat, shuffle=True, extraNoise=True)
data = torch.from_numpy(data).type(torch.float32)
pickle.dump([data, labels], open('mat1PercentData.p', 'wb'))
# data, labels = pickle.load(open("mat1PercentData.p", 'rb'))
dimIn = data[0].shape[1]**2
# denoise data!
dataStd = data.std()
dataMean = data.mean()
data = data-dataMean
data[data>dataStd] = 0
data[data<-dataStd] = 0
data = data.sign()*(data.abs()+1).log()
# Image.fromarray(data[4]*(255/20)).show()
labels = torch.from_numpy(labels.astype(np.long))
testData = data[:int(len(data)*0.2)]
testLabels = labels[:int(len(data)*0.2)]
trainData = data[int(len(data)*0.2):]
trainLabels = labels[int(len(data)*0.2):]

Net = RobustNet(dimIn=dimIn, dimOut=dimOut, dropout=0.2)
Net.cuda()
Net.load_state_dict(torch.load('trained_RobustNet_denoised.torch'))
print(Net)

optimizer = optim.Adam(Net.parameters(), lr=learning_rate)
criterion = nn.NLLLoss()
# Train the network
train()

learning_rate=0.0000001
optimizer = optim.Adam(Net.parameters(), lr=learning_rate, amsgrad=True)
epochs = 50

# Train the network more
train()

torch.save(Net.state_dict(), "trained_RobustNet_denoised.torch")

print('\nDone!')
