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

from .src.data.mat_data import get_mat_data, mat_data_loader
from .src.models.simple_net import RobustNet


def test():
    allAccuracy =[]
    allWrongs = []
    for batch_idx, (data, target) in enumerate(mat_data_loader(testData, testLabels, batchSize, shuffle=False)):
        data_temp = np.copy(data)
        data, target = Variable(data), Variable(target)
        data, target = data.cuda(), target.cuda()
        # data = data.view(-1, dimIn)
        data = data.view(-1, 1, dimIn, dimIn)
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
        logCount = 0
        for batch_idx, (data, target) in enumerate(mat_data_loader(trainData, trainLabels, batchSize, shuffle=True)):
            data, target = Variable(data), Variable(target)
            data, target = data.cuda(), target.cuda()
            # data = data.view(-1, dimIn)
            data = data.view(-1, 1, dimIn, dimIn)
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
            if logCount % 10 == 0:
                print(f"Train epoch: {epoch} and batch number {logCount}, loss is {np.mean(lossArr)}, accuracy is {np.mean(epochAcc)}")
            logCount += 1
        print(f"Train epoch: {epoch}, loss is {np.mean(lossArr)}, accuracy is {np.mean(epochAcc)}")
        if epoch % test_interval == 0:
            test()


# relevant variables
dimOut = 2
learning_rate = 0.001
epochs = 50
log_interval = 2
test_interval = 2
batchSize = 128
pathMat = "data/processed/freq_8_contrast_001_sample.mat"

# data, labels = get_mat_data(pathMat, shuffle=True, extraNoise=True)
# data = torch.from_numpy(data).type(torch.float32)
# pickle.dump([data, labels], open('mat1PercentNoNoiseData.p', 'wb'))
data, labels = pickle.load(open("mat1PercentData.p", 'rb'))
dimIn = data[0].shape[1]
# denoise data!
# dataStd = data.std()
# dataMean = data.mean()
# data = data-dataMean
# data[data>dataStd] = 0
# data[data<-dataStd] = 0
# data = data.sign()*(data.abs()+1).log()
# Image.fromarray(data[4]*(255/20)).show()
labels = torch.from_numpy(labels.astype(np.long))
testData = data[:int(len(data)*0.2)]
testLabels = labels[:int(len(data)*0.2)]
trainData = data[int(len(data)*0.2):]
trainLabels = labels[int(len(data)*0.2):]

# Net = RobustNet(dimIn=dimIn, dimOut=dimOut, dropout=0.2)

class GrayResnet18(models.ResNet):
    def __init__(self, dimOut, dropout):
        super(GrayResnet18, self).__init__(models.resnet.BasicBlock, [2, 2, 2, 2], num_classes=1000)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.extraFc1 = nn.Linear(1000, 200)
        self.drop1 = nn.Dropout(p=dropout)
        self.extraFc2 = nn.Linear(200, dimOut)

    def forward(self, x):
        x = super(GrayResnet18, self).forward(x)
        x = F.relu(self.extraFc1(x))
        x = self.drop1(x)
        x = self.extraFc2(x)
        return F.log_softmax(x, dim=1)


Net = GrayResnet18(dimOut, dropout=0.8)
Net.cuda()
# Net.load_state_dict(torch.load('trained_RobustNet_denoised.torch'))
print(Net)

optimizer = optim.Adam(Net.parameters(), lr=learning_rate)
criterion = nn.NLLLoss()
# Train the network
train()

learning_rate=0.0001
optimizer = optim.Adam(Net.parameters(), lr=learning_rate, amsgrad=True)
epochs = 50

# Train the network more
train()

torch.save(Net.state_dict(), "trained_ResNEt_with_noise.torch")

print('\nDone!')
