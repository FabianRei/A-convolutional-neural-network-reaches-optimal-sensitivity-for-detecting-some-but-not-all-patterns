from torch.autograd import Variable
import numpy as np
import torch

from deepLearning.src.models.optimal_observer import get_optimal_observer_acc, calculate_discriminability_index, get_optimal_observer_hit_false_alarm, get_optimal_observer_acc_parallel, calculate_dprime

from deepLearning.src.data.mat_data import poisson_noise_loader, mat_data_loader, PoissonNoiseLoaderClass


def test(batchSize, testData, testLabels, Net, dimIn, includePredictionLabels=False, test_eval=False):
    allAccuracy =[]
    allWrongs = []
    predictions = []
    labels = []
    if test_eval:
        Net.eval()
    for batch_idx, (data, target) in enumerate(mat_data_loader(testData, testLabels, batchSize, shuffle=False)):
        data_temp = np.copy(data)
        data, target = Variable(data), Variable(target)
        data, target = data.cuda(), target.cuda()
        # data = data.view(-1, dimIn)
        if len(data.shape) == 4:
            data = data.permute(0, 3, 1, 2)
        else:
            data = data.view(-1, 1, dimIn, dimIn)
        # Net.eval()
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
        predictions.extend(prediction.cpu().numpy())
        labels.extend(target.cpu().numpy())
    if test_eval:
        Net.train()
    print(f"Test accuracy is {np.mean(allAccuracy)}")
    if includePredictionLabels:
        return np.mean(allAccuracy), np.stack((predictions, testLabels)).T
    else:
        return np.mean(allAccuracy)


def train(epochs, batchSize, trainData, trainLabels, testData, testLabels, Net, test_interval, optimizer, criterion, dimIn):
    bestTestAcc = 0
    testAcc = 0
    Net.train()
    for epoch in range(epochs):
        epochAcc = []
        lossArr = []
        logCount = 0
        testAcc = 0
        for batch_idx, (data, target) in enumerate(mat_data_loader(trainData, trainLabels, batchSize, shuffle=True)):
            data, target = Variable(data), Variable(target)
            data, target = data.cuda(), target.cuda()
            # data = data.view(-1, dimIn)
            data = data.view(-1, 1, dimIn, dimIn)
            optimizer.zero_grad()
            # Net.train()
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
            testAcc = test(batchSize, testData, testLabels, Net, dimIn)
            if testAcc > bestTestAcc:
                bestTestAcc = testAcc
    return Net, testAcc


def train_poisson(epochs, numSamplesEpoch, batchSize, meanData, testData, testLabels, Net, test_interval, optimizer, criterion,
                  dimIn, mean_norm, std_norm, train_test_log=None, test_eval=False, PoissonDataObject=None):
    bestTestAcc = 0
    testAcc = 0
    meanData = torch.from_numpy(meanData).type(torch.float32).cuda()
    Net.train()
    if PoissonDataObject is None:
        PoissonDataObject = PoissonNoiseLoaderClass(meanData, batchSize)
    for epoch in range(epochs):
        epochAcc = []
        lossArr = []
        logCount = 0
        testAcc = 0
        predictions = []
        labels = []
        print(f"One epoch simulates {numSamplesEpoch} samples.")
        for batch_idx in range(int(np.round(numSamplesEpoch/batchSize))):
            data, target = PoissonDataObject.get_batches()
            data, target = data.cuda(), target.cuda()
            data -= mean_norm
            data /= std_norm
            # data = data.view(-1, dimIn)
            if len(data.shape) == 4:
                data = data.permute(0,3,1,2)
            else:
                data = data.view(-1, 1, dimIn, dimIn)
            optimizer.zero_grad()
            # Net.train()
            net_out = Net(data)
            prediction = net_out.max(1)[1]
            loss = criterion(net_out, target)
            loss.backward()
            optimizer.step()
            currAcc = (prediction == target).cpu().numpy()
            epochAcc.extend(list(currAcc))
            lossArr.append(loss.data.item())
            predictions.extend(prediction.cpu().numpy())
            labels.extend(target.cpu().numpy())
            if logCount % 10 == 0:
                print(f"Train epoch: {epoch} and batch number {logCount}, loss is {np.mean(lossArr)}, accuracy is {np.mean(epochAcc)}")
            logCount += 1
        print(f"Train epoch: {epoch}, loss is {np.mean(lossArr)}, accuracy is {np.mean(epochAcc)}")
        if train_test_log is not None:
            train_test_log[0].write_row(epoch=epoch, accuracy=np.mean(epochAcc), dprime=calculate_dprime(np.stack((predictions, labels)).T))
        if epoch % test_interval == 0:
            testAcc, prediction_labels = test(batchSize, testData, testLabels, Net, dimIn, includePredictionLabels=True, test_eval=test_eval)
            train_test_log[1].write_row(epoch=epoch, accuracy=testAcc, dprime=calculate_dprime(prediction_labels))
            if testAcc > bestTestAcc:
                bestTestAcc = testAcc
            #Net.train()
    return Net, testAcc
