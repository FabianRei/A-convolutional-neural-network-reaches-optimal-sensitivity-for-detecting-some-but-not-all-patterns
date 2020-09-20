from scipy.stats import poisson, norm
import scipy.io as sio
import numpy as np
import multiprocessing


def get_poisson_accuracy(dataSignal, dataNoSignal, pureSignal, pureNoSignal):
    allAccuracies = []
    for signal in dataSignal:
        llSignal = poisson.logpmf(signal, pureSignal).sum()
        llNoSignal = poisson.logpmf(signal, pureNoSignal).sum()
        if llSignal > llNoSignal:
            allAccuracies.append(1)
        else:
            allAccuracies.append(0)
    for noSignal in dataNoSignal:
        llSignal = poisson.logpmf(noSignal, pureSignal).sum()
        llNoSignal = poisson.logpmf(noSignal, pureNoSignal).sum()
        if llSignal < llNoSignal:
            allAccuracies.append(1)
        else:
            allAccuracies.append(0)
    return np.mean(allAccuracies)


def parallel_apply_along_axis(func1d, axis, arr, *args):
    """
    Like numpy.apply_along_axis(), but takes advantage of multiple
    cores.
    """
    # Effective axis where apply_along_axis() will be applied by each
    # worker (any non-zero axis number would work, so as to allow the use
    # of `np.array_split()`, which is only done on axis 0):
    effective_axis = 1 if axis == 0 else axis
    if effective_axis != axis:
        arr = arr.swapaxes(axis, effective_axis)

    # Chunks for the mapping (only a few chunks):
    chunks = [(func1d, effective_axis, sub_arr, *args)
              for sub_arr in np.array_split(arr, multiprocessing.cpu_count()//2)]

    pool = multiprocessing.Pool()
    individual_results = pool.starmap(np.apply_along_axis, chunks)
    # Freeing the workers:
    pool.close()
    pool.join()

    return np.concatenate(individual_results)


def get_optimal_observer_prediction(datum, meanData, signal_noise_even=False):
    llVals = []
    num_signals = len(meanData)-1
    for i, meanDatum in enumerate(meanData):
        log_probability = poisson.logpmf(datum, meanDatum).sum()
        # all is fine, if categories are evenly distributed. But if not, we have to add the log of their proba-
        # bility.
        if signal_noise_even:
            # if i is 0, we know it's the non signal case and use a p of 0.5
            if i == 0:
                log_probability += np.log(0.5)
            # if i > 0, we know it's a signal case
            else:
                log_probability += np.log(0.5/num_signals)
        llVals.append(log_probability)
    prediction = np.argmax(llVals)
    return prediction


def get_optimal_observer_acc_parallel(testData, testLabels, meanData, returnPredictionLabel=False, signal_noise_even=True):
    testData = testData.reshape(testData.shape[0], -1)
    meanData = meanData.reshape(meanData.shape[0], -1)
    print(f'signal_noise_even is {signal_noise_even}!')
    predictions = parallel_apply_along_axis(get_optimal_observer_prediction, 1, testData, meanData, signal_noise_even)
    # again, treat cases with more than 2 mean data arrays as multiple signal location cases.
    if meanData.shape[0] > 2:
        # predictions[predictions >= 1] = 1
        more_signals = False
        # To adjust the distribution of different classes to the ideal observer prior of evenly distributed classes
        if more_signals:
            extra_preds = []
            extra_labels = []
            for i in range(len(meanData)-2):
                extra_preds.extend(predictions[testLabels == 1])
                extra_labels.extend(testLabels[testLabels == 1])
            predictions = np.concatenate((np.array(extra_preds), predictions))
            testLabels = np.concatenate((np.array(extra_labels), testLabels))

    allAccuracies = np.mean(predictions == testLabels)
    predictionLabel = np.stack((predictions, testLabels)).T
    if returnPredictionLabel:
        return np.mean(allAccuracies), predictionLabel
    else:
        return np.mean(allAccuracies)


def calculate_dprime(prediction_label, adjust_imbalance=True):
    oo_predictions = prediction_label[:, 0]
    oo_labels = prediction_label[:, 1]
    i = 1
    selector = np.where(oo_predictions == i)[0]
    hit = (np.sum(oo_labels[selector] == i)) / (np.sum(oo_labels == i))
    false_alarm = (np.sum(oo_labels[selector] != i)) / (np.sum(oo_labels != i))
    # check if dataset classes are balanced (within a margin of 10)
    balanced = np.abs(np.sum(oo_labels)-len(oo_labels)) < 10
    if not (0<hit<1 and 0<false_alarm<1) and balanced:
        hit = (0.5 + np.sum(oo_labels[selector] == i)) / (np.sum(oo_labels == i) + 1)
        false_alarm = (0.5 + np.sum(oo_labels[selector] != i)) / (np.sum(oo_labels != i) + 1)
    # we adjust the addition of 0.5 to account for imbalance
    elif not (0<hit<1 and 0<false_alarm<1) and not balanced and adjust_imbalance:
        adjustment = np.sum(oo_labels == i)/np.sum(oo_labels != i)
        hit = (0.5*adjustment + np.sum(oo_labels[selector] == i)) / (np.sum(oo_labels == i) + 1*adjustment)
        false_alarm = (0.5 + np.sum(oo_labels[selector] != i)) / (np.sum(oo_labels != i) + 1)
    elif not (0<hit<1 and 0<false_alarm<1) and not balanced and not adjust_imbalance:
        if false_alarm == 1.0:
            print('set to 0')
            return 0
        hit = (0.5 + np.sum(oo_labels[selector] == i)) / (np.sum(oo_labels == i) + 1)
        false_alarm = (0.5 + np.sum(oo_labels[selector] != i)) / (np.sum(oo_labels != i) + 1)
    d = norm.ppf(hit) - norm.ppf(false_alarm)
    return d


def get_optimal_observer_acc(testData, testLabels, meanData, returnPredictionLabel=False, signal_noise_even=True):
    allAccuracies = []
    predictionLabel = np.empty((0, 2))
    num_signals = len(meanData)-1
    for datum, label in zip(testData, testLabels):
        llVals = []
        for i, meanDatum in enumerate(meanData):
            log_probability = poisson.logpmf(datum, meanDatum).sum()
            # all is fine, if categories are evenly distributed. But if not, we have to add the log of their proba-
            # bility.
            if signal_noise_even:
                # if i is 0, we know it's the non signal case and use a p of 0.5
                if i == 0:
                    log_probability += np.log(0.5)
                # if i > 0, we know it's a signal case
                else:
                    log_probability += np.log(1/num_signals)
            llVals.append(log_probability)
        prediction = np.argmax(llVals)
        allAccuracies.append(prediction == label)
        predictionLabel = np.append(predictionLabel, [[prediction, label]], axis=0)
        # print(f"prediction: {prediction}, label is {label}.")
    if returnPredictionLabel:
        return np.mean(allAccuracies), predictionLabel
    else:
        return np.mean(allAccuracies)

def get_optimal_observer_hit_false_alarm(testData, testLabels, meanData):
    hits = []
    falseAlarms = []
    allAccuracies = []
    predictions = []
    if len(meanData) > 2:
        return 0
    for datum, label in zip(testData, testLabels):
        llVals = []
        for meanDatum in meanData:
            llVals.append(poisson.logpmf(datum, meanDatum).sum())
        prediction = np.argmax(llVals)
        predictions.append(prediction)
        if label == 1: # signal
            hits.append(label == prediction)
        else:
            falseAlarms.append(label != prediction)
        if label != prediction:
            # print("test")
            pass
        allAccuracies.append(prediction == label)
    d = norm.ppf(np.mean(hits))-norm.ppf(np.mean(falseAlarms))
    return d


# selector = np.where(ooPredictions == i)[0]
# hit = (0.5 + np.sum(nnLabels[selector] == i)) / (np.sum(nnLabels == i) + 1)
# false_alarm = (0.5 + np.sum(nnLabels[selector] != i)) / (np.sum(nnLabels != i) + 1)
# d = norm.ppf(hit) - norm.ppf(false_alarm)

def calculate_discriminability_index(meanData):
    if len(meanData) > 2:
        return 0
    alpha = meanData[0]
    beta = meanData[1]
    d = np.sum((beta-alpha) * np.log(beta/alpha)) / np.sqrt(0.5*np.sum(((alpha+beta) * np.log(beta/alpha)**2)))
    # 8.474092465767908
    return d


if __name__ == '__main__':
    pathMat = "data/mat_files/30SamplesPerClass_freq_8_contrast_0_015_11-12-18_19_45.mat"
    matData = sio.loadmat(pathMat)
    dataSignal = np.transpose(matData['imgNoiseStimulus'], (2, 0, 1))
    dataNoSignal = np.transpose(matData['imgNoiseNoStimulus'], (2, 0, 1))
    pureSignal = np.transpose(matData['imgNoNoiseStimulus'], (2, 0, 1))[0]
    pureNoSignal = np.transpose(matData['imgNoNoiseNoStimulus'], (2, 0, 1))[0]

    accuray = get_poisson_accuracy(dataSignal, dataNoSignal, pureSignal, pureNoSignal)

    print(f"The optimal observer has {accuray*100:.2f}% accuracy on the data\n"
          f"({pathMat})")
