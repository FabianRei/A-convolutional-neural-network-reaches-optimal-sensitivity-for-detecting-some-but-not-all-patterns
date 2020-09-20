import pickle
import numpy as np
import itertools
import numpy as np
import matplotlib.pyplot as plt
# plt.style.use('seaborn-whitegrid')
from scipy.stats import norm
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import h5py
from deepLearning.src.data.mat_data import get_h5data
import os


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    classes = [round(c, 6) for c in classes]
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

folderPath = '/share/wandell/data/reith/frequencies_experiment/'
archivePaths = [os.path.join(folderPath, f) for f in os.listdir(folderPath)]
for archivePath in archivePaths:
    archive_name = os.path.basename(archivePath)
    sLabelsPath = os.path.join(archivePath, 'contrastLabels.p')
    h5Data = h5py.File(os.path.join(archivePath, f"{os.path.basename(archivePath)}.h5"))
    h5Dict = {k: np.array(h5Data[k]) for k in h5Data.keys()}
    # shiftLabels = pickle.load(open(sLabelsPath, "rb")).astype(np.float)
    shiftLabels = np.array(h5Dict['noNoiseImgPhase']).astype(np.float)
    seconds = shiftLabels *1500/360*3600

    ooPicklePath =  os.path.join(archivePath, 'optimalOpredictionLabel.p')
    ooPredictionLabel = pickle.load(open(ooPicklePath, 'rb'))
    ooPredictions = ooPredictionLabel[:,0]
    ooLabels = ooPredictionLabel[:,1]
    print(f'Optimal Observer accuracy is {np.mean(ooPredictions==ooLabels)*100}%')
    cnf_matrix = confusion_matrix(ooLabels, ooPredictions)
    classes = np.unique(ooLabels)
    plt.figure()
    plot_confusion_matrix(cnf_matrix, seconds, title="Optimal observer confusion matrix")
    fig = plt.gcf()
    fig.set_size_inches(12,12)
    fig.savefig(os.path.join(archivePath, f'ooConfusionMatrix_{archive_name}.png'), dpi=200)
    nnPicklePath = os.path.join(archivePath, 'nnPredictionLabels.p')
    nnPredictionLabel = pickle.load(open(nnPicklePath, 'rb'))
    nnPredictions = nnPredictionLabel[:,0].astype(np.int)
    nnLabels = nnPredictionLabel[:,1].astype(np.int)
    print(f'Neural Network accuracy is {np.mean(nnPredictions==nnLabels)*100}%')

    cnf_matrix = confusion_matrix(nnLabels, nnPredictions)
    classes = np.unique(nnLabels)
    plt.figure()
    plot_confusion_matrix(cnf_matrix, seconds, title='Neural network confusion matrix')
    fig = plt.gcf()
    fig.set_size_inches(12,12)
    fig.savefig(os.path.join(archivePath, f'nnConfusionMatrix_{archive_name}.png'), dpi=200)
    print(f'Accuracy is {np.mean(nnPredictions==nnLabels)*100}%')
    nnD = []
    ooD = []
    print("Results neural network:\n")
    for i in classes:
        selector = np.where(nnPredictions==i)[0]
        hit = np.sum(nnLabels[selector] == i)/np.sum(nnLabels == i)
        false_alarm = np.sum(nnLabels[selector] != i)/np.sum(nnLabels != i)
        d = norm.ppf(hit)-norm.ppf(false_alarm)
        print(f"d' for {seconds[i]:.2f} seconds is: {d:.3f}. Hit rate is: {hit*100:.2f}% and miss rate is {false_alarm*100:.2f}%. N is {len(selector)}")
        nnD.append(d)

    print("Results optimal observer:\n")
    for i in classes:
        selector = np.where(ooPredictions==i)[0]
        hit = np.sum(nnLabels[selector] == i)/np.sum(nnLabels == i)
        false_alarm = np.sum(nnLabels[selector] != i)/np.sum(nnLabels != i)
        d = norm.ppf(hit)-norm.ppf(false_alarm)
        print(f"d' for {seconds[i]:.2f} seconds is: {d:.3f}. Hit rate is: {hit*100:.2f}% and miss rate is {false_alarm*100:.2f}%. N is {len(selector)}")
        ooD.append(d)

    plt.figure()
    plt.xscale('log')
    plt.xlabel('shift values')
    plt.ylabel('d prime')
    plt.title(f'd_prime values for {archive_name}')
    nnD = np.array(nnD)
    ooD = np.array(ooD)
    goodValsnnD = np.where(~(np.isnan(nnD) | np.isinf(nnD)))[0]
    goodValsooD = np.where(~(np.isnan(ooD) | np.isinf(ooD)))[0]
    plt.plot(seconds[goodValsnnD], nnD[goodValsnnD], label="neural network")
    plt.plot(seconds[goodValsooD], ooD[goodValsooD], label="optimal observer")
    plt.legend()
    fig = plt.gcf()
    fig.set_size_inches(6,6)
    fig.savefig(os.path.join(archivePath, f'dPrimeCurves_{archive_name}.png'), dpi=200)
    # plt.show()
