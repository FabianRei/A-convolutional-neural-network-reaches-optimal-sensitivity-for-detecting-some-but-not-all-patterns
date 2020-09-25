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
import os
import h5py


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
        # print("Normalized confusion matrix")
    else:
        pass
        # print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    if not type(classes[0]) == str:
        classes = [round(c, 6) for c in classes]
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = (cm.max()+cm.min()) / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

def high_low(archive_path, cnf_matrix, folder_name):
    """
    Looking at the fed in n x n confusion matrix, this function creates multiple 2 x 2 confusion matrices.
    This is done by comparing all neighboring 2 rows with each other.
    all values to the upper left until the diagonal confusion matrix value (which is included in the upper left sum)
    are summed together and the complement calculations are done for the other three values.
    :param archive_path: Overal output folder
    :param cnf_matrix: confusion matrix fed into the function
    :param folder_name: subfolder being created to save the created diagrams
    :return:
    """
    out_folder = os.path.join(archive_path, folder_name)
    os.makedirs(out_folder, exist_ok=True)
    path_results_dprimes = os.path.join(out_folder, 'results_dprimes.txt')
    if os.path.exists(path_results_dprimes):
        os.remove(path_results_dprimes)
    path_results_raw = os.path.join(out_folder, 'results_raw.txt')
    with open(path_results_raw, 'w') as f:
        f.write("lower_row; higher_row; low_low; low_high; high_low; high_high\n")
    for i in range(cnf_matrix.shape[0]-1):
        # format is truth_prediction
        lower_row = cnf_matrix[i]
        higher_row = cnf_matrix[i+1]
        low_low = np.sum(lower_row[:i+1])
        low_high = np.sum(lower_row[i+1:])
        high_low = np.sum(higher_row[:i+1])
        high_high = np.sum(higher_row[i+1:])
        conf_matrix = np.array([[low_low, low_high], [high_low, high_high]])
        fig = plt.figure()
        plot_confusion_matrix(conf_matrix, ['low', 'high'], title="Low high confusion matrix", normalize=True)
        fig.savefig(os.path.join(out_folder, f"{i}_{i+1}_confusion_matrix.png"))
        plt.close(fig)
        # adjust for inf and nan by adding 0.5 - Hautus, 1995
        low_low_corrected = low_low + 0.5
        low_high_corrected = low_high + 0.5
        high_low_corrected = high_low + 0.5
        high_high_corrected = high_high + 0.5
        hit_rate = low_low_corrected/(low_low_corrected+low_high_corrected)
        false_alarm_rate = high_low_corrected/(high_low_corrected+high_high_corrected)
        d_prime = norm.ppf(hit_rate)-norm.ppf(false_alarm_rate)
        with open(path_results_dprimes, 'a') as f:
            f.write(f"{i}; {i+1}; {d_prime}\n")
        with open(path_results_raw, 'a') as f:
            f.write(f"{i}; {i+1}; {low_low}; {low_high}; {high_low}; {high_high}\n")
    print(f"high_low analysis done for {out_folder}.")


folderPath = '/share/wandell/data/reith/old_freq_experiments/frequencies_experiment/'
is_shift = False

archivePaths = [os.path.join(folderPath, f) for f in os.listdir(folderPath)]
# archivePaths = ['/share/wandell/data/reith/matlabData/circle_image_rad_4_v4/']
for archivePath in archivePaths:
    print(f"Processing {archivePath}..")
    archive_name = os.path.basename(archivePath)
    if is_shift:
        h5Data = h5py.File(os.path.join(archivePath, f"{archive_name}.h5"))
        shiftLabels = np.array(h5Data['noNoiseImgPhase']).astype(np.float)
        seconds = shiftLabels *1500/360*3600
    else:
        sLabelsPath = os.path.join(archivePath, 'contrastLabels.p')
        shiftLabels = pickle.load(open(sLabelsPath, "rb")).astype(np.float)
        seconds = shiftLabels # *1500/360*3600
    ooPicklePath =  os.path.join(archivePath, 'optimalOpredictionLabel.p')
    ooPredictionLabel = pickle.load(open(ooPicklePath, 'rb'))
    ooPredictions = ooPredictionLabel[:,0]
    ooLabels = ooPredictionLabel[:,1]
    print(f'Optimal Observer accuracy is {np.mean(ooPredictions==ooLabels)*100}%')
    cnf_matrix = confusion_matrix(ooLabels, ooPredictions)
    high_low(archivePath, cnf_matrix, 'high_low_oo')
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
    high_low(archivePath, cnf_matrix, 'high_low_nn')
    classes = np.unique(nnLabels)
    plt.figure()
    plot_confusion_matrix(cnf_matrix, seconds, title='Neural network confusion matrix')
    fig = plt.gcf()
    fig.set_size_inches(12,12)
    fig.savefig(os.path.join(archivePath, f'nnConfusionMatrix_{archive_name}.png'), dpi=200)
    print(f'Accuracy is {np.mean(nnPredictions==nnLabels)*100}%')
    nnD = []
    ooD = []
    # print("Results neural network:\n")
    for i in classes:
        selector = np.where(nnPredictions==i)[0]
        hit = (0.5 + np.sum(nnLabels[selector] == i))/(np.sum(nnLabels == i) + 1)
        false_alarm = (0.5 + np.sum(nnLabels[selector] != i))/(np.sum(nnLabels != i) + 1)
        d = norm.ppf(hit)-norm.ppf(false_alarm)
        # print(f"d' for {seconds[i]:.2f} seconds is: {d:.3f}. Hit rate is: {hit*100:.2f}% and miss rate is {false_alarm*100:.2f}%. N is {len(selector)}")
        nnD.append(d)

    # print("Results optimal observer:\n")
    for i in classes:
        selector = np.where(ooPredictions==i)[0]
        hit = (0.5 + np.sum(nnLabels[selector] == i))/(np.sum(nnLabels == i) + 1)
        false_alarm = (0.5 + np.sum(nnLabels[selector] != i))/(np.sum(nnLabels != i) + 1)
        d = norm.ppf(hit)-norm.ppf(false_alarm)
        # print(f"d' for {seconds[i]:.2f} seconds is: {d:.3f}. Hit rate is: {hit*100:.2f}% and miss rate is {false_alarm*100:.2f}%. N is {len(selector)}")
        ooD.append(d)

    plt.figure()
    plt.xscale('log')
    if is_shift:
        plt.xlabel('shift values')
    else:
        plt.xlabel('contrast values')
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
    plt.close(fig='all')
