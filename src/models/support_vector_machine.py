from sklearn import svm
import numpy as np
from deepLearning.src.data.mat_data import get_h5mean_data, poisson_noise_loader
from deepLearning.src.models.optimal_observer import calculate_dprime
import os
import csv
import datetime
import time
import pickle


def score_svm(h5_path, lock, test_data, test_labels, metric='contrast', num_samples=10000, signal_no_signal=False, random_seed=42, **kwargs):
    acc, dprime, metric_val = get_svm_accuracy(h5_path, test_data, test_labels, num_samples, lock=lock,
                                               signal_no_signal=signal_no_signal, random_seed=random_seed, **kwargs)
    write_svm_csv(acc, dprime, metric_val, os.path.dirname(h5_path), lock=lock, metric_name=metric, num_samples=num_samples)


def write_svm_csv(acc, dprime, metric, out_path, lock=None, metric_name='contrast', num_samples=10000):
    if lock is not None:
        lock.acquire()
    svm_csv = os.path.join(out_path, "svm_results_seeded.csv")
    file_exists = os.path.isfile(svm_csv)

    with open(svm_csv, 'a') as csv_file:
            headers = ['svm_accuracy', 'dprime_accuracy', metric_name, 'samples_used']
            writer = csv.DictWriter(csv_file, delimiter=';', lineterminator='\n',fieldnames=headers)
            if not file_exists:
                writer.writeheader()  # file doesn't exist yet, write a header
            writer.writerow({'svm_accuracy': acc, 'dprime_accuracy': dprime, metric_name: metric, 'samples_used': num_samples})
    if lock is not None:
        lock.release()


def get_svm_accuracy(path_mat, test_data, test_labels, num_samples=10000, lock=None, signal_no_signal=False, random_seed=42, **kwargs):
    start = time.time()
    if lock is not None:
        lock.acquire()
    meanData, meanDataLabels, dataMetric = get_h5mean_data(path_mat, **kwargs)
    if lock is not None:
        lock.release()
    train_data, train_labels = poisson_noise_loader(meanData, size=num_samples, numpyData=True, seed=random_seed*42, signal_no_signal=signal_no_signal)
    train_data = train_data.reshape(train_data.shape[0], -1)
    test_data = test_data.reshape(test_data.shape[0], -1)
    svc = svm.SVC(kernel='linear', max_iter=1000, random_state=random_seed)
    num_data = len(train_data)
    num_train = int(num_data)
    x_train, y_train = train_data, train_labels
    x_test, y_test = test_data, test_labels
    svc.fit(x_train, y_train)
    preds = svc.predict(x_test)
    acc = np.mean(preds == y_test)
    # save predictions, labels
    id_name = os.path.basename(path_mat).split('.')[0]
    out_path = os.path.dirname(path_mat)
    pickle.dump(np.stack([preds, y_test], axis=1), open(os.path.join(out_path, f"{id_name}_svm_pred_labels.p"), 'wb'))
    # score dprime
    dp_preds = (preds > 0).astype(np.int)
    dp_y_test = (y_test > 0).astype(np.int)
    dprime = calculate_dprime(np.stack([dp_preds, dp_y_test], axis=1))
    print(f'Accuracy is {acc}, Dprime is {dprime}  train samples is {num_train}, took {str(datetime.timedelta(seconds=time.time()-start))}.')
    return acc, dprime, float(dataMetric[1])


if __name__ == '__main__':
    print("starting out..")
    windows_db = True
    if windows_db:
        path_mat = r'C:\Users\Fabian\Documents\data\windows2rsync\windows_data\multiple_locations_hc\harmonic_frequency_of_1_loc_1_signalGridSize_4\1_samplesPerClass_freq_1_contrast_0_798104925988_loc_1_signalGrid_4.h5'
    else:
        path_mat = '/share/wandell/data/reith/2_class_MTF_freq_experiment/frequency_1/5_samplesPerClass_freq_1_contrast_oo_0_000414616956.h5'
    meanData, meanDataLabels, dataMetric = get_h5mean_data(path_mat, includeContrast=True)
    sample_numbers = np.logspace(np.log10(500), np.log10(50000), num=15).astype(np.int)
    test_data, test_labels = poisson_noise_loader(meanData, size=100, numpyData=True)
    # for num in sample_numbers:
    get_svm_accuracy(path_mat, test_data, test_labels, num_samples=200, includeContrast=True)
