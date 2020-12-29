if __name__ == '__main__':
    import os, sys, inspect

    current_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))))
    parent_dir = os.path.dirname(current_dir)
    sys.path.insert(0, parent_dir)

from src.models.resnet_train_test import train_poisson, test
from src.models.GrayResNet import GrayResnet18, GrayResnet101
from src.models.optimal_observer import get_optimal_observer_acc, calculate_discriminability_index, get_optimal_observer_hit_false_alarm, get_optimal_observer_acc_parallel, calculate_dprime
from src.data.mat_data import get_h5mean_data, poisson_noise_loader, PoissonNoiseLoaderClass, shuffle_pixels as shuffle_pixels_func, shuffle_1d
from src.data.logger import Logger, CsvWriter
from src.models.support_vector_machine import score_svm
import torch
import torch.nn as nn
from torch import optim
import numpy as np
import sys
import os
import csv
import time
import datetime
import pickle
from scipy.stats import norm
import multiprocessing as mp
from src.models.new_vgg import vgg16, vgg16bn


def determine_signal_contrast_cnn_svm_io_performance(pathMat, device=None, lock=None, train_nn=True, include_shift=False,
                                                     increase_network_depth=False, oo=True, svm=True, NetClass=None, NetClass_param=None,
                                                     include_angle=False, training_csv=True, num_epochs=30, initial_lr=0.001, lr_deviation=0.1,
                                                     lr_epoch_reps=3, them_cones=False, separate_rgb=False, meanData_rounding=None,
                                                     shuffled_pixels=0, shuffle_scope=-1, test_eval=True, random_seed_nn=True, train_set_size=-1,
                                                     test_size=5000, shuffle_portion=-1, ca_rule=-1, force_balance=False,
                                                     same_test_data_shuff_pixels=True, class_balance='class_based', random_seed=42):
    """
    Calculate CNN train performance, SVM performance and Ideal Observer performance for specific signal template.
    There are a lot of parameters in this function. Some were used for the experiments described in our paper. Others
    were for experiemtns that in the end didn't make it into the paper.

    This function focuses mainly on housekeeping. It runs all the functions for training the CNN & SVM. It tests
    for performance of CNN, SVM & ideal observer and then writes the results into CSV files.
    :param pathMat: path to .h5 signal file generatd by IsetCam
    :param device: set GPU on which the CNN training is going to run on
    :param lock: lock required to ensure atomic writing to the results csv file
    :param train_nn: run also for CNN
    :param include_shift: used for shifted signal used in other experiments
    :param increase_network_depth: increases network depth. used in another experiment
    :param oo: run also for Ideal Observer
    :param svm: run also for SVM
    :param NetClass: network class, e.g. VGG or AlexNet
    :param NetClass_param: general parameter passed into the construction of the CNN. Used to e.g. freeze layers
    :param include_angle:  models are trained for a rotation of the signal
    :param training_csv:  save training/testing results in CSV
    :param num_epochs: number of epochs used for training. One CNN epoch consists of 10,000 generated signals with Poisson noise
    :param initial_lr: Initial learning rate
    :param lr_deviation: how much is the learning rate decreased after X lr_epoch reps
    :param lr_epoch_reps: number of epochs until learning rate is changed
    :param them_cones: experiment that investigates signal on eye cones instead of sensor
    :param separate_rgb: separates RGB. Not used in paper
    :param meanData_rounding: round mean data signal template. Not used in paper
    :param shuffled_pixels: if False or 0 -> no shuffling. Otherwise:
                            if < 0: Shuffle along certain dimension:
                            if > 0: Shuffle all pixels (number hints block size)
    :param shuffle_scope: only shuffle (coherent) part of the image (centered)
    :param test_eval: evaluate performance on test set
    :param random_seed_nn: fixed random seed in pytoch part of CNN as well
    :param train_set_size: size of the training set
    :param test_size: size of the test set
    :param shuffle_portion: Shuffle certain percentage of (randomly selected) pixels
    :param ca_rule: Create cellular automaton here (not used)
    :param force_balance: Force exact balance of signal/non signal cases in test set
    :param same_test_data_shuff_pixels: ensure that test data have the same shuffle structure
    :param class_balance: can be 'signal_based' (all signal cases summed up are equal to all non signal cases) or
    # 'class_based' (all signal classes + non signal have equal sample size for train and test set). Relevant for signals
    based on multiple signal manifestations
    :param random_seed: Set seed for training
    :return:
    """

    # class_balance can be 'signal_based' (all signal cases summed up are equal to all non signal cases) or
    # 'class_based' (all signal classes + non signal have equal sample size for train and test set).
    if class_balance == 'class_based':
        signal_no_signal = False
    else:
        signal_no_signal = True

    shuffled_pixels_backup = 0
    startTime = time.time()
    print(device, pathMat)
    # set GPU to run CNN training on
    if device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device)
    test_interval = 1
    batchSize = 32
    numSamplesEpoch = 10000
    outPath = os.path.dirname(pathMat)
    fileName = os.path.basename(pathMat).split('.')[0]
    sys.stdout = Logger(f"{os.path.join(outPath, fileName)}_log.txt")
    # We want to add the same seeded poisson noise. We implement this by first getting the same meanData template
    # and add the seeded poisson noise. We then shuffle all test Data with the same mask.
    if same_test_data_shuff_pixels and (shuffled_pixels != 0):
        shuffled_pixels_backup = shuffled_pixels
        shuffled_pixels = False

    ###########################################
    # Exploration, not used in paper
    if include_shift:
        meanData, meanDataLabels, dataContrast, dataShift = get_h5mean_data(pathMat, includeContrast=True, includeShift=True,
                                                                            them_cones=them_cones, separate_rgb=separate_rgb,
                                                                            meanData_rounding=meanData_rounding, shuffled_pixels=shuffled_pixels,
                                                                            shuffle_scope=shuffle_scope, shuffle_portion=shuffle_portion,
                                                                            ca_rule=ca_rule)
    elif include_angle:
        meanData, meanDataLabels, dataContrast, dataAngle = get_h5mean_data(pathMat, includeContrast=True, includeAngle=True,
                                                                            them_cones=them_cones, separate_rgb=separate_rgb,
                                                                            meanData_rounding=meanData_rounding, shuffled_pixels=shuffled_pixels,
                                                                            shuffle_scope=shuffle_scope, shuffle_portion=shuffle_portion,
                                                                           ca_rule=ca_rule)
    ################################################
    # get signal data
    else:
        meanData, meanDataLabels, dataContrast = get_h5mean_data(pathMat, includeContrast=True,
                                                                 them_cones=them_cones, separate_rgb=separate_rgb,
                                                                 meanData_rounding=meanData_rounding, shuffled_pixels=shuffled_pixels,
                                                                 shuffle_scope=shuffle_scope, shuffle_portion=shuffle_portion,
                                                                 ca_rule=ca_rule)

    # CSV file housekeeping
    if training_csv:
        header = ['accuracy', 'dprime', 'epoch', 'contrast']
        default_vals = {}
        default_vals['contrast'] = max(dataContrast)
        if include_shift:
            header.append('shift')
            default_vals['shift'] = dataShift[1]
        if include_angle:
            header.append('angle')
            default_vals['angle'] = dataAngle[1]

        TrainWrt = CsvWriter(os.path.join(outPath, 'train_results.csv'), header=header, default_vals=default_vals, lock=lock)
        TestWrt = CsvWriter(os.path.join(outPath, 'test_results.csv'), header=header, default_vals=default_vals, lock=lock)
        train_test_log = [TrainWrt, TestWrt]
    else:
        train_test_log = None


    # Ensure that all data have the same shuffle pattern
    if same_test_data_shuff_pixels and shuffled_pixels_backup != 0:
        testDataFull, testLabelsFull = poisson_noise_loader(meanData, size=test_size, numpyData=True, seed=random_seed,
                                                            force_balance=force_balance, signal_no_signal=signal_no_signal)
        if shuffled_pixels_backup > 0:
            testDataFull = shuffle_pixels_func(testDataFull, shuffled_pixels_backup, shuffle_scope, shuffle_portion)
            meanData = shuffle_pixels_func(meanData, shuffled_pixels_backup, shuffle_scope, shuffle_portion)
            shuffled_pixels = shuffled_pixels_backup
        else:
            testDataFull = shuffle_1d(testDataFull, dimension=shuffled_pixels_backup)
            meanData = shuffle_1d(meanData, dimension=shuffled_pixels_backup)
            shuffled_pixels = shuffled_pixels_backup
        # also shuffle mean data. As the shuffle mask is seeded, we simply call the shuffle function again..
    else:
        testDataFull, testLabelsFull = poisson_noise_loader(meanData, size=test_size, numpyData=True, seed=random_seed,
                                                            force_balance=force_balance, signal_no_signal=signal_no_signal)

    # normalize values
    mean_norm = meanData.mean()
    std_norm = testDataFull.std()
    min_norm = testDataFull.min()
    max_norm = testDataFull.max()
    id_name = os.path.basename(pathMat).split('.')[0]

    # calculate accuracy of ideal observer on test set
    accOptimal, optimalOPredictionLabel = get_optimal_observer_acc_parallel(testDataFull, testLabelsFull, meanData,
                                                                            returnPredictionLabel=True)
    # save all predictions of ideal observer for further analysis
    pickle.dump(optimalOPredictionLabel, open(os.path.join(outPath, f"{id_name}_oo_pred_label.p"), 'wb'))
    pickle.dump(dataContrast, open(os.path.join(outPath, f"{id_name}_contrast_labels.p"), 'wb'))

    # calculate ideal observer d' or create dummy variables

    ########################################################################
    # Calculation of ideal observer d' is done here:
    if oo:
        if len(meanData) > 2:
            # set all signal cases to 1
            optimalOPredictionLabel[optimalOPredictionLabel > 0] = 1
            accOptimal = np.mean(optimalOPredictionLabel[:, 0] == optimalOPredictionLabel[:, 1])
            d1 = -1
            print(f"Theoretical d index is {d1}")
            d2 = calculate_dprime(optimalOPredictionLabel)
            print(f"Optimal observer d index is {d2}, acc is {accOptimal}.")

        else:
            # Calculate theoretical d' as a sanity test
            d1 = calculate_discriminability_index(meanData)
            print(f"Theoretical d index is {d1}")
            d2 = calculate_dprime(optimalOPredictionLabel)
            print(f"Optimal observer d index is {d2}")
        print(f"Optimal observer accuracy on all data is {accOptimal*100:.2f}%")
    ################################################################

    else:
        d1 = -1
        d2 = -1
        accOptimal = -1

    testData = testDataFull[:500]
    testLabels = testLabelsFull[:500]
    dimIn = testData[0].shape[1]
    dimOut = len(meanData)

    # train & test SVM
    if svm:
        include_contrast_svm = not (include_shift or include_angle)
        if include_contrast_svm:
            metric_svm = 'contrast'
        elif include_angle:
            metric_svm = 'angle'
        elif include_shift:
            metric_svm = 'shift'
        if train_set_size == -1:
            num_svm_samples = 10000
        else:
            num_svm_samples = train_set_size

        #############################################################
        # SVM training & testing is done here
        svm_process = mp.Process(target=score_svm, args=[pathMat, lock, testDataFull, testLabelsFull],
                                 kwargs={'them_cones': them_cones, 'includeContrast': include_contrast_svm, 'separate_rgb': separate_rgb, 'metric': metric_svm,
                                         'meanData_rounding': meanData_rounding, 'shuffled_pixels': shuffled_pixels, 'includeAngle': include_angle,
                                         'includeShift': include_shift, 'signal_no_signal': signal_no_signal, 'random_seed': random_seed, 'num_samples': num_svm_samples})
        svm_process.start()
        ##############################################################

    if train_nn:
        # set seed for CNN
        if random_seed_nn:
            torch.random.manual_seed(random_seed)
        if NetClass is None:
            if increase_network_depth:
                Net = GrayResnet101(dimOut)
            else:
                # CNN generally used for analysis
                Net = GrayResnet18(dimOut)
        else:
            if NetClass_param is None:
                Net = NetClass(dimOut, min_norm, max_norm, mean_norm, std_norm)
            else:
                Net = NetClass(dimOut, min_norm, max_norm, mean_norm, std_norm, freeze_until=NetClass_param)
        Net.cuda()
        print(Net)
        # Net.load_state_dict(torch.load('trained_RobustNet_denoised.torch'))
        # use negative log likelihood loss
        criterion = nn.NLLLoss()
        bestTestAcc = 0


        # Train the network
        # set parameters
        lr_deviation = lr_deviation
        num_epochs = num_epochs
        learning_rate = initial_lr
        testLabels = torch.from_numpy(testLabels.astype(np.long))
        testData = torch.from_numpy(testData).type(torch.float32)
        testData -= mean_norm
        testData /= std_norm

        # Object that adds Poisson noise to mean signal templates
        PoissonDataObject = PoissonNoiseLoaderClass(meanData, batchSize, train_set_size=train_set_size, data_seed=random_seed,
                                                    use_data_seed=True, signal_no_signal=signal_no_signal)

        ###################################################
        # CNN training/testing is done here
        for i in range(lr_epoch_reps):
            print(f"Trainig for {num_epochs/lr_epoch_reps} epochs with a learning rate of {learning_rate}..")
            optimizer = optim.Adam(Net.parameters(), lr=learning_rate)
            # import pdb; pdb.set_trace()
            Net, testAcc = train_poisson(round(num_epochs/lr_epoch_reps), numSamplesEpoch, batchSize, meanData, testData,
                                         testLabels, Net, test_interval, optimizer, criterion, dimIn, mean_norm, std_norm,
                                         train_test_log, test_eval, PoissonDataObject)
            print(f"Test accuracy is {testAcc*100:.2f} percent")
            learning_rate = learning_rate*lr_deviation

        ######################################################


        testLabelsFull = torch.from_numpy(testLabelsFull.astype(np.long))
        testDataFull = torch.from_numpy(testDataFull).type(torch.float32)
        testDataFull -= mean_norm
        testDataFull /= std_norm
        testAcc, nnPredictionLabels = test(batchSize, testDataFull, testLabelsFull, Net, dimIn, includePredictionLabels=True, test_eval=test_eval)
        if len(meanData) == 2 or optimalOPredictionLabel.max() <= 1:
            nnPredictionLabels_dprime = np.copy(nnPredictionLabels)
            nnPredictionLabels_dprime[nnPredictionLabels_dprime > 0] = 1
            nn_dprime = calculate_dprime(nnPredictionLabels_dprime)
        else:
            nn_dprime = -1
        pickle.dump(nnPredictionLabels, open(os.path.join(outPath, f"{id_name}_nn_pred_labels.p"), 'wb'))
    else:
        testAcc = 0.5
        nn_dprime = -1


    # print results to console
    print(f"ResNet accuracy is {testAcc*100:.2f}%")
    print(f"ResNet dprime is {nn_dprime}")
    print(f"Optimal observer accuracy is {accOptimal*100:.2f}%")
    print(f"Optimal observer d index is {d2}")
    print(f"Theoretical d index is {d1}")






    # Save results in a CSV files
    #########################################
    if train_nn or oo:
        if lock is not None:
            lock.acquire()
        resultCSV = os.path.join(outPath, "results.csv")
        file_exists = os.path.isfile(resultCSV)

        with open(resultCSV, 'a') as csvfile:
            if not include_shift and not include_angle:
                headers = ['ResNet_accuracy', 'optimal_observer_accuracy', 'theoretical_d_index', 'optimal_observer_d_index', 'contrast', 'nn_dprime']
                writer = csv.DictWriter(csvfile, delimiter=';', lineterminator='\n',fieldnames=headers)

                if not file_exists:
                    writer.writeheader()  # file doesn't exist yet, write a header

                writer.writerow({'ResNet_accuracy': testAcc, 'optimal_observer_accuracy': accOptimal, 'theoretical_d_index': d1, 'optimal_observer_d_index': d2, 'contrast': max(dataContrast).astype(np.float64), 'nn_dprime': nn_dprime})
            elif include_shift:
                headers = ['ResNet_accuracy', 'optimal_observer_accuracy', 'theoretical_d_index', 'optimal_observer_d_index', 'contrast', 'shift', 'nn_dprime']
                writer = csv.DictWriter(csvfile, delimiter=';', lineterminator='\n',fieldnames=headers)

                if not file_exists:
                    writer.writeheader()  # file doesn't exist yet, write a header

                writer.writerow({'ResNet_accuracy': testAcc, 'optimal_observer_accuracy': accOptimal, 'theoretical_d_index': d1, 'optimal_observer_d_index': d2, 'contrast': max(dataContrast).astype(np.float32), 'shift': dataShift[1].astype(np.float64), 'nn_dprime': nn_dprime})
            elif include_angle:
                headers = ['ResNet_accuracy', 'optimal_observer_accuracy', 'theoretical_d_index', 'optimal_observer_d_index', 'contrast', 'angle', 'nn_dprime']
                writer = csv.DictWriter(csvfile, delimiter=';', lineterminator='\n',fieldnames=headers)

                if not file_exists:
                    writer.writeheader()  # file doesn't exist yet, write a header

                writer.writerow({'ResNet_accuracy': testAcc, 'optimal_observer_accuracy': accOptimal, 'theoretical_d_index': d1, 'optimal_observer_d_index': d2, 'contrast': max(dataContrast).astype(np.float32), 'angle': dataAngle[1].astype(np.float64), 'nn_dprime': nn_dprime})

        print(f'Wrote results to {resultCSV}')
        if lock is not None:
            lock.release()

    endTime = time.time()
    print(f"done! It took {str(datetime.timedelta(seconds=endTime-startTime))} hours:min:seconds")
    sys.stdout = sys.stdout.revert()



