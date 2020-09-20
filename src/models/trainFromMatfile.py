if __name__ == '__main__':
    import os, sys, inspect

    current_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))))
    parent_dir = os.path.dirname(current_dir)
    sys.path.insert(0, parent_dir)

from deepLearning.src.models.resnet_train_test import train_poisson, test
from deepLearning.src.models.GrayResNet import GrayResnet18, GrayResnet101
from deepLearning.src.models.optimal_observer import get_optimal_observer_acc, calculate_discriminability_index, get_optimal_observer_hit_false_alarm, get_optimal_observer_acc_parallel, calculate_dprime
from deepLearning.src.data.mat_data import get_h5mean_data, poisson_noise_loader, PoissonNoiseLoaderClass, shuffle_pixels as shuffle_pixels_func, shuffle_1d
from deepLearning.src.data.logger import Logger, CsvWriter
from deepLearning.src.models.support_vector_machine import score_svm
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
from deepLearning.src.models.new_vgg import vgg16, vgg16bn



def autoTrain_Resnet_optimalObserver(pathMat, device=None, lock=None, train_nn=True, include_shift=False,
                                     deeper_pls=False, oo=True, svm=True, NetClass=None, NetClass_param=None,
                                     include_angle=False, training_csv=True, num_epochs=30, initial_lr=0.001, lr_deviation=0.1,
                                     lr_epoch_reps=3, them_cones=False, separate_rgb=False, meanData_rounding=None,
                                     shuffled_pixels=0, shuffle_scope=-1, test_eval=True, random_seed_nn=True, train_set_size=-1,
                                     test_size=5000, shuffle_portion=-1, ca_rule=-1, force_balance=False,
                                     same_test_data_shuff_pixels=True, class_balance='class_based', random_seed=42):


    # relevant variables
    # class_balance can be 'signal_based' (all signal cases summed up are equal to all non signal cases) or
    # 'class_based' (all signal classes + non signal have equal sample size for train and test set).
    if class_balance == 'class_based':
        signal_no_signal = False
    else:
        signal_no_signal = True

    shuffled_pixels_backup = 0
    startTime = time.time()
    print(device, pathMat)
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
    else:
        meanData, meanDataLabels, dataContrast = get_h5mean_data(pathMat, includeContrast=True,
                                                                 them_cones=them_cones, separate_rgb=separate_rgb,
                                                                 meanData_rounding=meanData_rounding, shuffled_pixels=shuffled_pixels,
                                                                 shuffle_scope=shuffle_scope, shuffle_portion=shuffle_portion,
                                                                 ca_rule=ca_rule)
    # data =    torch.from_numpy(data).type(torch.float32)
    # pickle.dump([data, labels, dataNoNoise], open('mat1PercentNoNoiseData.p', 'wb'))
    # data, labels, dataNoNoise = pickle.load(open("mat1PercentData.p", 'rb'))
    # Image.fromarray(data[4]*(255/20)).show()
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

    # normalization values
    mean_norm = meanData.mean()
    std_norm = testDataFull.std()
    min_norm = testDataFull.min()
    max_norm = testDataFull.max()
    id_name = os.path.basename(pathMat).split('.')[0]

    accOptimal, optimalOPredictionLabel = get_optimal_observer_acc_parallel(testDataFull, testLabelsFull, meanData,
                                                                            returnPredictionLabel=True)
    pickle.dump(optimalOPredictionLabel, open(os.path.join(outPath, f"{id_name}_oo_pred_label.p"), 'wb'))
    pickle.dump(dataContrast, open(os.path.join(outPath, f"{id_name}_contrast_labels.p"), 'wb'))

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
            d1 = calculate_discriminability_index(meanData)
            print(f"Theoretical d index is {d1}")
            d2 = calculate_dprime(optimalOPredictionLabel)
            print(f"Optimal observer d index is {d2}")
        print(f"Optimal observer accuracy on all data is {accOptimal*100:.2f}%")


    else:
        d1 = -1
        d2 = -1
        accOptimal = -1

    testData = testDataFull[:500]
    testLabels = testLabelsFull[:500]
    dimIn = testData[0].shape[1]
    dimOut = len(meanData)

    if svm:
        include_contrast_svm = not (include_shift or include_angle)
        if include_contrast_svm:
            metric_svm = 'contrast'
        elif include_angle:
            metric_svm = 'angle'
        elif include_shift:
            metric_svm = 'shift'

        # do_debug = False
        # if do_debug:
        #     kwords = {'them_cones': them_cones, 'includeContrast': include_contrast_svm, 'separate_rgb': separate_rgb, 'metric': metric_svm,
        #                                  'meanData_rounding': meanData_rounding, 'shuffled_pixels': shuffled_pixels, 'includeAngle': include_angle,
        #                                  'includeShift': include_shift}
        #     score_svm(pathMat, lock, **kwords)
        svm_process = mp.Process(target=score_svm, args=[pathMat, lock, testDataFull, testLabelsFull],
                                 kwargs={'them_cones': them_cones, 'includeContrast': include_contrast_svm, 'separate_rgb': separate_rgb, 'metric': metric_svm,
                                         'meanData_rounding': meanData_rounding, 'shuffled_pixels': shuffled_pixels, 'includeAngle': include_angle,
                                         'includeShift': include_shift, 'signal_no_signal': signal_no_signal, 'random_seed': random_seed})
        svm_process.start()

    if train_nn:
        if random_seed_nn:
            torch.random.manual_seed(random_seed)
        if NetClass is None:
            if deeper_pls:
                Net = GrayResnet101(dimOut)
            else:
                Net = GrayResnet18(dimOut)
        else:
            if NetClass_param is None:
                Net = NetClass(dimOut, min_norm, max_norm, mean_norm, std_norm)
            else:
                Net = NetClass(dimOut, min_norm, max_norm, mean_norm, std_norm, freeze_until=NetClass_param)
        Net.cuda()
        print(Net)
        # Net.load_state_dict(torch.load('trained_RobustNet_denoised.torch'))
        criterion = nn.NLLLoss()
        bestTestAcc = 0

        # Test the network
        # testAcc = test(batchSize, testData, testLabels, Net, dimIn)
        # Train the network
        lr_deviation = lr_deviation
        num_epochs = num_epochs
        learning_rate = initial_lr
        testLabels = torch.from_numpy(testLabels.astype(np.long))
        testData = torch.from_numpy(testData).type(torch.float32)
        testData -= mean_norm
        testData /= std_norm
        PoissonDataObject = PoissonNoiseLoaderClass(meanData, batchSize, train_set_size=train_set_size, data_seed=random_seed,
                                                    use_data_seed=True, signal_no_signal=signal_no_signal)
        for i in range(lr_epoch_reps):
            print(f"Trainig for {num_epochs/lr_epoch_reps} epochs with a learning rate of {learning_rate}..")
            optimizer = optim.Adam(Net.parameters(), lr=learning_rate)
            # import pdb; pdb.set_trace()
            Net, testAcc = train_poisson(round(num_epochs/lr_epoch_reps), numSamplesEpoch, batchSize, meanData, testData,
                                         testLabels, Net, test_interval, optimizer, criterion, dimIn, mean_norm, std_norm,
                                         train_test_log, test_eval, PoissonDataObject)
            print(f"Test accuracy is {testAcc*100:.2f} percent")
            learning_rate = learning_rate*lr_deviation

        # bestTestAcc = max(bestTestAcc, bestTestAccStep)
        # torch.save(Net.state_dict(), os.path.join(outPath, f"resNet_weights_{fileName}.torch"))
        # print("saved resNet weights to", f"resNet_weights_{fileName}.torch")
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


    print(f"ResNet accuracy is {testAcc*100:.2f}%")
    print(f"ResNet dprime is {nn_dprime}")
    print(f"Optimal observer accuracy is {accOptimal*100:.2f}%")
    print(f"Optimal observer d index is {d2}")
    print(f"Theoretical d index is {d1}")

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


if __name__ == '__main__':
    # mat_path = r'C:\Users\Fabian\Documents\data\svm_test\1_samplesPerClass_freq_1_contrast_oo_0_019952623150.h5'
    # mat_path = r'C:\Users\Fabian\Documents\data\rsync\redo_experiments\mtf_experiments\mtf_contrast_new_freq\harmonic_frequency_of_14\1_samplesPerClass_freq_14_contrast_0_019952623150.h5'
    # mat_path = r'C:\Users\Fabian\Documents\data\faces\multi_face_result\2_samplesPerClass_freq_1_contrast_0_019952623150_image_multi_face_result.h5'
    # mat_path = r'C:\Users\Fabian\Documents\data\faces\face_sq\2_samplesPerClass_freq_1_contrast_0_019952623150_image_face_sq.h5'
    # mat_path = r'C:\Users\Fabian\Documents\data\windows2rsync\windows_data\test_200_dummy2.h5'
    # mat_path = r'C:\Users\Fabian\Documents\data\windows2rsync\windows_data\mtf_shift\harmonic_frequency_of_124\2_samplesPerClass_freq_124_contrast_0_10_shift_0_100000000_pi.h5'
    # mat_path = r'C:\Users\Fabian\Documents\data\faces\face_guy_green\2_samplesPerClass_freq_1_contrast_0_019952623150_image_face_guy_green.h5'
    # mat_path = r'C:\Users\Fabian\Documents\data\windows2rsync\windows_data\mtf_angle_new_freq\harmonic_frequency_of_4/2_samplesPerClass_freq_4_contrast_0_10_angle_0_005336699_pi_oo.h5'
    # mat_path = r'C:\Users\Fabian\Documents\data\windows2rsync\windows_data\mtf_lines_angle_new_freq\harmonic_frequency_of_1/1_samplesPerClass_freq_1_contrast_0_10_angle_0_005336699_pi_oo.h5'
    # mat_path = r'C:\Users\Fabian\Documents\data\rsync\redo_experiments\lines_mtf_experiments\mtf_lines_shift_new_freq_lower_values\harmonic_frequency_of_1\1_samplesPerClass_freq_100_contrast_0_10_shift_0_000000001_pi.h5'
    # mat_path = r'C:\Users\Fabian\Documents\data\windows2rsync\windows_data\viz_templates\harmonic_frequency_of_1\1_samplesPerClass_freq_52_contrast_0_10_angle_0_125000000_pi_oo.h5'
    # mat_path = r'C:\Users\Fabian\Documents\data\windows2rsync\windows_data\disks\circle_with_radius_100\2_samplesPerClass_freq_1_contrast_0_010000000000_image_circle_with_radius_100.h5'
    # mat_path = r'C:\Users\Fabian\Documents\data\rsync\redo_experiments\face_experiment\multi_face_result\2_samplesPerClass_freq_1_contrast_0_019952623150_image_multi_face_result.h5'
    # mat_path = r'C:\Users\Fabian\Documents\data\windows2rsync\windows_data\redo_automaton\plain_automata\plain_automata_rule_3_class2\automata_rule_3_class2_contrast_0.01995262.h5'
    # mat_path = r'C:\Users\Fabian\Documents\data\windows2rsync\windows_data\disks\disk_templates\circle_with_radius_100.h5'
    # mat_path = r'C:\Users\Fabian\Documents\data\windows2rsync\windows_data\multiple_locations_templates\harmonic_frequency_of_1_loc_1_signalGridSize_4\1_samplesPerClass_freq_1_contrast_0_798104925988_loc_1_signalGrid_4.h5'
    # mat_path = r'C:\Users\Fabian\Documents\data\rsync\redo_experiments\multiple_locations\multiple_locations_experiment\harmonic_frequency_of_1_loc_1_signalGridSize_4\1_samplesPerClass_freq_1_contrast_0_012649110641_loc_1_signalGrid_4.h5'
    # mat_path = r'C:\Users\Fabian\Documents\data\windows2rsync\windows_data\multiple_locations_templates\harmonic_frequency_of_1_loc_4_signalGridSize_3\1_samplesPerClass_freq_1_contrast_0_798104925988_loc_4_signalGrid_3.h5'
    # mat_path = r'C:\Users\Fabian\Documents\data\windows2rsync\windows_data\multiple_locations_templates\harmonic_frequency_of_1_loc_1_signalGridSize_3\1_samplesPerClass_freq_1_contrast_0_798104925988_loc_1_signalGrid_3.h5'
    # mat_path = r'C:\Users\Fabian\Documents\data\windows2rsync\windows_data\multiple_locations_templates\harmonic_frequency_of_1_loc_1_signalGridSize_2\1_samplesPerClass_freq_1_contrast_0_798104925988_loc_1_signalGrid_2.h5'
    # mat_path = r'C:\Users\Fabian\Documents\data\windows2rsync\windows_data\multiple_locations_templates\harmonic_frequency_of_1_loc_1_signalGridSize_1\1_samplesPerClass_freq_1_contrast_0_798104925988_loc_1_signalGrid_1.h5'
    # mat_path = r'C:\Users\Fabian\Documents\data\rsync\redo_experiments\more_nn_backup\vgg\class_2_rule_3_5\2_samplesPerClass_freq_1_contrast_0_002511886432_image_automata_rule_3_class2.h5'
    mat_path = '/share/wandell/data/reith/redo_experiments/more_nn/vgg/class_2_rule_3_5/2_samplesPerClass_freq_1_contrast_0_001258925412_image_automata_rule_3_class2.h5'
    mat_path = r'C:\Users\Fabian\Documents\data\rsync\redo_experiments\redo_automaton\matlab_contrasts\automata_rule_22_class3\2_samplesPerClass_freq_1_contrast_0_019952623150_image_automata_rule_22_class3.h5'
    autoTrain_Resnet_optimalObserver(mat_path, shuffled_pixels=False, test_size=400,  train_nn=True, oo=False, svm=False, NetClass=vgg16)
    # autoTrain_Resnet_optimalObserver(mat_path, force_balance=True, shuffled_pixels=-2)
    # autoTrain_Resnet_optimalObserver(mat_path, shuffled_pixels=-2)
    # autoTrain_Resnet_optimalObserver(mat_path, shuffled_pixels=True, shuffle_scope=100, train_set_size=150, oo=False, svm=False, test_size=60, train_nn=True, shuffle_portion=2000)

