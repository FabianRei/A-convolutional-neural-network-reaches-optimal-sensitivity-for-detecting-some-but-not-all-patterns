from deepLearning.src.models.optimal_observer import get_optimal_observer_acc, calculate_dprime
import numpy as np
import pickle

fpath_nn = r'C:\Users\Fabian\Documents\data\rsync\redo_experiments\multiple_locations\multiple_locations_experiment_equal_class_samples\harmonic_frequency_of_1_loc_1_signalGridSize_4\1_samplesPerClass_freq_1_contrast_0_050357016472_loc_1_signalGrid_4_nn_pred_labels.p'
fpath_oo = r'C:\Users\Fabian\Documents\data\rsync\redo_experiments\multiple_locations\multiple_locations_experiment_equal_class_samples\harmonic_frequency_of_1_loc_1_signalGridSize_4\1_samplesPerClass_freq_1_contrast_0_050357016472_loc_1_signalGrid_4_oo_pred_label.p'

fpath_nn = r'C:\Users\Fabian\Documents\data\rsync\redo_experiments\multiple_locations\multiple_locations_experiment_equal_class_samples\harmonic_frequency_of_1_loc_1_signalGridSize_4\1_samplesPerClass_freq_1_contrast_0_000798104926_loc_1_signalGrid_4_nn_pred_labels.p'
fpath_oo = r'C:\Users\Fabian\Documents\data\rsync\redo_experiments\multiple_locations\multiple_locations_experiment_equal_class_samples\harmonic_frequency_of_1_loc_1_signalGridSize_4\1_samplesPerClass_freq_1_contrast_0_000798104926_loc_1_signalGrid_4_oo_pred_label.p'

# fpath_nn = r'C:\Users\Fabian\Documents\data\rsync\redo_experiments\multiple_locations\multiple_locations_experiment_equal_class_samples\harmonic_frequency_of_1_loc_1_signalGridSize_4\1_samplesPerClass_freq_1_contrast_0_100475457260_loc_1_signalGrid_4_nn_pred_labels.p'
# fpath_oo = r'C:\Users\Fabian\Documents\data\rsync\redo_experiments\multiple_locations\multiple_locations_experiment_equal_class_samples\harmonic_frequency_of_1_loc_1_signalGridSize_4\1_samplesPerClass_freq_1_contrast_0_100475457260_loc_1_signalGrid_4_oo_pred_label.p'

fpath_nn = r'\\?\C:\Users\Fabian\Documents\data\rsync\redo_experiments\multiple_locations\multiple_locations_experiment_equal_class_dprime_adjusted\harmonic_frequency_of_1_loc_1_signalGridSize_5\1_samplesPerClass_freq_1_contrast_0_400000000000_loc_1_signalGrid_5_nn_pred_labels.p'
fpath_oo = r'\\?\C:\Users\Fabian\Documents\data\rsync\redo_experiments\multiple_locations\multiple_locations_experiment_equal_class_dprime_adjusted\harmonic_frequency_of_1_loc_1_signalGridSize_5\1_samplesPerClass_freq_1_contrast_0_400000000000_loc_1_signalGrid_5_oo_pred_label.p'

with open(fpath_oo, 'rb') as f:
    oo = pickle.load(f)
with open(fpath_nn, 'rb') as f:
    nn = pickle.load(f)

n1 = nn[:,1]
n0 = nn[:,0]
o0 = oo[:,0]
o1 = oo[:,1]

nn1 = (n1>0).astype(np.int)
nn0 = (n0>0).astype(np.int)
oo0 = (o0>0).astype(np.int)
oo1 = (o1>0).astype(np.int)

nnn = (nn>0).astype(np.int)
ooo = (oo>0).astype(np.int)

calculate_dprime(ooo)
calculate_dprime(nnn)
