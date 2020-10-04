from src.models.trainFromMatfile import determine_signal_contrast_cnn_svm_io_performance
from glob import glob
import time
import datetime

path_signal_templates = '/share/wandell/data/reith/experiment_freq_1_log_contrasts20_frozen_pretrained_resnet/'

h5_files = glob(f'{path_signal_templates}*.h5')
# h5_files.sort()
program_start = time.time()
for matFile in h5_files:
    determine_signal_contrast_cnn_svm_io_performance(matFile, train_nn=False, include_shift=False)

program_end = time.time()

print(f"Whole program finished! It took {str(datetime.timedelta(seconds=program_end - program_start))} hours:min:seconds")
print("done!")
