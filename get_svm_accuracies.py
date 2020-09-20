import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from deepLearning.src.models.support_vector_machine import write_svm_csv, get_svm_accuracy
import os
import multiprocessing as mp
from glob import glob
import time
import datetime
import numpy as np

def h5gen(folder):
    h5_files = glob(f"{folder}*.h5")
    for f in h5_files:
        yield f


def score_svm(h5_path, lock, metric='contrast', num_samples=15000, **kwargs):
    acc, dprime, metric_val = get_svm_accuracy(h5_path, num_samples, lock=lock, **kwargs)
    write_svm_csv(acc, dprime, metric_val, os.path.dirname(h5_path), lock=lock, metric_name=metric, num_samples=num_samples)


def run_svm_on_h5(folder, num_cpus, metric, lock=None, **kwargs):
    function_start = time.time()
    if lock is None:
        lock = mp.Lock()
    h5 = h5gen(folder)
    cpus = list(range(num_cpus))
    procs = {}
    while True:
        try:
            if procs == {}:
                for cpu in cpus:
                    h5_file = next(h5)
                    print(f"Svm scoring {h5_file}")
                    curr_p = mp.Process(target=score_svm, args=[h5_file, lock, metric], kwargs=kwargs)
                    procs[cpu] = curr_p
                    curr_p.start()
            for cpu, proc in procs.items():
                if not proc.is_alive():
                    h5_file = next(h5)
                    print(f"Svm scoring {h5_file}")
                    curr_p = mp.Process(target=score_svm, args=[h5_file, lock, metric], kwargs=kwargs)
                    procs[cpu] = curr_p
                    curr_p.start()
        except StopIteration:
            break

        time.sleep(5)

    for proc in procs.values():
        proc.join()

    function_end = time.time()
    with open(os.path.join(folder, 'time_svm.txt'), 'w') as txt:
        txt.write(f"Whole program finished! It took {str(datetime.timedelta(seconds=function_end-function_start))} hours:min:seconds")
    print(f"Whole program finished! It took {str(datetime.timedelta(seconds=function_end-function_start))} hours:min:seconds")
    time.sleep(60)
    print("done!")


def subfolder_gen(super_folder):
    sub_folders = [f.path + '/' for f in os.scandir(super_folder) if f.is_dir()]
    for f in sub_folders:
        yield f

def num_iterations_gen(arr):
    for a in arr:
        yield a

if __name__ == '__main__':
    sub_folder = r'C:\Users\Fabian\Documents\data\svm_test'
    sub_folder = '/share/wandell/data/reith/redo_experiments/sample_number_contrast/svm_1_sample/'
    sub_folder = '/share/wandell/data/reith/redo_experiments/sample_number_contrast/svm_1_sample_lowerer_contrast/'
    metric = 'contrast'
    kwargs = {'includeContrast': True}
    function_start = time.time()
    parallel_folders = list(range(1))
    num_cpus = 3
    processes = {}
    num_its = np.logspace(np.log10(1000), np.log10(60000), 8).astype(np.int)+5000
    mult_factor = 1.7948229213792886
    num_its = [60000*mult_factor+5000]
    num_its.append((num_its[0]-5000)*mult_factor+5000)
    num_its = np.array(num_its).astype(np.int)
    iter_gen = num_iterations_gen(num_its)
    while True:
        try:
            if processes == {}:
                for f in parallel_folders:
                    iterations = next(iter_gen)
                    print(f"scoring {iterations}")
                    curr_p = mp.Process(target=run_svm_on_h5, args=[sub_folder, num_cpus, metric], kwargs={'includeContrast': True, 'num_samples': iterations})
                    processes[f] = curr_p
                    curr_p.start()
            for f, proc in processes.items():
                if not proc.is_alive():
                    iterations = next(iter_gen)
                    print(f"scoring {iterations}")
                    curr_p = mp.Process(target=run_svm_on_h5, args=[sub_folder, num_cpus, metric], kwargs={'includeContrast': True, 'num_samples': iterations})
                    processes[f] = curr_p
                    curr_p.start()
        except StopIteration:
            break

        time.sleep(5)

    for proc in processes.values():
        proc.join()
    function_end = time.time()
    with open(os.path.join(sub_folder, 'time_svm.txt'), 'w') as txt:
        txt.write(f"Whole program finished! It took {str(datetime.timedelta(seconds=function_end-function_start))} hours:min:seconds")
    print(f"Whole program finished! It took {str(datetime.timedelta(seconds=function_end-function_start))} hours:min:seconds")
    print("done!")



"""
Older examples:
##########################################################################
if __name__ == '__main__':
    sub_folder = '/share/wandell/data/reith/svm_num_samples_freq1_contrast/'
    metric = 'contrast'
    kwargs = {'includeContrast': True}
    function_start = time.time()
    parallel_folders = list(range(1))
    num_cpus = 2
    processes = {}
    iter_gen = num_iterations_gen([60000])
    while True:
        try:
            if processes == {}:
                for f in parallel_folders:
                    iterations = next(iter_gen)
                    print(f"scoring {iterations}")
                    curr_p = mp.Process(target=run_svm_on_h5, args=[sub_folder, num_cpus, metric], kwargs={'includeContrast': True, 'num_samples': iterations})
                    processes[f] = curr_p
                    curr_p.start()
            for f, proc in processes.items():
                if not proc.is_alive():
                    iterations = next(iter_gen)
                    print(f"scoring {iterations}")
                    curr_p = mp.Process(target=run_svm_on_h5, args=[sub_folder, num_cpus, metric], kwargs={'includeContrast': True, 'num_samples': iterations})
                    processes[f] = curr_p
                    curr_p.start()
        except StopIteration:
            break

        time.sleep(5)

    for proc in processes.values():
        proc.join()
    function_end = time.time()
    with open(os.path.join(sub_folder, 'time_svm.txt'), 'w') as txt:
        txt.write(f"Whole program finished! It took {str(datetime.timedelta(seconds=function_end-function_start))} hours:min:seconds")
    print(f"Whole program finished! It took {str(datetime.timedelta(seconds=function_end-function_start))} hours:min:seconds")
    print("done!")

#####################################
if __name__ == '__main__':
    super_folder = '/share/wandell/data/reith/2_class_MTF_angle_experiment/'
    metric = 'angle'
    kwargs = {'includeAngle': True}
    function_start = time.time()
    folder_gen = subfolder_gen(super_folder)
    parallel_folders = list(range(2))
    num_cpus = 6
    processes = {}
    while True:
        try:
            if processes == {}:
                for f in parallel_folders:
                    sub_folder = next(folder_gen)
                    print(f"scoring {sub_folder}")
                    curr_p = mp.Process(target=run_svm_on_h5, args=[sub_folder, num_cpus, metric], kwargs=kwargs)
                    processes[f] = curr_p
                    curr_p.start()
            for f, proc in processes.items():
                if not proc.is_alive():
                    sub_folder = next(folder_gen)
                    print(f"scoring {sub_folder}")
                    curr_p = mp.Process(target=run_svm_on_h5, args=[sub_folder, num_cpus, metric], kwargs=kwargs)
                    processes[f] = curr_p
                    curr_p.start()
        except StopIteration:
            break

        time.sleep(5)

    for proc in processes.values():
        proc.join()

    function_end = time.time()
    with open(os.path.join(super_folder, 'time_svm.txt'), 'w') as txt:
        txt.write(f"Whole program finished! It took {str(datetime.timedelta(seconds=function_end-function_start))} hours:min:seconds")
    print(f"Whole program finished! It took {str(datetime.timedelta(seconds=function_end-function_start))} hours:min:seconds")
    print("done!")
##############################################

"""
