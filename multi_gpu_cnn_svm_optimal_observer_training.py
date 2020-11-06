import os, sys,inspect
# change PYTHONPATH variable to include specific directories. Important when run from console
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from src.models.trainFromMatfile import determine_signal_contrast_cnn_svm_io_performance
from glob import glob
import GPUtil
import multiprocessing as mp
import time
import datetime
import os
import fnmatch

from src.models.new_inception import inceptionv3
from src.models.new_vgg import vgg16, vgg16bn
from src.models.new_alexnet import alexnet

from on_root_path import on_root_path


def matfile_gen(pathMatDir):
    """
    yields one path to a .h5 file containing the signal/non-signal template
    :param pathMatDir:
    :return:
    """
    matFiles = glob(f'{pathMatDir}**/*.h5', recursive=True)
    matFiles.sort()
    for matFile in matFiles:
        yield matFile



def run_across_contrast_levels(dirname, increase_network_depth=False, NetClass=None, NetClass_param=None, **kwargs):
    """
    Train CNN, ideal observer & SVM for all contrast levels found within the designated folder
    :param dirname: designated folder with signal contrast levels as .h5 files
    :param increase_network_depth: Increase depth of the network. Used for experiments that didn't make it into the paper
    :param NetClass: choose CNN used for training. If None, ResNEt50 is used
    :param NetClass_param: used to add specific parameter to CNN construction. E.g. used to freeze CNN up to some point.
    :param kwargs: other arguments, described in the docs of "determine_signal_contrast_cnn_svm_io_performance
    :return:
    """
    kword_args = {'train_nn': True, 'include_shift': False, 'NetClass': NetClass, 'increase_network_depth': increase_network_depth,
                  'NetClass_param': NetClass_param, 'include_angle': False, 'svm': True, 'force_balance': False}
    # Find all GPUs available on the server
    deviceIDs = GPUtil.getAvailable(order='first', limit=6, maxLoad=0.1, maxMemory=0.1, excludeID=[], excludeUUID=[])
    print(deviceIDs)
    # measure time passed
    function_start = time.time()
    # generate all pahts to the corresponding signal .h5 templates
    pathGen = matfile_gen(dirname)
    Procs = {}
    lock = mp.Lock()
    # run processes that run individual signals on a specified GPU. Run, until all signal .h5 templates/files are
    # processed
    while True:
        try:
            if Procs == {}:
                for device in deviceIDs:
                    pathMat = next(pathGen)
                    print(f"Running {pathMat} on GPU {device}")
                    currP = mp.Process(target=determine_signal_contrast_cnn_svm_io_performance, args=[pathMat],
                                       kwargs={'device': int(device), 'lock': lock, **kword_args, **kwargs})
                    Procs[str(device)] = currP
                    currP.start()
            for device, proc in Procs.items():
                if not proc.is_alive():
                    time.sleep(30)
                    pathMat = next(pathGen)
                    print(f"Running {pathMat} on GPU {device}")
                    currP = mp.Process(target=determine_signal_contrast_cnn_svm_io_performance, args=[pathMat],
                                       kwargs={'device': int(device), 'lock': lock, **kword_args, **kwargs})
                    Procs[str(device)] = currP
                    currP.start()
        except StopIteration:
            break

        time.sleep(30)

    # wait for all processes to be finished. Then - continue

    # this might be faster than proc.join() for all processes
    # (should exclude subprocesses (svm) which can continue running)
    one_proc_alive = True
    while one_proc_alive:
        alive_procs = []
        for proc in Procs.values():
            alive_procs.append(proc.is_alive())
        one_proc_alive = max(alive_procs)
        time.sleep(20)


    function_end = time.time()
    with open(os.path.join(dirname, 'time.txt'), 'w') as txt:
        txt.write(f"Whole program finished! It took {str(datetime.timedelta(seconds=function_end-function_start))} hours:min:seconds")
    print(f"Whole program finished! It took {str(datetime.timedelta(seconds=function_end-function_start))} hours:min:seconds")
    time.sleep(120)
    print("done!")


# Runs experiment. Each declared path contains .h5 signal arrays of varying contrast levels
# if __name__ == '__main__':
#     # run automata experiments for various seeds
#     # run a select group of experiments for various seeds.
#     full_start = time.time()
#     faces_path = os.path.join(on_root_path(), 'local', 'experiment', 'faces')
#     folder_paths = [p.path for p in os.scandir(faces_path) if p.is_dir()]
#     for folder_path in folder_paths:
#         print(folder_path)
#         fpaths = [p.path for p in os.scandir(folder_path) if p.is_dir()]
#         seed = int(folder_path.split('_')[-1])
#         for fpath in fpaths:
#             run_across_contrast_levels(fpath, random_seed=seed)
#     print(f"Whole program finished! It took {str(datetime.timedelta(seconds=time.time()-full_start))} hours:min:seconds")


if __name__ == '__main__':
    # run automata experiments for various seeds
    # run a select group of experiments for various seeds.
    full_start = time.time()
    super_path = r'/share/wandell/data/reith/redo_experiments/sd_more_nn'
    seed_paths = [p.path for p in os.scandir(super_path) if p.is_dir()]
    for sp in seed_paths:
        seed = int(sp.split('_')[-1])
        seed_folders = [p.path for p in os.scandir(sp) if p.is_dir()]
        for sf in seed_folders:
            print(sf)
            import pbd; pdb.set_trace()
            network = sf.split('_')[0]
            if network == 'vgg':
                NetClass=vgg16
            elif network == 'alexnet':
                NetClass=alexnet
            else:
                NetClass = None
            if sf.split('_')[-1] == '2':
                shuffle_pixels = 1
            else:
                shuffle_pixels = False
            if seed > 42 and not (sf.split('_')[-1] == '2'):
                continue
            run_across_contrast_levels(sf, random_seed=seed, NetClass=NetClass, shuffled_pixels=shuffle_pixels)

    print(f"Whole program finished! It took {str(datetime.timedelta(seconds=time.time()-full_start))} hours:min:seconds")




r"""
################################################################
ALL RUNS ARE AUTOMATICALLY SEEDED
################################################################
FIGURE 2
################################################################
if __name__ == '__main__':
    full_start = time.time()
    super_path = os.path.join(on_root_path(), 'local', 'experiments', 'harmonic_frequencies')
    fpaths = [p.path for p in os.scandir(super_path) if p.is_dir()]
    fpaths.sort(key=lambda x: int(x.split('_')[-1]))
    for fpath in fpaths:
        run_across_contrast_levels(fpath)
    print(f"Whole program finished! It took {str(datetime.timedelta(seconds=time.time()-full_start))} hours:min:seconds")

################################################################
FIGURE 3
################################################################
if __name__ == '__main__':
    # disk mtf calculation. size is in pixel
    full_start = time.time()
    super_path = os.path.join(on_root_path(), 'local', 'experiments', 'disks')    
    fpaths = [p.path for p in os.scandir(super_path) if p.is_dir()]
    fpaths.sort(key=lambda x: int(x.split('_')[-1]))
    for fpath in fpaths:
        run_across_contrast_levels(fpath, include_shift=True)
    print(f"Whole program finished! It took {str(datetime.timedelta(seconds=time.time()-full_start))} hours:min:seconds")

################################################################
FIGURE 4
################################################################
if __name__ == '__main__':
    # individual faces
    full_start = time.time()
    super_path = os.path.join(on_root_path(), 'local', 'experiments', 'faces') 
    fpaths = [p.path for p in os.scandir(super_path) if p.is_dir()]
    # fpaths.sort(key=lambda x: int(x.split('_')[-1]))
    for fpath in fpaths:
        run_across_contrast_levels(fpath)
    print(f"Whole program finished! It took {str(datetime.timedelta(seconds=time.time()-full_start))} hours:min:seconds")

################################################################
FIGURE 5
################################################################
if __name__ == '__main__':
    full_start = time.time()
    super_path = os.path.join(on_root_path(), 'local', 'experiments', 'automata') 
    fpaths = [p.path for p in os.scandir(super_path) if p.is_dir()]
    # fpaths.sort(key=lambda x: int(x.split('_')[-1]))
    for fpath in fpaths:
        run_across_contrast_levels(fpath)
    print(f"Whole program finished! It took {str(datetime.timedelta(seconds=time.time()-full_start))} hours:min:seconds")

################################################################
FIGURE 6
################################################################
if __name__ == '__main__':
    full_start = time.time()
        super_path = os.path.join(on_root_path(), 'local', 'experiments', 'shuffled_pixels') 
    fpaths = [p.path for p in os.scandir(super_path) if p.is_dir()]
    fpaths.sort(key=lambda x: int(x.split('x')[-1]))
    for fpath in fpaths:
        s_pixels = int(fpath.split('x')[-1])
        run_across_contrast_levels(fpath, shuffled_pixels=True)
    print(f"Whole program finished! It took {str(datetime.timedelta(seconds=time.time()-full_start))} hours:min:seconds")

################################################################
FIGURE 7
################################################################
if __name__ == '__main__':
    full_start = time.time()
    # run only on ideal observer, account for varying sample sizes in calculation
    fpaths = [p.path for p in os.scandir(os.path.join(on_root_path(), 'local', 'experiments', 'multiple_locations)) if p.is_dir()]
    fpaths.sort(key=lambda x: int(x.split('_')[-1]), reverse=False)
    for fpath in fpaths:
        run_across_contrast_levels(fpath)
    print(f"Whole program finished! It took {str(datetime.timedelta(seconds=time.time()-full_start))} hours:min:seconds")

################################################################
FIGURE 8
################################################################
if __name__ == '__main__':
    full_start = time.time()
    # run only on ideal observer, account for varying sample sizes in calculation
    fpaths = [p.path for p in os.scandir(os.path.join(on_root_path(), 'local', 'experiments', 'resnet')) if p.is_dir()]
    fpaths.sort(key=lambda x: int(x.split('_')[-1]), reverse=False)
    for fpath in fpaths:
        num = fpath.split('_')[-1]
        if num == '2':
            run_across_contrast_levels(fpath, shuffled_pixels=1)
        elif num == '3':
            run_across_contrast_levels(fpath, include_shift=True)
        else:
            run_across_contrast_levels(fpath)
    print(f"Whole program finished! It took {str(datetime.timedelta(seconds=time.time()-full_start))} hours:min:seconds")


if __name__ == '__main__':
    full_start = time.time()
    # run only on ideal observer, account for varying sample sizes in calculation
    fpaths = [p.path for p in os.scandir(os.path.join(on_root_path(), 'local', 'experiments', 'vgg')) if p.is_dir()]
    fpaths.sort(key=lambda x: int(x.split('_')[-1]), reverse=False)
    net_class = vgg16
    for fpath in fpaths:
        num = fpath.split('_')[-1]
        if num == '2':
            run_across_contrast_levels(fpath, shuffled_pixels=1, NetClass=net_class, initial_lr=0.00001)
        elif num == '3':
            run_across_contrast_levels(fpath, include_shift=True, NetClass=net_class, initial_lr=0.00001)
        else:
            run_across_contrast_levels(fpath, NetClass=net_class, initial_lr=0.00001)
    print(f"Whole program finished! It took {str(datetime.timedelta(seconds=time.time()-full_start))} hours:min:seconds")

if __name__ == '__main__':
    full_start = time.time()
    # run only on ideal observer, account for varying sample sizes in calculation
    fpaths = [p.path for p in os.scandir(os.path.join(on_root_path(), 'local', 'experiments', 'alexnet')) if p.is_dir()]
    fpaths.sort(key=lambda x: int(x.split('_')[-1]), reverse=False)
    net_class = alexnet
    for fpath in fpaths:
        num = fpath.split('_')[-1]
        if num == '2':
            run_across_contrast_levels(fpath, shuffled_pixels=1, NetClass=net_class, initial_lr=0.00001)
        elif num == '3':
            run_across_contrast_levels(fpath, include_shift=True, NetClass=net_class, initial_lr=0.00001)
        else:
            run_across_contrast_levels(fpath, NetClass=net_class, initial_lr=0.00001)
    print(f"Whole program finished! It took {str(datetime.timedelta(seconds=time.time()-full_start))} hours:min:seconds")

__________________________________________________________________________________________________________

################################################################
FIGURE A1
################################################################
if __name__ == '__main__':
    full_start = time.time()
    super_path = os.path.join(on_root_path(), 'local', 'experiments', 'sample_number_contrasts')
    fpaths = [p.path for p in os.scandir(super_path) if p.is_dir()]
    fpaths.sort(key=lambda k: int(k.split("_")[-1]))
    for fpath in fpaths:
        train_set_size = int(fpath.split('_')[-1])
        run_across_contrast_levels(fpath, train_set_size=train_set_size)
    print(f"Whole program finished! It took {str(datetime.timedelta(seconds=time.time()-full_start))} hours:min:seconds")

################################################################
FIGURE A2
################################################################
if __name__ == '__main__':
    full_start = time.time()
    fpaths = [p.path for p in os.scandir(os.path.join(on_root_path(), 'local', 'experiments', 'vgg')) if p.is_dir()]
    fpaths.sort(key=lambda x: int(x.split('_')[-1]), reverse=False)
    net_class = vgg16
    for fpath in fpaths:
        num = fpath.split('_')[-1]
        if num != '2':
            continue
        else:
            run_across_contrast_levels(fpath, shuffled_pixels=1, NetClass=net_class, initial_lr=0.00001)
    print(f"Whole program finished! It took {str(datetime.timedelta(seconds=time.time()-full_start))} hours:min:seconds")

if __name__ == '__main__':
    full_start = time.time()
    fpaths = [p.path for p in os.scandir(os.path.join(on_root_path(), 'local', 'experiments', 'alexnet')) if p.is_dir()]
    fpaths.sort(key=lambda x: int(x.split('_')[-1]), reverse=False)
    net_class = alexnet
    for fpath in fpaths:
        num = fpath.split('_')[-1]
        if num != '2':
            continue
        else:
            run_across_contrast_levels(fpath, shuffled_pixels=1, NetClass=net_class, initial_lr=0.00001)
    print(f"Whole program finished! It took {str(datetime.timedelta(seconds=time.time()-full_start))} hours:min:seconds")


################################################################
ADDITIONAL RUNS TO GET RESULTS WITH OTHER SEEDS. 
################################################################
if __name__ == '__main__':
    # run a select group of experiments for various seeds.
    full_start = time.time()
    folder_paths = [os.path.join(on_root_path(), 'local', 'experiments', 'seed42'),
                    os.path.join(on_root_path(), 'local', 'experiments', 'seed43'),
                    os.path.join(on_root_path(), 'local', 'experiments', 'seed44'),
                    os.path.join(on_root_path(), 'local', 'experiments', 'seed45'),
                    os.path.join(on_root_path(), 'local', 'experiments', 'seed46')]
    
    for folder_path in folder_paths:
        fpaths = [p.path for p in os.scandir(folder_path) if p.is_dir()]
        seed = int(folder_path.split('_')[-1])
        for fpath in fpaths:
            run_across_contrast_levels(fpath, random_seed=seed)
    print(f"Whole program finished! It took {str(datetime.timedelta(seconds=time.time()-full_start))} hours:min:seconds")
###############################################################

if __name__ == '__main__':
    # run automata experiments for various seeds
    # run a select group of experiments for various seeds.
    full_start = time.time()
    super_path = '/share/wandell/data/reith/redo_experiments/sd_automata'
    folder_paths = glob(os.path.join(super_path, 'seed*'))
    sorted(folder_paths)
    print(folder_paths)
    for folder_path in folder_paths:
        print(folder_path)
        fpaths = [p.path for p in os.scandir(folder_path) if p.is_dir()]
        seed = int(folder_path.split('_')[-1])
        for fpath in fpaths:
            run_across_contrast_levels(fpath, random_seed=seed)
    print(f"Whole program finished! It took {str(datetime.timedelta(seconds=time.time()-full_start))} hours:min:seconds")
################################################################
Multiple seeds for faces
######################################################
if __name__ == '__main__':
    # run automata experiments for various seeds
    # run a select group of experiments for various seeds.
    full_start = time.time()
    faces_path = '/share/wandell/data/reith/redo_experiments/sd_faces'
    # folder_paths = ['/share/wandell/data/reith/redo_experiments/multiloc_addition/sd_seed_42']
    folder_paths = [p.path for p in os.scandir(faces_path) if p.is_dir()]
    for folder_path in folder_paths:
        print(folder_path)
        fpaths = [p.path for p in os.scandir(folder_path) if p.is_dir()]
        seed = int(folder_path.split('_')[-1])
        for fpath in fpaths:
            run_across_contrast_levels(fpath, random_seed=seed)
    print(f"Whole program finished! It took {str(datetime.timedelta(seconds=time.time()-full_start))} hours:min:seconds")
#############################################################
"""
