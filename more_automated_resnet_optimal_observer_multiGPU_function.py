import os, sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from deepLearning.src.models.trainFromMatfile import autoTrain_Resnet_optimalObserver
from deepLearning.src.models.Resnet import PretrainedResnetFrozen, NotPretrainedResnet
from glob import glob
import GPUtil
import multiprocessing as mp
import time
import datetime
import os
import fnmatch

from deepLearning.src.models.new_inception import inceptionv3
from deepLearning.src.models.new_vgg import vgg16, vgg16bn
from deepLearning.src.models.new_alexnet import alexnet


def matfile_gen(pathMatDir):
    matFiles = glob(f'{pathMatDir}/*.h5', recursive=True)
    matFiles.sort()
    for matFile in matFiles:
        yield matFile


def run_on_folder(dirname, deeper_pls=False, NetClass=None, NetClass_param=None, **kwargs):
    kword_args = {'train_nn': True, 'include_shift': False, 'NetClass': NetClass, 'deeper_pls': deeper_pls,
                  'NetClass_param': NetClass_param, 'include_angle': False, 'svm': True, 'force_balance': False}
    deviceIDs = GPUtil.getAvailable(order='first', limit=6, maxLoad=0.1, maxMemory=0.1, excludeID=[], excludeUUID=[])
    # deviceIDs = [1,2,3,4,5,6]
    print(deviceIDs)
    function_start = time.time()
    pathGen = matfile_gen(dirname)
    Procs = {}
    lock = mp.Lock()
    while True:
        try:
            if Procs == {}:
                for device in deviceIDs:
                    pathMat = next(pathGen)
                    print(f"Running {pathMat} on GPU {device}")
                    currP = mp.Process(target=autoTrain_Resnet_optimalObserver, args=[pathMat],
                                       kwargs={'device': int(device), 'lock': lock, **kword_args, **kwargs})
                    Procs[str(device)] = currP
                    currP.start()
            for device, proc in Procs.items():
                if not proc.is_alive():
                    time.sleep(30)
                    pathMat = next(pathGen)
                    print(f"Running {pathMat} on GPU {device}")
                    currP = mp.Process(target=autoTrain_Resnet_optimalObserver, args=[pathMat],
                                       kwargs={'device': int(device), 'lock': lock, **kword_args, **kwargs})
                    Procs[str(device)] = currP
                    currP.start()
        except StopIteration:
            break

        time.sleep(30)

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


if __name__ == '__main__':
    # run a select group of experiments for various seeds.
    full_start = time.time()
    folder_paths = ['/share/wandell/data/reith/redo_experiments/sd_experiment/sd_seed_43',
                    '/share/wandell/data/reith/redo_experiments/sd_experiment/sd_seed_44',
                    '/share/wandell/data/reith/redo_experiments/sd_experiment/sd_seed_45',
                    '/share/wandell/data/reith/redo_experiments/sd_experiment/sd_seed_46']
    # rerun this first, as error in automaton..
    # folder_paths = ['/share/wandell/data/reith/redo_experiments/sd_experiment/sd_seed_42']
    for folder_path in folder_paths:
        fpaths = [p.path for p in os.scandir(folder_path) if p.is_dir()]
        seed = int(folder_path.split('_')[-1])
        for fpath in fpaths:
            # only run for multiloc (without the ones that already worked
            # if not 'multiloc_16' in fpath:
            #     continue
            if '1dshuff' in fpath:
                run_on_folder(fpath, shuffled_pixels=-2, random_seed=seed)
            elif '2dshuff' in fpath:
                run_on_folder(fpath, shuffled_pixels=1, random_seed=seed)
            elif 'multiloc' in fpath:
                run_on_folder(fpath, class_balance='signal_based', random_seed=seed)
            else:
                run_on_folder(fpath, random_seed=seed)
    print(f"Whole program finished! It took {str(datetime.timedelta(seconds=time.time()-full_start))} hours:min:seconds")





r"""
PAST RUNS
########################################################################################################
if __name__ == '__main__':
    full_start = time.time()
    # run only on ideal observer, account for varying sample sizes in calculation
    fpaths = [p.path for p in os.scandir('/share/wandell/data/reith/redo_experiments/more_nn/vgg') if p.is_dir()]
    fpaths.sort(key=lambda x: int(x.split('_')[-1]), reverse=False)
    net_class = vgg16
    for fpath in fpaths:
        num = fpath.split('_')[-1]
        if num != '2':
            continue
        else:
            run_on_folder(fpath, shuffled_pixels=1, NetClass=net_class, initial_lr=0.00001)
    print(f"Whole program finished! It took {str(datetime.timedelta(seconds=time.time()-full_start))} hours:min:seconds")

if __name__ == '__main__':
    full_start = time.time()
    # run only on ideal observer, account for varying sample sizes in calculation
    fpaths = [p.path for p in os.scandir('/share/wandell/data/reith/redo_experiments/more_nn/alexnet') if p.is_dir()]
    fpaths.sort(key=lambda x: int(x.split('_')[-1]), reverse=False)
    net_class = alexnet
    for fpath in fpaths:
        num = fpath.split('_')[-1]
        if num != '2':
            continue
        else:
            run_on_folder(fpath, shuffled_pixels=1, NetClass=net_class, initial_lr=0.00001)
    print(f"Whole program finished! It took {str(datetime.timedelta(seconds=time.time()-full_start))} hours:min:seconds")
#####################################################################################################
if __name__ == '__main__':
    full_start = time.time()
    # run only on ideal observer, account for varying sample sizes in calculation
    fpaths = [p.path for p in os.scandir('/share/wandell/data/reith/redo_experiments/more_nn/resnet') if p.is_dir()]
    fpaths.sort(key=lambda x: int(x.split('_')[-1]), reverse=False)
    for fpath in fpaths:
        num = fpath.split('_')[-1]
        if num == '2':
            run_on_folder(fpath, shuffled_pixels=1)
        elif num == '3':
            run_on_folder(fpath, include_shift=True)
        else:
            run_on_folder(fpath)
    print(f"Whole program finished! It took {str(datetime.timedelta(seconds=time.time()-full_start))} hours:min:seconds")


if __name__ == '__main__':
    full_start = time.time()
    # run only on ideal observer, account for varying sample sizes in calculation
    fpaths = [p.path for p in os.scandir('/share/wandell/data/reith/redo_experiments/more_nn/vgg') if p.is_dir()]
    fpaths.sort(key=lambda x: int(x.split('_')[-1]), reverse=False)
    net_class = vgg16
    for fpath in fpaths:
        num = fpath.split('_')[-1]
        if num == '2':
            run_on_folder(fpath, shuffled_pixels=1, NetClass=net_class, initial_lr=0.00001)
        elif num == '3':
            run_on_folder(fpath, include_shift=True, NetClass=net_class, initial_lr=0.00001)
        else:
            run_on_folder(fpath, NetClass=net_class, initial_lr=0.00001)
    print(f"Whole program finished! It took {str(datetime.timedelta(seconds=time.time()-full_start))} hours:min:seconds")

if __name__ == '__main__':
    full_start = time.time()
    # run only on ideal observer, account for varying sample sizes in calculation
    fpaths = [p.path for p in os.scandir('/share/wandell/data/reith/redo_experiments/more_nn/alexnet') if p.is_dir()]
    fpaths.sort(key=lambda x: int(x.split('_')[-1]), reverse=False)
    net_class = alexnet
    for fpath in fpaths:
        num = fpath.split('_')[-1]
        if num == '2':
            run_on_folder(fpath, shuffled_pixels=1, NetClass=net_class, initial_lr=0.00001)
        elif num == '3':
            run_on_folder(fpath, include_shift=True, NetClass=net_class, initial_lr=0.00001)
        else:
            run_on_folder(fpath, NetClass=net_class, initial_lr=0.00001)
    print(f"Whole program finished! It took {str(datetime.timedelta(seconds=time.time()-full_start))} hours:min:seconds")
################################################################
if __name__ == '__main__':
    full_start = time.time()
    # run only on ideal observer, account for varying sample sizes in calculation
    fpaths = [p.path for p in os.scandir('/scratch/reith/oo/more_nn_oo/vgg16') if p.is_dir()]
    fpaths.sort(key=lambda x: int(x.split('_')[-1]), reverse=False)
    net_class = vgg16
    for fpath in fpaths:
        num = fpath.split('_')[-1]
        if num == '2':
            run_on_folder(fpath, shuffled_pixels=1, NetClass=net_class, initial_lr=0.00001, train_nn=False, svm=False)
        elif num == '3':
            run_on_folder(fpath, include_shift=True, NetClass=net_class, initial_lr=0.00001, train_nn=False, svm=False)
        else:
            run_on_folder(fpath, NetClass=net_class, initial_lr=0.00001, train_nn=False, svm=False)
    print(f"Whole program finished! It took {str(datetime.timedelta(seconds=time.time()-full_start))} hours:min:seconds")

if __name__ == '__main__':
    full_start = time.time()
    # run only on ideal observer, account for varying sample sizes in calculation
    fpaths = [p.path for p in os.scandir('/scratch/reith/oo/more_nn_oo/alexnet') if p.is_dir()]
    fpaths.sort(key=lambda x: int(x.split('_')[-1]), reverse=False)
    net_class = alexnet
    for fpath in fpaths:
        num = fpath.split('_')[-1]
        if num == '2':
            run_on_folder(fpath, shuffled_pixels=1, NetClass=net_class, initial_lr=0.00001, train_nn=False, svm=False)
        elif num == '3':
            run_on_folder(fpath, include_shift=True, NetClass=net_class, initial_lr=0.00001, train_nn=False, svm=False)
        else:
            run_on_folder(fpath, NetClass=net_class, initial_lr=0.00001, train_nn=False, svm=False)
    print(f"Whole program finished! It took {str(datetime.timedelta(seconds=time.time()-full_start))} hours:min:seconds")

if __name__ == '__main__':
    full_start = time.time()
    # run only on ideal observer, account for varying sample sizes in calculation
    fpaths = [p.path for p in os.scandir('/scratch/reith/oo/more_nn_oo/resnet') if p.is_dir()]
    fpaths.sort(key=lambda x: int(x.split('_')[-1]), reverse=False)
    net_class = vgg16
    for fpath in fpaths:
        num = fpath.split('_')[-1]
        if num == '2':
            run_on_folder(fpath, shuffled_pixels=1, train_nn=False, svm=False)
        elif num == '3':
            run_on_folder(fpath, include_shift=True, train_nn=False, svm=False)
        else:
            run_on_folder(fpath, train_nn=False, svm=False)
    print(f"Whole program finished! It took {str(datetime.timedelta(seconds=time.time()-full_start))} hours:min:seconds")
#################################################################
if __name__ == '__main__':
    full_start = time.time()
    # run only on ideal observer, account for varying sample sizes in calculation
    fpaths = [p.path for p in os.scandir('/share/wandell/data/reith/redo_experiments/multiple_locations/multiple_locations_experiment_ideal_observer_adjusted_oo') if p.is_dir()]
    fpaths.sort(key=lambda x: int(x.split('_')[-1]), reverse=False)
    for fpath in fpaths:
        run_on_folder(fpath, svm=False, train_nn=False)
    print(f"Whole program finished! It took {str(datetime.timedelta(seconds=time.time()-full_start))} hours:min:seconds")
###################################################################
if __name__ == '__main__':
    full_start = time.time()
    # run only on ideal observer, account for varying sample sizes in calculation
    fpaths = [p.path for p in os.scandir('/share/wandell/data/reith/redo_experiments/multiple_locations/multiple_locations_experiment_ideal_observer_adjusted_oo') if p.is_dir()]
    fpaths.sort(key=lambda x: int(x.split('_')[-1]), reverse=False)
    for fpath in fpaths:
        run_on_folder(fpath, svm=False, train_nn=False)
    print(f"Whole program finished! It took {str(datetime.timedelta(seconds=time.time()-full_start))} hours:min:seconds")
#################################################################
if __name__ == '__main__':
    full_start = time.time()
    # run only on ideal observer, account for varying sample sizes in calculation
    fpaths = [p.path for p in os.scandir('/share/wandell/data/reith/redo_experiments/multiple_locations/multiple_locations_experiment_ideal_observer_adjusted') if p.is_dir()]
    fpaths.sort(key=lambda x: int(x.split('_')[-1]), reverse=False)
    for fpath in fpaths:
        run_on_folder(fpath)
    print(f"Whole program finished! It took {str(datetime.timedelta(seconds=time.time()-full_start))} hours:min:seconds")
#######################################################################3
if __name__ == '__main__':
    full_start = time.time()
    # fpaths = ['/share/wandell/data/reith/redo_experiments/multiple_locations/harmonic_frequency_of_1_loc_1_signalGridSize_1']
    fpaths = [p.path for p in os.scandir('/share/wandell/data/reith/redo_experiments/multiple_locations/multiple_locations_experiment_equal_class_dprime_adjusted') if p.is_dir()]
    # fpaths = [p.path for p in os.scandir('/share/wandell/data/reith/redo_experiments/multiple_locations/multiple_locations_experiment_equal_class_samples_all_signal') if p.is_dir()]
    fpaths.sort(key=lambda x: int(x.split('_')[-1]), reverse=True)
    for fpath in fpaths:
        run_on_folder(fpath)
    print(f"Whole program finished! It took {str(datetime.timedelta(seconds=time.time()-full_start))} hours:min:seconds")

################################################################
if __name__ == '__main__':
    # shuffle columns and normal
    full_start = time.time()
    super_path = r'/share/wandell/data/reith/redo_experiments/shuffled_pixels/redo_columns'
    columns = glob(f"{super_path}/*columns*")
    print(columns)
    for fpath in columns:
        run_on_folder(fpath, shuffled_pixels=-2)
    normal = glob(f"{super_path}/*normal*")
    print(normal)
    for fpath in normal:
        run_on_folder(fpath)
    print(f"Whole program finished! It took {str(datetime.timedelta(seconds=time.time()-full_start))} hours:min:seconds")
####################################################################
if __name__ == '__main__':
    # shuffling
    full_start = time.time()
    # super_path = '/share/wandell/data/reith/redo_experiments/redo_automaton/matlab_contrasts'
    # fpaths = [p.path for p in os.scandir(super_path) if p.is_dir()]
    # # fpaths.sort(key=lambda x: int(x.split('_')[-1]))
    # for fpath in fpaths:
    #     run_on_folder(fpath)
    super_path = '/share/wandell/data/reith/redo_experiments/shuffled_pixels/redo_shuffle_blocks'
    fpaths = [p.path for p in os.scandir(super_path) if p.is_dir()]
    fpaths.sort(key=lambda x: int(x.split('x')[-1]))
    for fpath in fpaths:
        s_pixels = int(fpath.split('x')[-1])
        run_on_folder(fpath, shuffled_pixels=s_pixels, train_nn=False, oo=False)
    print(f"Whole program finished! It took {str(datetime.timedelta(seconds=time.time()-full_start))} hours:min:seconds")
###################################################################
if __name__ == '__main__':
    # individual faces
    full_start = time.time()
    super_path = '/share/wandell/data/reith/redo_experiments/face_experiment/single_faces'
    fpaths = [p.path for p in os.scandir(super_path) if p.is_dir()]
    # fpaths.sort(key=lambda x: int(x.split('_')[-1]))
    for fpath in fpaths:
        run_on_folder(fpath)
    print(f"Whole program finished! It took {str(datetime.timedelta(seconds=time.time()-full_start))} hours:min:seconds")
###################################################################
if __name__ == '__main__':
    # automata
    full_start = time.time()
    # super_path = '/share/wandell/data/reith/redo_experiments/redo_automaton/matlab_contrasts'
    # fpaths = [p.path for p in os.scandir(super_path) if p.is_dir()]
    # # fpaths.sort(key=lambda x: int(x.split('_')[-1]))
    # for fpath in fpaths:
    #     run_on_folder(fpath)
    super_path = '/share/wandell/data/reith/redo_experiments/redo_automaton/plain_automata'
    fpaths = [p.path for p in os.scandir(super_path) if p.is_dir()]
    # fpaths.sort(key=lambda x: int(x.split('_')[-1]))
    for fpath in fpaths:
        run_on_folder(fpath)
    print(f"Whole program finished! It took {str(datetime.timedelta(seconds=time.time()-full_start))} hours:min:seconds")
######################################################################
if __name__ == '__main__':
    # disk mtf calculation. size is in pixel
    full_start = time.time()
    super_path = '/share/wandell/data/reith/redo_experiments/lines_mtf_experiments/mtf_lines_shift_higher_scene_res'
    fpaths = [p.path for p in os.scandir(super_path) if p.is_dir()]
    fpaths.sort(key=lambda x: int(x.split('_')[-1]))
    for fpath in fpaths:
        run_on_folder(fpath, include_shift=True)
    print(f"Whole program finished! It took {str(datetime.timedelta(seconds=time.time()-full_start))} hours:min:seconds")
##################################################################
if __name__ == '__main__':
    full_start = time.time()
    # super_path = '/share/wandell/data/reith/redo_experiments/lines_mtf_experiments/mtf_lines_shift_new_freq'
    # fpaths = [p.path for p in os.scandir(super_path) if p.is_dir()]
    # fpaths.sort(key=lambda x: int(x.split('_')[-1]))
    # for fpath in fpaths:
    #     run_on_folder(fpath, include_shift=True)
    # super_path = '/share/wandell/data/reith/redo_experiments/lines_mtf_experiments/mtf_lines_angle_new_freq'
    # fpaths = [p.path for p in os.scandir(super_path) if p.is_dir()]
    # fpaths.sort(key=lambda x: int(x.split('_')[-1]))
    # for fpath in fpaths:
    #     run_on_folder(fpath, include_angle=True)
    super_path = '/share/wandell/data/reith/redo_experiments/lines_mtf_experiments/mtf_lines_contrast_new_freq'
    fpaths = [p.path for p in os.scandir(super_path) if p.is_dir()]
    fpaths.sort(key=lambda x: int(x.split('_')[-1]))
    for fpath in fpaths:
        run_on_folder(fpath)
    print(f"Whole program finished! It took {str(datetime.timedelta(seconds=time.time()-full_start))} hours:min:seconds")
################################################################
"""





r"""
#####################################################################
Older runs for documentation purposes..
#####################################################################
if __name__ == '__main__':
    # disk mtf calculation. size is in pixel
    full_start = time.time()
    super_path = '/share/wandell/data/reith/redo_experiments/disks_mtf_experiment/disk_more_contrast'
    fpaths = [p.path for p in os.scandir(super_path) if p.is_dir()]
    fpaths.sort(key=lambda x: int(x.split('_')[-1]))
    for fpath in fpaths:
        run_on_folder(fpath)
    print(f"Whole program finished! It took {str(datetime.timedelta(seconds=time.time()-full_start))} hours:min:seconds")
######################################################################

if __name__ == '__main__':
    # adjust shift/angle values for lines experiment
    full_start = time.time()
    super_path = '/share/wandell/data/reith/redo_experiments/lines_mtf_experiments/mtf_lines_shift_new_freq_updated_values'
    fpaths = [p.path for p in os.scandir(super_path) if p.is_dir()]
    fpaths.sort(key=lambda x: int(x.split('_')[-1]))
    for fpath in fpaths:
        run_on_folder(fpath, include_shift=True)
    super_path = '/share/wandell/data/reith/redo_experiments/lines_mtf_experiments/mtf_lines_angle_new_freq_updated_values'
    fpaths = [p.path for p in os.scandir(super_path) if p.is_dir()]
    fpaths.sort(key=lambda x: int(x.split('_')[-1]))
    for fpath in fpaths:
        run_on_folder(fpath, include_angle=True)
    # super_path = '/share/wandell/data/reith/redo_experiments/lines_mtf_experiments/mtf_lines_contrast_new_freq'
    # fpaths = [p.path for p in os.scandir(super_path) if p.is_dir()]
    # fpaths.sort(key=lambda x: int(x.split('_')[-1]))
    # for fpath in fpaths:
    #     run_on_folder(fpath)
    print(f"Whole program finished! It took {str(datetime.timedelta(seconds=time.time()-full_start))} hours:min:seconds")
############################################################################
if __name__ == '__main__':
    # disk mtf calculation. size is in pixel
    full_start = time.time()
    super_path = '/share/wandell/data/reith/redo_experiments/disks_mtf_experiment/disk_experiment'
    fpaths = [p.path for p in os.scandir(super_path) if p.is_dir()]
    fpaths.sort(key=lambda x: int(x.split('_')[-1]))
    for fpath in fpaths:
        run_on_folder(fpath)
    print(f"Whole program finished! It took {str(datetime.timedelta(seconds=time.time()-full_start))} hours:min:seconds")
#####################################################################
if __name__ == '__main__':
    full_start = time.time()
    # super_path = '/share/wandell/data/reith/redo_experiments/lines_mtf_experiments/mtf_lines_shift_new_freq'
    # fpaths = [p.path for p in os.scandir(super_path) if p.is_dir()]
    # fpaths.sort(key=lambda x: int(x.split('_')[-1]))
    # for fpath in fpaths:
    #     run_on_folder(fpath, include_shift=True)
    # super_path = '/share/wandell/data/reith/redo_experiments/lines_mtf_experiments/mtf_lines_angle_new_freq'
    # fpaths = [p.path for p in os.scandir(super_path) if p.is_dir()]
    # fpaths.sort(key=lambda x: int(x.split('_')[-1]))
    # for fpath in fpaths:
    #     run_on_folder(fpath, include_angle=True)
    super_path = '/share/wandell/data/reith/redo_experiments/lines_mtf_experiments/mtf_lines_contrast_new_freq'
    fpaths = [p.path for p in os.scandir(super_path) if p.is_dir()]
    fpaths.sort(key=lambda x: int(x.split('_')[-1]))
    for fpath in fpaths:
        run_on_folder(fpath)
    print(f"Whole program finished! It took {str(datetime.timedelta(seconds=time.time()-full_start))} hours:min:seconds")
#####################################################################
if __name__ == '__main__':
    full_start = time.time()
    # super_path = '/share/wandell/data/reith/redo_experiments/face_experiment'
    fpaths = ['/share/wandell/data/reith/redo_experiments/shuffled_pixels/face_signal/faces_shuff_rows']
    for fpath in fpaths:
        # run_on_folder(fpath, include_angle=True)
        run_on_folder(fpath, shuffled_pixels=-1)
    fpaths = ['/share/wandell/data/reith/redo_experiments/shuffled_pixels/face_signal/faces_shuff_columns']
    for fpath in fpaths:
        # run_on_folder(fpath, include_angle=True)
        run_on_folder(fpath, shuffled_pixels=-2)
    fpaths = ['/share/wandell/data/reith/redo_experiments/shuffled_pixels/face_signal/faces_shuff_pixels']
    for fpath in fpaths:
        # run_on_folder(fpath, include_angle=True)
        run_on_folder(fpath, shuffled_pixels=True)
    print(f"Whole program finished! It took {str(datetime.timedelta(seconds=time.time()-full_start))} hours:min:seconds")
#######################################################################
if __name__ == '__main__':
    full_start = time.time()
    # super_path = '/share/wandell/data/reith/redo_experiments/face_experiment'
    super_paths = ['/share/wandell/data/reith/redo_experiments/cellular_automaton/class_3', '/share/wandell/data/reith/redo_experiments/cellular_automaton/class_2']
    # super_paths = [r'C:\Users\Fabian\Documents\data\rsync\redo_experiments\cellular_automaton\class_2', r'C:\Users\Fabian\Documents\data\rsync\redo_experiments\cellular_automaton\class_3']
    fpaths = [p.path for p in os.scandir(super_paths[0]) if p.is_dir()]
    fpaths.extend([p.path for p in os.scandir(super_paths[1]) if p.is_dir()])
    for fpath in fpaths:
        # run_on_folder(fpath, include_angle=True)
        ca_rule = int(fpath.split('_')[-1])
        run_on_folder(fpath, ca_rule=ca_rule)
    print(f"Whole program finished! It took {str(datetime.timedelta(seconds=time.time()-full_start))} hours:min:seconds")
#####################################################################
if __name__ == '__main__':
    full_start = time.time()
    # super_path = '/share/wandell/data/reith/redo_experiments/face_experiment'
    fpaths = ['/share/wandell/data/reith/redo_experiments/shuffled_pixels/shuffled_columns']
    for fpath in fpaths:
        # run_on_folder(fpath, include_angle=True)
        run_on_folder(fpath, shuffled_pixels=-2)
    print(f"Whole program finished! It took {str(datetime.timedelta(seconds=time.time()-full_start))} hours:min:seconds")
#####################################################################
if __name__ == '__main__':
    full_start = time.time()
    # super_path = '/share/wandell/data/reith/redo_experiments/face_experiment'
    fpaths = ['/share/wandell/data/reith/redo_experiments/cellular_automaton/rule_110_on_harmonic_freq_1']
    # fpaths = ['/share/wandell/data/reith/redo_experiments/shuffled_pixels/shuffled_columns']
    # fpaths.sort(key=lambda x: int(x.split('_')[-1]))
    for fpath in fpaths:
        # run_on_folder(fpath, include_angle=True)
        # run_on_folder(fpath, shuffled_pixels=-2)
        run_on_folder(fpath)
    print(f"Whole program finished! It took {str(datetime.timedelta(seconds=time.time()-full_start))} hours:min:seconds")
#########################################################################
if __name__ == '__main__':
    full_start = time.time()
    # super_path = '/share/wandell/data/reith/redo_experiments/face_experiment'
    # fpaths = ['/share/wandell/data/reith/redo_experiments/cellular_automaton/rule_110_on_harmonic_freq_1']
    fpaths = ['/share/wandell/data/reith/redo_experiments/cellular_automaton/rule_45_on_harmonic_freq_1',
              '/share/wandell/data/reith/redo_experiments/cellular_automaton/rule_105_on_harmonic_freq_1',
              '/share/wandell/data/reith/redo_experiments/cellular_automaton/rule_154_on_harmonic_freq_1']
    # fpaths.sort(key=lambda x: int(x.split('_')[-1]))
    for fpath in fpaths:
        # run_on_folder(fpath, include_angle=True)
        run_on_folder(fpath, ca_rule=110)
    print(f"Whole program finished! It took {str(datetime.timedelta(seconds=time.time()-full_start))} hours:min:seconds")
######################################################################
if __name__ == '__main__':
    full_start = time.time()
    super_path = '/share/wandell/data/reith/redo_experiments/face_experiment'
    # super_path = r'C:\Users\Fabian\Documents\data\windows2rsync\windows_data\mtf'
    super_path = '/share/wandell/data/reith/redo_experiments/mtf_experiments/mtf_angle_new_freq'
    super_path = '/share/wandell/data/reith/redo_experiments/mtf_experiments/mtf_shift_new_freq'
    fpaths = [p.path for p in os.scandir(super_path) if p.is_dir()]
    # fpaths.sort(key=lambda x: int(x.split('_')[-1]))
    for fpath in fpaths:
        # run_on_folder(fpath, include_angle=True)
        run_on_folder(fpath, include_shift=True)
    print(f"Whole program finished! It took {str(datetime.timedelta(seconds=time.time()-full_start))} hours:min:seconds")


if __name__ == '__main__':
    full_start = time.time()
    # super_path = '/share/wandell/data/reith/redo_experiments/face_experiment'
    fpaths = ['/share/wandell/data/reith/redo_experiments/face_experiment/multi_face_result']
    # fpaths.sort(key=lambda x: int(x.split('_')[-1]))
    for fpath in fpaths:
        # run_on_folder(fpath, include_angle=True)
        run_on_folder(fpath)
    print(f"Whole program finished! It took {str(datetime.timedelta(seconds=time.time()-full_start))} hours:min:seconds")
####################################################################
if __name__ == '__main__':
    full_start = time.time()
    super_path = '/share/wandell/data/reith/redo_experiments/mtf_experiments/mtf_contrast_new_freq'
    # super_path = r'C:\Users\Fabian\Documents\data\windows2rsync\windows_data\mtf'
    fpaths = [p.path for p in os.scandir(super_path) if p.is_dir()]
    fpaths.sort(key=lambda x: int(x.split('_')[-1]))
    for fpath in fpaths:
        run_on_folder(fpath)
    print(f"Whole program finished! It took {str(datetime.timedelta(seconds=time.time()-full_start))} hours:min:seconds")
#######################################################
if __name__ == '__main__':
    full_start = time.time()
    super_path = '/share/wandell/data/reith/redo_experiments/mtf'
    # super_path = r'C:\Users\Fabian\Documents\data\windows2rsync\windows_data\mtf'
    fpaths = [p.path for p in os.scandir(super_path) if p.is_dir()]
    fpaths.sort(key=lambda x: int(x.split('_')[-1]))
    for fpath in fpaths:
        run_on_folder(fpath)
    print(f"Whole program finished! It took {str(datetime.timedelta(seconds=time.time()-full_start))} hours:min:seconds")
#####################################################################
if __name__ == '__main__':
    full_start = time.time()
    super_path = '/share/wandell/data/reith/redo_experiments/shuffled_pixels/different_shuffle_portion_inverse'
    super_path = '/share/wandell/data/reith/redo_experiments/shuffled_pixels/different_shuffle_portion'
    # super_path = r'C:\Users\Fabian\Documents\data\rsync\redo_experiments\shuffled_pixels\different_shuffle_portion'
    fpaths = [p.path for p in os.scandir(super_path) if p.is_dir()]
    fpaths.sort(key=lambda x: int(x.split('_')[-1]))
    for fpath in fpaths:
        # if int(fpath.split('x')[-1]) not in [1, 2, 4, 7, 21, 35]:
        #     continue
        shuffle_portion = int(fpath.split('_')[-1])
        run_on_folder(fpath, them_cones=False, separate_rgb=False, meanData_rounding=None, shuffled_pixels=True, shuffle_portion=shuffle_portion, svm=True, test_eval=True)
    print(f"Whole program finished! It took {str(datetime.timedelta(seconds=time.time()-full_start))} hours:min:seconds")
###########################################################################
if __name__ == '__main__' and False:
    full_start = time.time()
    super_path = '/share/wandell/data/reith/redo_experiments/shuffled_pixels/different_shuffle_scope'
    # super_path = r'C:\Users\Fabian\Documents\data\rsync\redo_experiments\shuffled_pixels\different_shuffle_scope'
    fpaths = [p.path for p in os.scandir(super_path) if p.is_dir()]
    fpaths.sort(key=lambda x: int(x.split('_')[-1]))
    for fpath in fpaths:
        # if int(fpath.split('x')[-1]) not in [1, 2, 4, 7, 21, 35]:
        #     continue
        shuffle_scope = int(fpath.split('_')[-1])
        run_on_folder(fpath, them_cones=False, separate_rgb=False, meanData_rounding=None, shuffled_pixels=True, shuffle_scope=shuffle_scope, svm=True, test_eval=True)
    print(f"Whole program finished! It took {str(datetime.timedelta(seconds=time.time()-full_start))} hours:min:seconds")
##########################################################################################################
if __name__ == '__main__':
    full_start = time.time()
    super_path = '/share/wandell/data/reith/redo_experiments/sample_number_contrast/resnet'
    # super_path = r'C:\Users\Fabian\Documents\data\rsync\redo_experiments\sample_number_contrast\resnet'
    fpaths = [p.path for p in os.scandir(super_path) if p.is_dir()]
    fpaths.sort(key=lambda k: int(k.split("_")[-1]))
    for fpath in fpaths:
        train_set_size = int(fpath.split('_')[-1])
        if train_set_size not in [107689, 193283]:
            continue
        run_on_folder(fpath, train_set_size=train_set_size)
    print(f"Whole program finished! It took {str(datetime.timedelta(seconds=time.time()-full_start))} hours:min:seconds")
############################################################################################################
if __name__ == '__main__':
    full_start = time.time()
    fpath = '/share/wandell/data/reith/redo_experiments/sanity_test_random_data_generation'
    run_on_folder(fpath, them_cones=False, separate_rgb=False, meanData_rounding=None, shuffled_pixels=False, svm=True, test_eval=True)
    print(f"Whole program finished! It took {str(datetime.timedelta(seconds=time.time()-full_start))} hours:min:seconds")
############################################################################################################
if __name__ == '__main__':
    full_start = time.time()
    super_path = '/share/wandell/data/reith/redo_experiments/shuffled_pixels/different_patch_sizes'
    # super_path = r'C:\Users\Fabian\Documents\data\rsync\redo_experiments\shuffled_pixels\different_patch_sizes'
    fpaths = [p.path for p in os.scandir(super_path) if p.is_dir()]
    for fpath in fpaths:
        if int(fpath.split('x')[-1]) not in [1, 2, 4, 7, 21, 35]:
            continue
        run_on_folder(fpath, them_cones=False, separate_rgb=False, meanData_rounding=None, shuffled_pixels=int(fpath.split('x')[-1]), svm=True, test_eval=True)
    print(f"Whole program finished! It took {str(datetime.timedelta(seconds=time.time()-full_start))} hours:min:seconds")
########################################################3
if __name__ == '__main__':
    full_start = time.time()
    fpath = '/share/wandell/data/reith/redo_experiments/shuffled_pixels/sensor_harmonic_contrasts/'
    run_on_folder(fpath, them_cones=False, separate_rgb=False, meanData_rounding=None, shuffled_pixels=True, svm=True, test_eval=True)
    fpath = '/share/wandell/data/reith/redo_experiments/shuffled_pixels/sensor_harmonic_phase_shift/'
    run_on_folder(fpath, them_cones=False, separate_rgb=False, meanData_rounding=None, shuffled_pixels=True, svm=True, test_eval=True, include_shift=True)
    fpath = '/share/wandell/data/reith/redo_experiments/shuffled_pixels/sensor_harmonic_rotation/'
    run_on_folder(fpath, them_cones=False, separate_rgb=False, meanData_rounding=None, shuffled_pixels=True, svm=True, test_eval=True, include_angle=True)
    print(f"Whole program finished! It took {str(datetime.timedelta(seconds=time.time()-full_start))} hours:min:seconds")
###################################################
if __name__ == '__main__':
    full_start = time.time()
    fpath = '/share/wandell/data/reith/circle_fun/h5_data/white_circle_rad_2/'
    run_on_folder(fpath, them_cones=False, separate_rgb=False, meanData_rounding=None, shuffled_pixels=False, svm=True, test_eval=False)
    fpath = '/share/wandell/data/reith/circle_fun/h5_data/white_circle_rad_4/'
    run_on_folder(fpath, them_cones=False, separate_rgb=False, meanData_rounding=None, shuffled_pixels=False, svm=True, test_eval=False)
    fpath = '/share/wandell/data/reith/circle_fun/h5_data/white_circle_rad_8/'
    run_on_folder(fpath, them_cones=False, separate_rgb=False, meanData_rounding=None, shuffled_pixels=False, svm=True, test_eval=False)
    fpath = '/share/wandell/data/reith/circle_fun/h5_data/white_circle_rad_15/'
    run_on_folder(fpath, them_cones=False, separate_rgb=False, meanData_rounding=None, shuffled_pixels=False, svm=True, test_eval=False)
    fpath = '/share/wandell/data/reith/circle_fun/h5_data/white_circle_rad_32/'
    run_on_folder(fpath, them_cones=False, separate_rgb=False, meanData_rounding=None, shuffled_pixels=False, svm=True, test_eval=False)
    print(f"Whole program finished! It took {str(datetime.timedelta(seconds=time.time()-full_start))} hours:min:seconds")
###################################################
if __name__ == '__main__':
    full_start = time.time()
    fpath = '/share/wandell/data/reith/coneMosaik/signal_location_experiment_bnfix/multiple_locations_freq1/'
    run_on_folder(fpath, them_cones=True, separate_rgb=False, meanData_rounding=None, shuffled_pixels=False, svm=True)
    fpath = '/share/wandell/data/reith/coneMosaik/signal_location_experiment_bnfix/one_location_freq1/'
    run_on_folder(fpath, them_cones=True, separate_rgb=False, meanData_rounding=None, shuffled_pixels=False, svm=True)
    print(f"Whole program finished! It took {str(datetime.timedelta(seconds=time.time()-full_start))} hours:min:seconds")
#################################################
if __name__ == '__main__':
    full_start = time.time()
    super_path = '/share/wandell/data/reith/coneMosaik/various_rounding_rounds_eval/'
    fpaths = [p.path for p in os.scandir(super_path) if p.is_dir()]
    fpaths.sort()
    run_on_folder(fpaths.pop(0), them_cones=False, separate_rgb=False, meanData_rounding=None, svm=True, test_eval=True)
    for i, path in enumerate(fpaths, start=0):
        print(f'Running on {path} with rounding to {i}f decimals.')
        run_on_folder(path, svm=True, them_cones=False, separate_rgb=False, meanData_rounding=i, test_eval=True)
    print(f"Whole program finished! It took {str(datetime.timedelta(seconds=time.time()-full_start))} hours:min:seconds")

if __name__ == '__main__':
    full_start = time.time()
    fpath = '/share/wandell/data/reith/coneMosaik/shuffled_pixels/'
    run_on_folder(fpath, them_cones=False, separate_rgb=False, meanData_rounding=None, shuffled_pixels=True, svm=True)
    print(f"Whole program finished! It took {str(datetime.timedelta(seconds=time.time()-full_start))} hours:min:seconds")

#################################################
if __name__ == '__main__':
    full_start = time.time()
    fpath = '/share/wandell/data/reith/circle_fun/h5_data/white_circle_rad_69/'
    run_on_folder(fpath, them_cones=False, separate_rgb=False, meanData_rounding=None, shuffled_pixels=False, svm=True, test_eval=False)
    fpath = '/share/wandell/data/reith/coneMosaik/signal_location_experiment/one_location_freq1/'
    # run_on_folder(fpath, them_cones=True, separate_rgb=False, meanData_rounding=None, shuffled_pixels=False, svm=True)
    print(f"Whole program finished! It took {str(datetime.timedelta(seconds=time.time()-full_start))} hours:min:seconds")
#################################################
if __name__ == '__main__':
    full_start = time.time()
    fpath = '/share/wandell/data/reith/coneMosaik/signal_location_experiment/multiple_locations_freq1/'
    run_on_folder(fpath, them_cones=True, separate_rgb=False, meanData_rounding=None, shuffled_pixels=False, svm=True)
    fpath = '/share/wandell/data/reith/coneMosaik/signal_location_experiment/one_location_freq1/'
    # run_on_folder(fpath, them_cones=True, separate_rgb=False, meanData_rounding=None, shuffled_pixels=False, svm=True)
    print(f"Whole program finished! It took {str(datetime.timedelta(seconds=time.time()-full_start))} hours:min:seconds")
#################################################
if __name__ == '__main__':
    full_start = time.time()
    fpath = '/share/wandell/data/reith/coneMosaik/signal_location_experiment/multiple_locations_freq1/'
    run_on_folder(fpath, them_cones=True, separate_rgb=False, meanData_rounding=None, shuffled_pixels=False, svm=True)
    fpath = '/share/wandell/data/reith/coneMosaik/signal_location_experiment/one_location_freq1/'
    # run_on_folder(fpath, them_cones=True, separate_rgb=False, meanData_rounding=None, shuffled_pixels=False, svm=True)
    print(f"Whole program finished! It took {str(datetime.timedelta(seconds=time.time()-full_start))} hours:min:seconds")
#################################################
if __name__ == '__main__':
    full_start = time.time()
    super_path = '/share/wandell/data/reith/coneMosaik/various_rounding_rounds/'
    fpaths = [p.path for p in os.scandir(super_path) if p.is_dir()]
    fpaths.sort()
    run_on_folder(fpaths.pop(0), them_cones=False, separate_rgb=False, meanData_rounding=None, svm=True)
    for i, path in enumerate(fpaths, start=1):
        print(f'Running on {path} with rounding to {i}f decimals.')
        run_on_folder(path, svm=True, them_cones=False, separate_rgb=False, meanData_rounding=i)
    print(f"Whole program finished! It took {str(datetime.timedelta(seconds=time.time()-full_start))} hours:min:seconds")
#################################################
if __name__ == '__main__':
    full_start = time.time()
    i = 0
    print(f"Round to {i} decimals.")
    run_on_folder('/share/wandell/data/reith/coneMosaik/freq1_sensor_data_round/', them_cones=False, separate_rgb=False, meanData_rounding=i, svm=True)
    print(f"Whole program finished! It took {str(datetime.timedelta(seconds=time.time()-full_start))} hours:min:seconds")
#################################################
if __name__ == '__main__':
    full_start = time.time()
    super_path = '/share/wandell/data/reith/coneMosaik/various_rounding_rounds/'
    fpaths = [p.path for p in os.scandir(super_path) if p.is_dir()]
    fpaths.sort()
    # run_on_folder(fpaths.pop(0), them_cones=False, separate_rgb=False, meanData_rounding=None, svm=True)
    for i, path in enumerate(fpaths, start=1):
        print(path)
        # run_on_folder(path, svm=True, them_cones=False, separate_rgb=False, meanData_rounding=i)

    print(f"Whole program finished! It took {str(datetime.timedelta(seconds=time.time()-full_start))} hours:min:seconds")

#################################################
if __name__ == '__main__':
    full_start = time.time()
    fpath = '/share/wandell/data/reith/coneMosaik/shuffled_pixels/'
    run_on_folder(fpath, them_cones=False, separate_rgb=False, meanData_rounding=None, shuffled_pixels=True, svm=True)
    print(f"Whole program finished! It took {str(datetime.timedelta(seconds=time.time()-full_start))} hours:min:seconds")
#################################################
if __name__ == '__main__':
    full_start = time.time()
    run_on_folder('/share/wandell/data/reith/coneMosaik/sensor_sanity_7decimal_mean/', separate_rgb=False, svm=True)
    print(f"Whole program finished! It took {str(datetime.timedelta(seconds=time.time()-full_start))} hours:min:seconds")
###################################################
Calibrate static cone mosaic case
    full_start = time.time()
    run_on_folder('/share/wandell/data/reith/coneMosaik/static_case_freq1_var_contrasts/', separate_rgb=False, svm=True)
    run_on_folder('/share/wandell/data/reith/coneMosaik/static_case_freq1_var_contrasts_rgb/', separate_rgb=True, svm=True)
    print(f"Whole program finished! It took {str(datetime.timedelta(seconds=time.time()-full_start))} hours:min:seconds")
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
original one:
Calibrate static cone mosaic case
if __name__ == '__main__':
    full_start = time.time()
    run_on_folder('/share/wandell/data/reith/coneMosaik/static_case_freq1_var_contrasts/', separate_rgb=False, svm=True)
    run_on_folder('/share/wandell/data/reith/coneMosaik/static_case_freq1_var_contrasts_rgb/', separate_rgb=True, NetClass=NotPretrainedResnet, svm=True)
    print(f"Whole program finished! It took {str(datetime.timedelta(seconds=time.time()-full_start))} hours:min:seconds")
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#################################################
test higher start lr rates
if __name__ == '__main__':
    full_start = time.time()
    run_on_folder('/share/wandell/data/reith/imagenet_training/different_training_params/30_epochs_lr_0_01_lr_deviation_0_1_lr_epoch_reps_3/', num_epochs=30, initial_lr=0.01, lr_deviation=0.1, lr_epoch_reps=3)
    run_on_folder('/share/wandell/data/reith/imagenet_training/different_training_params/48_epochs_lr_0_01_lr_deviation_0_1_lr_epoch_reps_3', num_epochs=48, initial_lr=0.01, lr_deviation=0.1, lr_epoch_reps=3)
    run_on_folder('/share/wandell/data/reith/imagenet_training/different_training_params/48_epochs_lr_0_01_lr_deviation_0_1_lr_epoch_reps_4', num_epochs=48, initial_lr=0.01, lr_deviation=0.1, lr_epoch_reps=4)
    run_on_folder('/share/wandell/data/reith/imagenet_training/different_training_params/48_epochs_lr_0_01_lr_deviation_0_33_lr_epoch_reps_6', num_epochs=48, initial_lr=0.01, lr_deviation=0.33, lr_epoch_reps=6)
    run_on_folder('/share/wandell/data/reith/imagenet_training/different_training_params/48_epochs_lr_0_001_lr_deviation_0_33_lr_epoch_reps_6', num_epochs=48, initial_lr=0.001, lr_deviation=0.33, lr_epoch_reps=6)
    run_on_folder('/share/wandell/data/reith/imagenet_training/different_training_params/48_epochs_lr_0_01_lr_deviation_0_33_lr_epoch_reps_12', num_epochs=48, initial_lr=0.01, lr_deviation=0.33, lr_epoch_reps=12)
    print(f"Whole program finished! It took {str(datetime.timedelta(seconds=time.time()-full_start))} hours:min:seconds")
##################################################
if __name__ == '__main__':
    full_start = time.time()
    # run_on_folder('/share/wandell/data/reith/imagenet_training/different_training_params/more_epochs_lower_lr/', num_epochs=48, initial_lr=0.001, lr_deviation=0.1, lr_epoch_reps=4)
    # run_on_folder('/share/wandell/data/reith/imagenet_training/different_training_params/more_epochs_same_lr/', num_epochs=48, initial_lr=0.001, lr_deviation=0.1, lr_epoch_reps=3)
    run_on_folder('/share/wandell/data/reith/imagenet_training/different_training_params/more_epochs_slower_lr_decline/', num_epochs=48, initial_lr=0.001, lr_deviation=0.33, lr_epoch_reps=12)
    run_on_folder('/share/wandell/data/reith/imagenet_training/different_training_params/standard_lr/', num_epochs=30, initial_lr=0.001, lr_deviation=0.1, lr_epoch_reps=3)
    print(f"Whole program finished! It took {str(datetime.timedelta(seconds=time.time()-full_start))} hours:min:seconds")

##################################################
if __name__ == '__main__':
    full_start = time.time()
    run_on_folder('/share/wandell/data/reith/imagenet_training/freq1_harmonic_random', deeper_pls=False, NetClass=NotPretrainedResnet)
    run_on_folder('/share/wandell/data/reith/imagenet_training/freq1_harmonic_pretrained', deeper_pls=False, NetClass=PretrainedResnetFrozen, NetClass_param=0)
    print(f"Whole program finished! It took {str(datetime.timedelta(seconds=time.time()-full_start))} hours:min:seconds")
###################################################
if __name__ == '__main__':
    full_start = time.time()
    general_folder = '/share/wandell/data/reith/2_class_MTF_angle_experiment/'
    frequency_folders = [os.path.join(general_folder, f) for f in os.listdir(general_folder)]
    frequency_folders.sort(key=lambda k: int(k.split('_')[-1]))
    for f in frequency_folders:
        run_on_folder(f, deeper_pls=False, NetClass=None)
    with open(general_folder, 'w') as txt:
        txt.write(f"Whole program finished! It took {str(datetime.timedelta(seconds=time.time()-full_start))} hours:min:seconds")
    print(f"Whole program finished! It took {str(datetime.timedelta(seconds=time.time()-full_start))} hours:min:seconds")
###################################################`
if __name__ == '__main__':
    full_start = time.time()
    run_folder = '/share/wandell/data/reith/harmonic_angle_calibration/'
    run_on_folder(run_folder, deeper_pls=False, NetClass=None)
    with open(run_folder + 'time.txt', 'w') as txt:
        txt.write(f"Whole program finished! It took {str(datetime.timedelta(seconds=time.time()-full_start))} hours:min:seconds")
    print(f"Whole program finished! It took {str(datetime.timedelta(seconds=time.time()-full_start))} hours:min:seconds")
##################################################
if __name__ == '__main__':
    full_start = time.time()
    general_folder = '/share/wandell/data/reith/2_class_MTF_shift_experiment/'
    frequency_folders = [os.path.join(general_folder, f) for f in os.listdir(general_folder)]
    frequency_folders.sort(key=lambda k: int(k.split('_')[-1]))
    for f in frequency_folders:
        run_on_folder(f, deeper_pls=False, NetClass=None)
    with open(general_folder, 'w') as txt:
        txt.write(f"Whole program finished! It took {str(datetime.timedelta(seconds=time.time()-full_start))} hours:min:seconds")
    print(f"Whole program finished! It took {str(datetime.timedelta(seconds=time.time()-full_start))} hours:min:seconds")
###################################################
if __name__ == '__main__':
    full_start = time.time()
    run_on_folder('/share/wandell/data/reith/harmonic_shift_calibration_include_shifts/', deeper_pls=False, NetClass=None)
    with open('/share/wandell/data/reith/harmonic_shift_calibration_include_shifts/time.txt', 'w') as txt:
        txt.write(f"Whole program finished! It took {str(datetime.timedelta(seconds=time.time()-full_start))} hours:min:seconds")
    print(f"Whole program finished! It took {str(datetime.timedelta(seconds=time.time()-full_start))} hours:min:seconds")
##################################################
if __name__ == '__main__':
    full_start = time.time()
    for i in range(1,21):
        run_on_folder(f'/share/wandell/data/reith/2_class_MTF_freq_experiment/frequency_{i}/', deeper_pls=False, NetClass=None)
    with open('/share/wandell/data/reith/2_class_MTF_freq_experiment/time.txt', 'w') as txt:
        txt.write(f"Whole program finished! It took {str(datetime.timedelta(seconds=time.time()-full_start))} hours:min:seconds")
    print(f"Whole program finished! It took {str(datetime.timedelta(seconds=time.time()-full_start))} hours:min:seconds")
##################################################
if __name__ == '__main__':
    print("sleeping for a bit..")
    time.sleep(2*3600)
    full_start = time.time()
    run_on_folder('/share/wandell/data/reith/experiment_freq_1_log_contrasts30_frozen_until_0', deeper_pls=False, NetClass=NotPretrainedResnet)
    run_on_folder('/share/wandell/data/reith/experiment_freq_1_log_contrasts30_frozen_until_1', deeper_pls=False, NetClass=PretrainedResnetFrozen, NetClass_param=1)
    run_on_folder('/share/wandell/data/reith/experiment_freq_1_log_contrasts30_frozen_until_2', deeper_pls=False, NetClass=PretrainedResnetFrozen, NetClass_param=2)
    run_on_folder('/share/wandell/data/reith/experiment_freq_1_log_contrasts30_frozen_until_3', deeper_pls=False, NetClass=PretrainedResnetFrozen, NetClass_param=3)
    run_on_folder('/share/wandell/data/reith/experiment_freq_1_log_contrasts30_frozen_until_4', deeper_pls=False, NetClass=PretrainedResnetFrozen, NetClass_param=4)
    print(f"Whole program finished! It took {str(datetime.timedelta(seconds=time.time()-full_start))} hours:min:seconds")
##################################################

"""
