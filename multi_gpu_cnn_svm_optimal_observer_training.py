import os, sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from .src.models.trainFromMatfile import train_cnn_svm_optimal_observer
from glob import glob
import GPUtil
import multiprocessing as mp
import time
import datetime
import os
import fnmatch

from .src.models.new_inception import inceptionv3
from .src.models.new_vgg import vgg16, vgg16bn
from .src.models.new_alexnet import alexnet


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
                    currP = mp.Process(target=train_cnn_svm_optimal_observer, args=[pathMat],
                                       kwargs={'device': int(device), 'lock': lock, **kword_args, **kwargs})
                    Procs[str(device)] = currP
                    currP.start()
            for device, proc in Procs.items():
                if not proc.is_alive():
                    time.sleep(30)
                    pathMat = next(pathGen)
                    print(f"Running {pathMat} on GPU {device}")
                    currP = mp.Process(target=train_cnn_svm_optimal_observer, args=[pathMat],
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
################################################################
ALL RUNS ARE AUTOMATICALLY SEEDED
################################################################
FIGURE 2
################################################################
if __name__ == '__main__':
    full_start = time.time()
    super_path = '/share/wandell/data/reith/redo_experiments/lines_mtf_experiments/mtf_lines_contrast_new_freq'
    fpaths = [p.path for p in os.scandir(super_path) if p.is_dir()]
    fpaths.sort(key=lambda x: int(x.split('_')[-1]))
    for fpath in fpaths:
        run_on_folder(fpath)
    print(f"Whole program finished! It took {str(datetime.timedelta(seconds=time.time()-full_start))} hours:min:seconds")

################################################################
FIGURE 3
################################################################
if __name__ == '__main__':
    # disk mtf calculation. size is in pixel
    full_start = time.time()
    super_path = '/share/wandell/data/reith/redo_experiments/lines_mtf_experiments/mtf_lines_shift_higher_scene_res'
    fpaths = [p.path for p in os.scandir(super_path) if p.is_dir()]
    fpaths.sort(key=lambda x: int(x.split('_')[-1]))
    for fpath in fpaths:
        run_on_folder(fpath, include_shift=True)
    print(f"Whole program finished! It took {str(datetime.timedelta(seconds=time.time()-full_start))} hours:min:seconds")

################################################################
FIGURE 4
################################################################
if __name__ == '__main__':
    # individual faces
    full_start = time.time()
    super_path = '/share/wandell/data/reith/redo_experiments/face_experiment/single_faces'
    fpaths = [p.path for p in os.scandir(super_path) if p.is_dir()]
    # fpaths.sort(key=lambda x: int(x.split('_')[-1]))
    for fpath in fpaths:
        run_on_folder(fpath)
    print(f"Whole program finished! It took {str(datetime.timedelta(seconds=time.time()-full_start))} hours:min:seconds")

################################################################
FIGURE 5
################################################################
if __name__ == '__main__':
    full_start = time.time()
    super_path = '/share/wandell/data/reith/redo_experiments/redo_automaton/matlab_contrasts'
    fpaths = [p.path for p in os.scandir(super_path) if p.is_dir()]
    # fpaths.sort(key=lambda x: int(x.split('_')[-1]))
    for fpath in fpaths:
        run_on_folder(fpath)
    print(f"Whole program finished! It took {str(datetime.timedelta(seconds=time.time()-full_start))} hours:min:seconds")

################################################################
FIGURE 6
################################################################
if __name__ == '__main__':
    full_start = time.time()
    super_path = '/share/wandell/data/reith/redo_experiments/shuffled_pixels/redo_shuffle_blocks'
    fpaths = [p.path for p in os.scandir(super_path) if p.is_dir()]
    fpaths.sort(key=lambda x: int(x.split('x')[-1]))
    for fpath in fpaths:
        s_pixels = int(fpath.split('x')[-1])
        run_on_folder(fpath, shuffled_pixels=s_pixels, train_nn=False, oo=False)
    print(f"Whole program finished! It took {str(datetime.timedelta(seconds=time.time()-full_start))} hours:min:seconds")

################################################################
FIGURE 7
################################################################
if __name__ == '__main__':
    full_start = time.time()
    # run only on ideal observer, account for varying sample sizes in calculation
    fpaths = [p.path for p in os.scandir('/share/wandell/data/reith/redo_experiments/multiple_locations/multiple_locations_experiment_ideal_observer_adjusted_oo') if p.is_dir()]
    fpaths.sort(key=lambda x: int(x.split('_')[-1]), reverse=False)
    for fpath in fpaths:
        run_on_folder(fpath, svm=True, train_nn=True)
    print(f"Whole program finished! It took {str(datetime.timedelta(seconds=time.time()-full_start))} hours:min:seconds")

################################################################
FIGURE 8
################################################################
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

__________________________________________________________________________________________________________

################################################################
FIGURE A1
################################################################
if __name__ == '__main__':
    full_start = time.time()
    super_path = '/share/wandell/data/reith/redo_experiments/sample_number_contrast/resnet'
    fpaths = [p.path for p in os.scandir(super_path) if p.is_dir()]
    fpaths.sort(key=lambda k: int(k.split("_")[-1]))
    for fpath in fpaths:
        train_set_size = int(fpath.split('_')[-1])
        run_on_folder(fpath, train_set_size=train_set_size)
    print(f"Whole program finished! It took {str(datetime.timedelta(seconds=time.time()-full_start))} hours:min:seconds")

################################################################
FIGURE A2
################################################################
if __name__ == '__main__':
    full_start = time.time()
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



################################################################
ADDITIONAL RUNS TO GET RESULTS WITH OTHER SEEDS. FOR SPECIFIC FIGURES WITH THE PARAMETERS SHOWN ABOVE
################################################################
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
            run_on_folder(fpath, random_seed=seed)
    print(f"Whole program finished! It took {str(datetime.timedelta(seconds=time.time()-full_start))} hours:min:seconds")
"""
