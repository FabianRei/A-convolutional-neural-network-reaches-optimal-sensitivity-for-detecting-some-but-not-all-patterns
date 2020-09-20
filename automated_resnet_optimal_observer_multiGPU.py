from deepLearning.src.models.trainFromMatfile import autoTrain_Resnet_optimalObserver
from deepLearning.src.models.Resnet import PretrainedResnetFrozen, NotPretrainedResnet
from glob import glob
import GPUtil
import multiprocessing as mp
import time
import datetime

deviceIDs = GPUtil.getAvailable(order = 'first', limit = 6, maxLoad = 0.1, maxMemory = 0.1, excludeID=[], excludeUUID=[])
pathMatDir = "/share/wandell/data/reith/experiment_freq_1_log_contrasts30_higher_nonfrozen_resnet/"
programStart = time.time()
print(deviceIDs)


def matfile_gen(pathMatDir):
    matFiles = glob(f'{pathMatDir}**/*.h5', recursive=True)
    matFiles.sort()
    for matFile in matFiles:
        yield matFile


pathGen = matfile_gen(pathMatDir)
Procs = {}
lock = mp.Lock()
while True:
    try:
        if Procs == {}:
            for device in deviceIDs:
                pathMat = next(pathGen)
                print(f"Running {pathMat} on GPU {device}")
                currP = mp.Process(target=autoTrain_Resnet_optimalObserver, args=[pathMat],
                                   kwargs={'device': int(device), 'lock': lock, 'train_nn': True, 'include_shift': False,
                                           'NetClass': NotPretrainedResnet})
                Procs[str(device)] = currP
                currP.start()
        for device, proc in Procs.items():
            if not proc.is_alive():
                pathMat = next(pathGen)
                print(f"Running {pathMat} on GPU {device}")
                currP = mp.Process(target=autoTrain_Resnet_optimalObserver, args=[pathMat],
                                   kwargs={'device': int(device), 'lock': lock, 'train_nn': True, 'include_shift': False,
                                           'NetClass': NotPretrainedResnet})
                Procs[str(device)] = currP
                currP.start()
    except StopIteration:
        break

    time.sleep(5)


for proc in Procs.values():
    proc.join()

programEnd = time.time()

print(f"Whole program finished! It took {str(datetime.timedelta(seconds=programEnd-programStart))} hours:min:seconds")
print("done!")
