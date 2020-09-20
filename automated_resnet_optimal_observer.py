from deepLearning.src.models.trainFromMatfile import autoTrain_Resnet_optimalObserver
from deepLearning.src.models.Resnet import PretrainedResnetFrozen
from glob import glob
import time
import datetime

pathMatDir = '/share/wandell/data/reith/experiment_freq_1_log_contrasts20_frozen_pretrained_resnet/'

matFiles = glob(f'{pathMatDir}*.h5')
matFiles.sort()
matFiles = matFiles[15:17]
programStart = time.time()
for matFile in matFiles:
    if matFile[-5:-3] == 'oo':
        print(f"Only optimal observer for: {matFile}")
        autoTrain_Resnet_optimalObserver(matFile, train_nn=False, include_shift=True)
    else:
        print(matFile)
        autoTrain_Resnet_optimalObserver(matFile, train_nn=True, include_shift=False, deeper_pls=False, oo=True, NetClass=PretrainedResnetFrozen)


programEnd = time.time()

print(f"Whole program finished! It took {str(datetime.timedelta(seconds=programEnd-programStart))} hours:min:seconds")
print("done!")
