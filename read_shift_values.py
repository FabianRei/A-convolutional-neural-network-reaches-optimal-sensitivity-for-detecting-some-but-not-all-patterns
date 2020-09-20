from glob import glob
from deepLearning.src.data.mat_data import get_h5mean_data
import os

pathMatDir = '/share/wandell/data/reith/matlabData/shift_contrast100/'

matFiles = glob(f'{pathMatDir}*.h5')
matFiles.sort()
for f in matFiles:
    meanData, meanDataLabels, dataContrast, dataShift = get_h5mean_data(f, includeContrast=True, includeShift=True)
    with open(os.path.join(pathMatDir, "shiftVals.txt"), 'a') as txt:
        txt.write(str(dataShift[1])+'\n')
