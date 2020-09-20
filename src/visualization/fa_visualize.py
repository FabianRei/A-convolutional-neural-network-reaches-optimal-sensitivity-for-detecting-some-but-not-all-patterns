from preprocess import getTensorList_general
from dipy.reconst.dti import fractional_anisotropy
import numpy as np
from skimage.measure import block_reduce
import pickle
from PIL import Image

def six2mat(voxel6):
    return voxel6[[0,3,4,3,1,5,4,5,2]].reshape(3,3)


def getFAfromSix(sixArr, axis=-1):
    mat = six2mat(sixArr)
    eig = np.linalg.eig(mat)[0]
    fa = fractional_anisotropy(eig)
    return fa


tensorDir = '/home/reith/PycharmProjects/axisFlipDetector/data/tensors/'
# tensors = getTensorList_general(tensorDir)
tensorListFa = pickle.load(open("tensorsFullFa.b", "rb"))
maxIndices = []
maxValues = []
for t in tensorListFa:
    argmaxIndex = np.unravel_index(np.argmax(t), t.shape)
    maxVal = t[argmaxIndex]
    maxIndices.append(argmaxIndex)
    maxValues.append(maxVal)


for k in range(len(maxValues)):
    x, y, z = maxIndices[k]
    image = tensorListFa[k][:,y,:]
    image = (image / np.max(image))*255
    img = Image.fromarray(image)
    img.convert('L').save(f"image_{k}_white.bmp")
    image[x - 2:x + 2, z - 2:z + 2] = 0
    img = Image.fromarray(image)
    img.convert('L').save(f"image_{k}_black.bmp")
    print(image.shape)


