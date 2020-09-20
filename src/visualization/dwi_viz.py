import os
from dipy.io.image import load_nifti, save_nifti
from dipy.io import read_bvals_bvecs
from dipy.core.gradients import gradient_table
from dipy.reconst.dti import TensorModel, color_fa
import numpy as np
import nibabel as nib
from dipy.reconst.dti import fractional_anisotropy
from PIL import Image



def six2mat(voxel6):
    return voxel6[[0, 1, 3, 1, 2, 4, 3, 4, 5]].reshape(3,3)


def saveAllImagesFromDWI(fpath, outputFolder):
    arr = nib.load(fpath).get_fdata().squeeze()
    matArr = np.apply_along_axis(six2mat, -1, arr)
    evalArr, evecArr = np.linalg.eig(matArr)
    faArr = fractional_anisotropy(evalArr)
    colorImg = color_fa(faArr, evecArr)
    fileName = os.path.basename(fpath).split('_')[0]
    outPath = os.path.join(outputFolder, fileName)
    os.makedirs(outPath, exist_ok=True)
    os.makedirs(os.path.join(outPath, "x"), exist_ok=True)
    for i in range(colorImg.shape[0]):
        img = Image.fromarray((colorImg[i,:,:] * 255).astype(np.uint8))
        img.save(os.path.join(outPath, "x", f"{fileName}_x_slice_{i}.png"))

    os.makedirs(os.path.join(outPath, "y"), exist_ok=True)
    for i in range(colorImg.shape[1]):
        img = Image.fromarray((colorImg[:,i,:] * 255).astype(np.uint8))
        img.save(os.path.join(outPath, "y", f"{fileName}_y_slice_{i}.png"))

    os.makedirs(os.path.join(outPath, "z"), exist_ok=True)
    for i in range(colorImg.shape[2]):
        img = Image.fromarray((colorImg[:,:,i] * 255).astype(np.uint8))
        img.save(os.path.join(outPath, "z", f"{fileName}_z_slice_{i}.png"))





fpath = '/home/reith/PycharmProjects/axisFlipDetector/data/tensors/5aac14517f233f00193eb257_tensors.nii.gz'
outputFolder = '/black/localhome/reith/Desktop/projects/Tensors/color_images_test/'

tensorPath = '/black/localhome/reith/Desktop/projects/Tensors/test/'
fpaths = [os.path.join(tensorPath , p) for p in os.listdir(tensorPath)]

for fpath in fpaths:
    saveAllImagesFromDWI(fpath, outputFolder)

print("done")











'''
fpath = '/home/reith/PycharmProjects/axisFlipDetector/dwi/'

fdwi =  os.path.join(fpath, 'file.nii.gz')
fbval = os.path.join(fpath, 'file.bval')
fbvec = os.path.join(fpath, 'file.bvec')



data, affine = load_nifti(fdwi)
bvals, bvecs = read_bvals_bvecs(fbval, fbvec)
gtab = gradient_table(bvals, bvecs)

tenmodel = TensorModel(gtab)
tenfit = tenmodel.fit(data)
img = Image.fromarray((test[:,:,30,:]*255).astype(np.uint8))
img.show()
save_nifti('colorfa.nii.gz', tenfit.color_fa, affine)
'''
