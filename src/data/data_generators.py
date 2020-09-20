import numpy as np
from skimage.measure import block_reduce
import torch


def block_resize(tensorList, rF):
    newTensorList = []
    for t in tensorList:
        newTensorList.append(block_reduce(t, (rF,rF,rF,1)))
    return newTensorList


def shift_crop(tensorlist, maxShift=0, standardCrop=[slice(7, 55), slice(40, 88), slice(14, 62)]):
    # Tensor shape is (81, 106, 76, 6)
    newTensorlist = []
    for t in tensorlist:
        shift = np.random.randint(-maxShift, maxShift+1, size=3)
        newSlice = []
        for i, sl in enumerate(standardCrop):
            start = sl.start + shift[i]
            stop = sl.stop + shift[i]
            newSlice.append(slice(start,stop))
        newTensorlist.append(t[tuple(newSlice)])
    return newTensorlist


def dummy_loader(batchSize, numIn, numOut):
    for i in range(10):
        yield torch.rand([batchSize, numIn]), torch.zeros([batchSize, numOut])


def data_loader(tensorList, batchSize, shuffle=True, randomAxis=True, names=None, maxShift=None, resizeFactor=None, simulationFunction=None):
    if maxShift is not None:
        tensorList = shift_crop(tensorList, maxShift=maxShift)
    if resizeFactor is not None:
        tensorList = block_resize(tensorList, resizeFactor)
    tensorArr = torch.from_numpy(np.stack(tensorList, axis=0)).type(torch.float32)
    epochArr = torch.clone(tensorArr)
    if names is not None:
        epochNames = names
    if shuffle:
        selector = np.random.permutation(len(epochArr))
        epochArr = epochArr[selector]
        if names is not None:
            epochNames = [epochNames[i] for i in selector]
    sz = epochArr.shape[0]
    i, j = 0, batchSize
    while j < sz + batchSize:
        batch = epochArr[i:min(j, sz)]
        sz_batch = batch.shape[0]
        targets = []
        for k in range(sz_batch):
            batch[k], target = simulationFunction(batch[k], randomAxis)
            targets.append(target)
        targets = np.stack(targets, axis=0).astype(np.long)
        if names is not None:
            yield batch, torch.from_numpy(targets), epochNames[i:min(j, sz)]
        else:
            yield batch, torch.from_numpy(targets)
        i += batchSize
        j += batchSize
