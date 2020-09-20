import numpy as np


def flipEncodedAxes(arr, randomAxis):
    # x, y, flipped == z flipped -> 3
    # x, z flipped == y flipped -> 2
    # y, z flipped == x flipped -> 1
    # x, y, z flipped == no flip -> 0
    if randomAxis:
        target = np.random.randint(4)
    else:
        target = 0
    if target == 1:
        arr[:, :, :, [1, 2]] = -arr[:, :, :, [1, 2]]
    if target == 2:
        arr[:, :, :, [1]] = -arr[:, :, :, [1]]
    if target == 3:
        arr[:, :, :, [2]] = -arr[:, :, :, [2]]
    return arr, target


def flipAxes(arr, randomAxis=False, specificAxis=None):
    # x, y, flipped == z flipped -> 3
    # x, z flipped == y flipped -> 2
    # y, z flipped == x flipped -> 1
    # x, y, z flipped == no flip -> 0
    if randomAxis:
        target = np.random.randint(4)
    elif specificAxis is not None:
        target = specificAxis
    else:
        target = 0
    if target == 1:
        arr[:, :, :, [1, 3]] = -arr[:, :, :, [1, 3]]
    if target == 2:
        arr[:, :, :, [1, 4]] = -arr[:, :, :, [1, 4]]
    if target == 3:
        arr[:, :, :, [3, 4]] = -arr[:, :, :, [3, 4]]
    return arr, target


def switchAxes(arr, randomAxis):
    # xyz = 0, yxz = 1, zxy = 2, xzy = 3, yzx = 4, zyx = 5
    if randomAxis:
        target = np.random.randint(6)
    else:
        target = 0
    if target == 1:
        arr[:,:,:] = arr[:,:,:, [2, 3, 0, 1, 4, 5]]
    if target == 2:
        arr[:,:,:] = arr[:,:,:, [4, 5, 0, 1, 2, 3]]
    if target == 3:
        arr[:,:,:] = arr[:,:,:, [0, 1, 4, 5, 2, 3]]
    if target == 4:
        arr[:,:,:] = arr[:,:,:, [2, 3, 4, 5, 0, 1]]
    if target == 5:
        arr[:,:,:] = arr[:,:,:, [4, 5, 2, 3, 0, 1]]
    return arr, target
