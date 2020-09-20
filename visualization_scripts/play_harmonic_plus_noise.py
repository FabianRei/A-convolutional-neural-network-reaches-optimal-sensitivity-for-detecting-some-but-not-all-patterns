from matplotlib import pyplot as plt
from glob import glob
import h5py
import numpy as np

def get_mean_data(h5_path, shuffled_pixels=False):
    h5Data = h5py.File(h5_path)
    h5Dict = {k:np.array(h5Data[k]) for k in h5Data.keys()}
    result = h5Dict['noNoiseImg']
    result = np.transpose(result, (0, 2, 1))
    if shuffled_pixels:
        np.random.seed(42)
        rows = result.shape[-2]
        cols = result.shape[-1]
        shuff_args = np.random.permutation(np.arange(rows*cols))
        r = []
        for md in result:
            r.append(md.flatten()[shuff_args].reshape(rows,cols))
        result = np.stack(r)
        print("nice")
    return result

fpath = '/share/wandell/data/reith/coneMosaik/dummy_folder/'
h5s = glob(f'{fpath}*.h5')
h5s.sort()

h5 = h5s[-4]

sample = get_mean_data(h5)
poisson = np.random.poisson(sample)
poisson = poisson.astype(np.double)
poisson[0] = poisson[0]/poisson[0].max()
poisson[1] = poisson[1]/poisson[1].max()
plt.figure()
plt.title('normal pixel order, no signal')
plt.imshow(poisson[0], cmap='gray')
plt.show()

plt.figure()
plt.title('normal pixel order, signal')
plt.imshow(poisson[1], cmap='gray')
plt.show()
sample2 = get_mean_data(h5, shuffled_pixels=True)
poisson2 = np.random.poisson(sample2)
poisson2 = poisson2.astype(np.double)
poisson2[0] = poisson2[0]/poisson2[0].max()
poisson2[1] = poisson2[1]/poisson2[1].max()
plt.figure()
plt.title('randomized pixel order, no signal')
plt.imshow(poisson2[0], cmap='gray')
plt.show()

plt.figure()
plt.title('randomized pixel order, signal')
plt.imshow(poisson2[1], cmap='gray')
plt.show()
print('nice!')
