import scipy.io as sio
import numpy as np
import h5py
import torch
from skimage.util import view_as_blocks
from src.data.create_complex_pattern import create_automaton


def get_mat_data(pathMat, shuffle=False):
    matData = sio.loadmat(pathMat)
    data = np.transpose(matData['imgNoise'], (2, 0, 1))
    labels = matData['imgNoiseLabels'].squeeze()
    meanData = np.transpose(matData['meanImg'], (2, 0, 1))
    meanDataLabels = matData['meanImgLabels'].squeeze()
    if shuffle:
        np.random.seed(42)
        selector = np.random.permutation(labels.shape[0])
        data = data[selector]
        labels = labels[selector]
    return data, labels, meanData, meanDataLabels


def get_h5data(pathMat, shuffle=False):
    h5Data = h5py.File(pathMat)
    h5Dict = {k:np.array(h5Data[k]) for k in h5Data.keys()}
    data = h5Dict['imgNoise']
    labels = h5Dict['imgNoiseFreqs']
    meanData = h5Dict['noNoiseImg']
    meanDataLabels = h5Dict['noNoiseImgFreq']
    if shuffle:
        np.random.seed(42)
        selector = np.random.permutation(labels.shape[0])
        data = data[selector]
        labels = labels[selector]
    return data, labels, meanData, meanDataLabels


def get_h5mean_data(pathMat, includeContrast=False, includeShift=False, includeAngle=False, them_cones=False,
                    separate_rgb=False, meanData_rounding=None, shuffled_pixels=False, shuffle_scope=-1,
                    shuffle_portion=-1, ca_rule=-1):

    """
    Extract mean data template from the .h5 file. Additional tools allow shuffling.
    :param pathMat: path to .h5 data
    :param includeContrast:
    :param includeShift:
    :param includeAngle:
    :param them_cones:
    :param separate_rgb:
    :param meanData_rounding:
    :param shuffled_pixels:
    :param shuffle_scope:
    :param shuffle_portion:
    :param ca_rule: allows to create a cellular automaton here. Not used in paper
    :return:
    """
    h5Data = h5py.File(pathMat)
    h5Dict = {k:np.array(h5Data[k]) for k in h5Data.keys()}
    args = []
    # used to process cone data
    if them_cones:
        # 2 = red, 3 = green, 4 = blue
        mosaic = h5Dict['mosaicPattern']
        img_data = h5Dict['excitationsData']
        if len(img_data.shape) == 4:
            # quickfix different h5 files... (to be deleted)
            if img_data.shape[-1] == 1:
                img_data = np.squeeze(img_data)
            else:
                img_data = np.transpose(img_data, (3, 1, 2, 0))
        else:
            img_data = np.transpose(img_data, (2, 1, 0))
        if separate_rgb:
            if len(img_data.shape) == 4:
                r = np.copy(img_data)
                r[:, mosaic != 2, :] = 0
                g = np.copy(img_data)
                g[:, mosaic != 3, :] = 0
                b = np.copy(img_data)
                b[:, mosaic != 4, :] = 0
                stack_list = []
                for i in range(img_data.shape[3]):
                    stack_list.append(r[..., i])
                    stack_list.append(g[..., i])
                    stack_list.append(b[..., i])
                img_data = np.stack(stack_list, axis=-1)
            else:
                r = np.copy(img_data)
                r[:, mosaic != 2] = 0
                g = np.copy(img_data)
                g[:, mosaic != 3] = 0
                b = np.copy(img_data)
                b[:, mosaic != 4] = 0
                img_data = np.stack((r, g, b), axis=-1)
        # img_data = np.transpose(img_data, (3, 1, 2, 0))
        args.append(img_data)
        args.append(h5Dict['spatialFrequencyCyclesPerDeg'])
        if includeContrast:
            args.append(h5Dict['contrast'])
        if includeShift:
            args.append(h5Dict['shift'])
        if includeAngle:
            args.append(h5Dict['rotation'])
    # used in paper for monochrome sensor signals
    else:
        experiment = h5Dict['noNoiseImg']
        # adjust into correct formatting
        experiment = np.transpose(experiment, (0, 2, 1))
        if ca_rule != -1:
            automaton = create_automaton(rule=ca_rule, size=experiment.shape[1:], seed=1337)
            pure_signal = experiment[1] - experiment[0]
            automaton_signal = pure_signal*automaton
            experiment[1] = automaton_signal + experiment[0]

        # experiment += 0.567891011121314
        if meanData_rounding is not None:
            print(f"Rounding mean_data to {meanData_rounding} decimals..")
            experiment = np.round(experiment, meanData_rounding)
        # experiment -= 0.567891011121314
        # shuffle pixels to create alternative signals. Only some of this functionality is used in the paper
        if shuffled_pixels > 0:
            experiment = shuffle_pixels(experiment, shuffled_pixels, shuffle_scope, shuffle_portion)
        elif shuffled_pixels < 0:
            experiment = shuffle_1d(experiment, shuffled_pixels)
        args.append(experiment)
        #####################
        # args.append(h5Dict['noNoiseImg'])
        args.append(h5Dict['noNoiseImgFreq'])
        if includeContrast:
            args.append(h5Dict['noNoiseImgContrast'])
        if includeShift:
            args.append(h5Dict['noNoiseImgPhase'])
        if includeAngle:
            args.append(h5Dict['noNoiseImgAngle'])
    return args


def shuffle_1d(matrices, dimension):
    # 1st dimension -> shuffle rows, 2nd dimension -> shuffle columns
    print('db')
    np.random.seed(42)
    dim_idx = dimension*-1
    shuff_idxs = np.random.permutation(matrices.shape[dim_idx])
    matrices = np.take(matrices, shuff_idxs, axis=dim_idx)
    return matrices


def shuffle_pixels(matrices, block_size, shuffle_scope, shuffle_portion):
    """
    Shuffle pixels of the signal. Used to assess CNN performance on various pixel localizations
    :param matrices: signal matrices
    :param block_size: Shuffle blocks of the image of specific size
    :param shuffle_scope: only shuffle (coherent) part of the image
    :param shuffle_portion: shuffle a certain percentage of pixels
    :return:
    """
    block_size = int(block_size)
    np.random.seed(42)
    original_width = matrices.shape[-1]
    original_height = matrices.shape[-2]
    start_width = original_width // 2 - shuffle_scope // 2
    start_height = original_height // 2 - shuffle_scope // 2
    if not shuffle_scope == -1:
        original_matrices = matrices.copy()
        matrices = matrices[:, start_height:start_height+shuffle_scope, start_width:start_width+shuffle_scope]
    rows = matrices.shape[-2]
    cols = matrices.shape[-1]
    shuffle_rows = rows // block_size
    shuffle_cols = cols // block_size
    padding_rows = rows % block_size
    padding_cols = cols % block_size
    shuff_args = np.random.permutation(np.arange(shuffle_rows * shuffle_cols))
    if not shuffle_portion == -1:
        shuff_idxs_change_points = np.random.permutation(np.arange(shuffle_portion))
    result = []
    for md in matrices:
        # remove padding
        pad_up = np.ceil(padding_rows/2).astype(np.int)
        pad_down = np.floor(padding_rows/2).astype(np.int)
        pad_left = np.ceil(padding_cols/2).astype(np.int)
        pad_right = np.floor(padding_cols/2).astype(np.int)
        pad_mat_up = md[:pad_up]
        md = md[pad_up:]
        if pad_down == 0:
            pad_mat_down = md[:0]
        else:
            pad_mat_down = md[-pad_down:]
            md = md[:-pad_down]
        pad_mat_left = md[:, :pad_left]
        md = md[:, pad_left:]
        if pad_down == 0:
            pad_mat_right = md[:, :0]
        else:
            pad_mat_right = md[:, -pad_right:]
            md = md[:, :-pad_right]
        md = view_as_blocks(md, (block_size, block_size))
        height = md.shape[0]
        width = md.shape[1]
        if not shuffle_portion == -1:
            points_to_change = shuff_args[:shuffle_portion]
            change_points = md.reshape(-1, block_size, block_size)[points_to_change]
            # simply shuffling shuffles each individual md slice differently and messes everything up.
            change_points = change_points[shuff_idxs_change_points]
            md.reshape(-1, block_size, block_size)[points_to_change] = change_points
            res = md
        else:
            res = md.reshape(-1, block_size, block_size)[shuff_args].reshape(height, width, block_size, block_size)
        res = res.transpose(0, 2, 1, 3).reshape(-1, res.shape[1] * res.shape[3])

        # reassemble
        res = np.block([[pad_mat_up], [pad_mat_left, res, pad_mat_right], [pad_mat_down]])
        result.append(res)
    matrices = np.stack(result)
    if not shuffle_scope == -1:
        original_matrices[:, start_height:start_height+shuffle_scope, start_width:start_width+shuffle_scope] = matrices
        matrices = original_matrices
    return matrices


def mat_data_loader(data, labels, batchSize, shuffle=True):
    epochData = data.clone()
    epochLabels = labels.clone()
    if shuffle:
        selector = np.random.permutation(len(epochData))
        epochData = epochData[selector]
        epochLabels = epochLabels[selector]
    sz = epochData.shape[0]
    i, j = 0, batchSize
    while j < sz + batchSize:
        batchData = epochData[i:min(j, sz)]
        batchLabels = epochLabels[i:min(j, sz)]
        yield batchData, batchLabels
        i += batchSize
        j += batchSize


def poisson_noise_loader(meanData, size, numpyData=False, seed=-1, force_balance=False, signal_no_signal=True):
    if seed != -1:
        if numpyData:
            np.random.seed(seed)
        else:
            torch.random.manual_seed(seed)
    if numpyData:
        # more data means all data after #0 are signal cases
        data = []
        if len(meanData) == 2:
            if force_balance:
                labels = np.random.permutation(np.concatenate((np.ones(size//2), np.zeros(size-size//2)))).astype(int)
            else:
                labels = np.random.randint(2, size=size)
            for l in labels:
                if l == 0:
                    data.append(np.random.poisson(meanData[l]))
                else:
                    selector = np.random.randint(1, len(meanData))
                    data.append(np.random.poisson(meanData[selector]))
        elif signal_no_signal and len(meanData) > 2:
            labels = np.random.randint(2, size=size)
            for l in labels:
                if l == 0:
                    data.append(np.random.poisson(meanData[l]))
                else:
                    selector = np.random.randint(1, len(meanData))
                    data.append(np.random.poisson(meanData[selector]))
        else:
            # balances as good as possible, given required size. We balance classes perfectly to achieve theoretically
            # optimal ideal observer performance
            if force_balance:
                class_size = size//len(meanData)
                labels = np.random.permutation(np.repeat(np.array(range(len(meanData))), class_size)).astype(int)
                rest_size = size-(class_size*len(meanData))
                rest_labels = np.random.randint(len(meanData), size=rest_size)
                labels = np.append(labels, rest_labels)
                print('todo')
            else:
                labels = np.random.randint(len(meanData), size=size)
            for l in labels:
                data.append(np.random.poisson(meanData[l]))
        data = np.stack(data)
        return data, labels
    else:
        data = []
        if len(meanData) > 2 and signal_no_signal:
            labels = torch.randint(2, (size,)).type(torch.long)
            for l in labels:
                if l==0:
                    data.append(torch.poisson(meanData[l]))
                else:
                    selector = np.random.randint(1, len(meanData))
                    data.append(torch.poisson(meanData[selector]))
        else:
            labels = torch.randint(len(meanData), (size,)).type(torch.long)
            for l in labels:
                data.append(torch.poisson(meanData[l]))
        data = torch.stack(data)
        return data, labels


class PoissonNoiseLoaderClass:
    def __init__(self, mean_data, batch_size, train_set_size=-1, numpy_data=False, use_data_seed=False, data_seed=12,
                 signal_no_signal=False):
        if type(mean_data) == np.ndarray:
            mean_data = torch.from_numpy(mean_data).type(torch.float32)
        elif mean_data.is_cuda:
            mean_data = mean_data.cpu()
        self.mean_data = mean_data
        self.train_set_size = train_set_size
        self.numpy_data = numpy_data
        self.use_data_seed = use_data_seed
        self.data_seed = data_seed
        self.dataset, self.labels = self.get_data()
        self.ii, self.jj = -1, -1
        self.batch_size = batch_size
        self.original_sorting = np.array(range(train_set_size))
        self.signal_no_signal = signal_no_signal
        if self.use_data_seed:
            if self.data_seed != -1:
                if self.numpy_data:
                    np.random.seed(self.data_seed)
                else:
                    torch.random.manual_seed(self.data_seed)

    def get_data(self):
        if self.train_set_size == -1:
            dataset = -1
            labels = -1

        else:
            dataset, labels = self.poisson_noise_loader(self.train_set_size, self.numpy_data, self.data_seed)
        return dataset, labels

    def get_batches(self, batch_size=-1, shuffle=True, reset=False):
        if batch_size == -1:
            batch_size = self.batch_size
        if self.train_set_size == -1:
            # using a seed here would be problematic, as all batches would be the same
            batch_data, batch_labels = self.poisson_noise_loader(batch_size, self.numpy_data)
        else:
            if not self.jj < self.train_set_size or self.jj == -1 or reset:
                if shuffle:
                    selector = np.random.permutation(len(self.dataset))
                    self.dataset = self.dataset[selector]
                    self.labels = self.labels[selector]
                    self.original_sorting = self.original_sorting[selector]
                self.ii, self.jj = 0, batch_size
            batch_data = self.dataset[self.ii:min(self.jj, self.train_set_size)]
            batch_labels = self.labels[self.ii:min(self.jj, self.train_set_size)]
            self.ii += batch_size
            self.jj += batch_size
        batch_data = batch_data.cuda()
        batch_labels = batch_labels.cuda()
        return batch_data, batch_labels

    def poisson_noise_loader(self, size, numpyData=False, seed=-1):
        if seed != -1:
            if numpyData:
                np.random.seed(self.data_seed)
            else:
                torch.random.manual_seed(self.data_seed)
        if numpyData:
            # more data means all data after #0 are signal cases
            data = []
            if len(self.mean_data) > 2 and self.signal_no_signal:
                labels = np.random.randint(2, size=size)
                for l in labels:
                    if l == 0:
                        data.append(np.random.poisson(self.mean_data[l]))
                    else:
                        selector = np.random.randint(1, len(self.mean_data))
                        data.append(np.random.poisson(self.mean_data[selector]))
            else:
                labels = np.random.randint(len(self.mean_data), size=size)
                for l in labels:
                    data.append(np.random.poisson(self.mean_data[l]))
            data = np.stack(data)
            return data, labels
        else:
            data = []
            if len(self.mean_data) > 2 and self.signal_no_signal:
                labels = torch.randint(2, (size,)).type(torch.long)
                for l in labels:
                    if l == 0:
                        data.append(torch.poisson(self.mean_data[l]))
                    else:
                        selector = np.random.randint(1, len(self.mean_data))
                        data.append(torch.poisson(self.mean_data[selector]))
            else:
                labels = torch.randint(len(self.mean_data), (size,)).type(torch.long)
                for l in labels:
                    data.append(torch.poisson(self.mean_data[l]))
            data = torch.stack(data)
            return data, labels


if __name__ == '__main__':
    # pathMat = "/black/localhome/reith/Desktop/projects/WLDiscriminationNetwork/deepLearning/data/mat_files/10000_samplesPerClass_freq_1_contrast_0_001.h5"
    # pathMat = r"C:\Users\Fabian\stanford\optimal_networks\local\automata_contrasts\face_1\2_samplesPerClass_freq_1_contrast_0_019952623150_image_face_1.h5"
    # pathMat = r'C:\Users\Fabian\Documents\data\windows2rsync\windows_data\disks\circle_with_radius_27\2_samplesPerClass_freq_1_contrast_0_019952623150_image_circle_with_radius_27.h5'
    # pathMat = r'C:\Users\Fabian\Documents\data\windows2rsync\windows_data\face_experiment\single_faces\bald_guy\2_samplesPerClass_freq_1_contrast_0_019952623150_image_bald_guy.h5'
    # pathMat = r'C:\Users\Fabian\stanford\optimal_networks\local\faces_contrasts\face_1\2_samplesPerClass_freq_1_contrast_0_019952623150_image_face_1.h5'
    # pathMat = r'C:\Users\Fabian\stanford\optimal_networks\local\multiple_locations_harmonic_frequency_of_1_loc_4_signalGridSize_3\1_samplesPerClass_freq_1_contrast_0_100475457260_loc_4_signalGrid_3.h5'
    # pathMat = r'C:\Users\Fabian\stanford\optimal_networks\local\multiple_locations_harmonic_frequency_of_1_loc_4_signalGridSize_3\1_samplesPerClass_freq_1_contrast_0_798104925988_loc_4_signalGrid_3.h5'
    # meanData, meanDataLabels = get_h5mean_data(pathMat)
    from matplotlib import pyplot as plt
    plt.imshow(meanData[2])
    plt.show()
    data, labels = poisson_noise_loader(meanData, 16)
    print("done")
