from deepLearning.src.models.GrayResNet_skip_connections import GrayResnet18, GrayResnet101
from deepLearning.src.data.mat_data import get_h5mean_data, poisson_noise_loader
import torch

weights_path = '/share/wandell/data/reith/experiment_freq_1_log_contrasts30_resnet18/resNet_weights_5_samplesPerClass_freq_1_contrast_oo_0_181393069391.torch'
h5_path = '/share/wandell/data/reith/experiment_freq_1_log_contrasts30_resnet18/5_samplesPerClass_freq_1_contrast_oo_0_181393069391.h5'
Net = GrayResnet18(2)

Net.load_state_dict(torch.load(weights_path))
Net.cuda()
Net.eval()
meanData, meanDataLabels, dataContrast = get_h5mean_data(h5_path, includeContrast=True)
testDataFull, testLabelsFull = poisson_noise_loader(torch.tensor(meanData), size=64, numpyData=False)
dim_in = testDataFull.shape[-1]
testDataFull = testDataFull.view(-1, 1, dim_in, dim_in).cuda().float()
testDataFull -= testDataFull.mean()
testDataFull /= testDataFull.std()
out = Net(testDataFull)
print("Resnet101:")
weights_path = '/share/wandell/data/reith/experiment_freq_1_log_contrasts30_resnet101/resNet_weights_5_samplesPerClass_freq_1_contrast_oo_0_181393069391.torch'
h5_path = '/share/wandell/data/reith/experiment_freq_1_log_contrasts30_resnet101/5_samplesPerClass_freq_1_contrast_oo_0_181393069391.h5'
Net = GrayResnet101(2)

Net.load_state_dict(torch.load(weights_path))
Net.cuda()
Net.eval()
out = Net(testDataFull)
# for n, p in Net.named_parameters():
#     print(n)
print("nice!")
