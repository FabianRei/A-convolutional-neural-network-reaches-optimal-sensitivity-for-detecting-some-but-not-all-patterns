from deepLearning.src.models.Resnet import PretrainedResnet
from fnmatch import fnmatch

Net = PretrainedResnet(dim_out=2)

for name, param in Net.named_parameters():
    if fnmatch(name, '*fc.*'):
        param.requires_grad = True
    else:
        param.requires_grad = False

for name, param in Net.named_parameters():
    if param.requires_grad:
        print(name)

# print(Net)
print("done")
