import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch
from deepLearning.src.models.Resnet import NotPretrainedResnet

def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class GrayVGG(models.VGG):
    def __init__(self, dimOut):
        super(GrayVGG, self).__init__(make_layers(cfg['D']), num_classes=dimOut)
        print("magic")


    def forward(self, x):
        x = super(GrayVGG, self).forward(x)
        # x = F.relu(self.extraFc1(x))
        # x = self.drop1(x)
        # x = self.extraFc2(x)
        return F.log_softmax(x, dim=1)



if __name__ == '__main__':
    import numpy as np

    net = GrayVGG(2)
    net2 = NotPretrainedResnet(2)
    net2.cuda()
    input = np.random.rand(1,3,224,224)
    input = torch.autograd.Variable(torch.tensor(input))
    input = input.type(torch.float32)
    text = net(input)
    print(text)
    print('nice')




'''
past attempt:

class VggNet19(nn.Module):
    def __init__(self, dimOut):
        super().__init__()
        self.net = models.vgg19(num_classes=dimOut)

    def forward(self, x):
        x = self.net(x)
        return x


if __name__ == '__main__':
    net = VggNet19(2)
    print("nice")
    input = np.random.rand(1,3,224,224)
    input = torch.autograd.Variable(torch.tensor(input).double())
    test = net(input)
    print("very nice!")

'''
