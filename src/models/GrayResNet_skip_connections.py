import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        # print(f"out: {out.abs().mean()} - residual: {residual.abs().mean()}")
        # print(f"difference is {(out-residual).abs().mean()}")
        out += residual
        out = self.relu(out)
        print(f"{residual.abs().mean()}, {residual.mean()} difference totally is {(out-residual).abs().mean()/(out.abs().mean()+residual.abs().mean())/2}")
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        print(f"{residual.abs().mean()}, {residual.mean()} difference totally is {(out-residual).abs().mean()/(out.abs().mean()+residual.abs().mean())/2}")
        return out


class GrayResnet18(models.ResNet):
    def __init__(self, dimOut):
        super(GrayResnet18, self).__init__(BasicBlock, [2, 2, 2, 2], num_classes=dimOut)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        # self.extraFc1 = nn.Linear(1000, 200)
        # self.drop1 = nn.Dropout(p=dropout)
        # self.extraFc2 = nn.Linear(200, dimOut)

    def forward(self, x):
        x = super(GrayResnet18, self).forward(x)
        # x = F.relu(self.extraFc1(x))
        # x = self.drop1(x)
        # x = self.extraFc2(x)
        return F.log_softmax(x, dim=1)


class GrayResnet101(models.ResNet):
    def __init__(self, dimOut):
        super(GrayResnet101, self).__init__(Bottleneck, [3, 4, 23, 3], num_classes=dimOut)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        # self.extraFc1 = nn.Linear(1000, 200)
        # self.drop1 = nn.Dropout(p=dropout)
        # self.extraFc2 = nn.Linear(200, dimOut)

    def forward(self, x):
        x = super(GrayResnet101, self).forward(x)
        # x = F.relu(self.extraFc1(x))
        # x = self.drop1(x)
        # x = self.extraFc2(x)
        return F.log_softmax(x, dim=1)
