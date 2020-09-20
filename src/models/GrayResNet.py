import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class GrayResnet18(models.ResNet):
    def __init__(self, dimOut):
        super(GrayResnet18, self).__init__(models.resnet.BasicBlock, [2, 2, 2, 2], num_classes=dimOut)
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
        super(GrayResnet101, self).__init__(models.resnet.Bottleneck, [3, 4, 23, 3], num_classes=dimOut)
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


if __name__ == '__main__':
    test = GrayResnet18(2)
    print('nice')