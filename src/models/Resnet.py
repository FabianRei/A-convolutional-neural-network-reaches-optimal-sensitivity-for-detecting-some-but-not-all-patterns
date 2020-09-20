from torchvision import models
from torch import nn
import torch.nn.functional as F
from fnmatch import fnmatch
from torch import tensor
import torch
import numpy as np


class PretrainedResnetFrozen(nn.Module):
    def __init__(self, dim_out, min_norm=np.float64(0), max_norm=np.float64(1), mean_norm=np.float64(0),
                 std_norm=np.float64(1), freeze_until=4):
        super().__init__()
        self.min_norm = torch.as_tensor(min_norm.astype(np.float)).cuda()
        self.max_norm = torch.as_tensor(max_norm.astype(np.float)).cuda()
        self.mean_norm = torch.as_tensor(mean_norm.astype(np.float)).cuda()
        self.std_norm = torch.as_tensor(std_norm.astype(np.float)).cuda()
        self.channel_mean = tensor([0.485, 0.456, 0.406]).cuda().reshape(1, -1, 1, 1)
        self.channel_std = tensor([0.229, 0.224, 0.225]).cuda().reshape(1, -1, 1, 1)
        self.freeze_until = freeze_until
        self.ResNet = models.resnet18(pretrained=True)
        # pytorch's standard implementation throws errors at some image sizes..
        self.ResNet.avgpool = nn.AdaptiveAvgPool2d(1)
        self.ResNet.fc = nn.Linear(512, dim_out)
        self.freeze_net()

    def forward(self, x):
        """
        We are using pretrained weights here. So the best way to use them, is to multiply the grayscale
        image to rbg. After that, it is best to normalize the image the way, the images were normalized.
        To do so, we first squeeze the values between 0 and 1 and then transform the way, the resnet images
        were transformed.
        Normally, this functionality is put into the dataloader, but I want to keep the dataloader general.

        :param x:
        :return:
        """
        # squeeze values between 0 and 1. Use values from test_data instead of small batch to decrease statistical variance
        # If you think about the math, this does the trick (max_norm and min_norm are from the pre-normalized distribution)
        x = x*(self.std_norm/(self.max_norm-self.min_norm)) + (self.mean_norm-self.min_norm)/(self.max_norm - self.min_norm)
        # copy to 3 channels
        x = x.repeat(1, 3, 1, 1)
        # substract imagenet mean and scale imagenet std
        x -= self.channel_mean
        x /= self.channel_std
        x = self.ResNet(x)
        return F.log_softmax(x)

    def freeze_net(self, freeze_until_val=-1):
        if freeze_until_val == -1:
            freeze_until_val = self.freeze_until
        for name, param in self.named_parameters():
            if fnmatch(name, f"*.layer[{freeze_until_val+1}-5]*") or fnmatch(name, '*.fc.*'):
                param.requires_grad = True
            else:
                if fnmatch(name, "ResNet.[b-c]*") and freeze_until_val == 0:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        pass

    def freeze_except_fc(self):
        for name, param in self.named_parameters():
            if fnmatch(name, '*fc.*'):
                param.requires_grad = True
            else:
                param.requires_grad = False

    def unfreeze_all(self):
        for param in self.parameters():
            param.requires_grad = True


class NotPretrainedResnet(nn.Module):
    def __init__(self, dim_out, min_norm=np.float64(1), max_norm=np.float64(1), mean_norm=np.float64(1), std_norm=np.float64(1)):
        super().__init__()
        self.min_norm = torch.as_tensor(min_norm.astype(np.float)).cuda()
        self.max_norm = torch.as_tensor(max_norm.astype(np.float)).cuda()
        self.mean_norm = torch.as_tensor(mean_norm.astype(np.float)).cuda()
        self.std_norm = torch.as_tensor(std_norm.astype(np.float)).cuda()
        self.channel_mean = tensor([0.485, 0.456, 0.406]).cuda().reshape(1, -1, 1, 1)
        self.channel_std = tensor([0.229, 0.224, 0.225]).cuda().reshape(1, -1, 1, 1)
        self.ResNet = models.resnet18(pretrained=False)
        # pytorch's standard implementation throws errors at some image sizes..
        self.ResNet.avgpool = nn.AdaptiveAvgPool2d(1)
        self.ResNet.fc = nn.Linear(512, dim_out)
        # self.freeze_except_fc()

    def forward(self, x):
        """
        We are using pretrained weights here. So the best way to use them, is to multiply the grayscale
        image to rbg. After that, it is best to normalize the image the way, the images were normalized.
        To do so, we first squeeze the values between 0 and 1 and then transform the way, the resnet images
        were transformed.
        Normally, this functionality is put into the dataloader, but I want to keep the dataloader general.

        :param x:
        :return:
        """
        # squeeze values between 0 and 1. Use values from test_data instead of small batch to decrease statistical variance
        # If you think about the math, this does the trick (max_norm and min_norm are from the pre-normalized distribution)
        # x = x*(self.std_norm/(self.max_norm-self.min_norm)) + (self.mean_norm-self.min_norm)/(self.max_norm - self.min_norm)
        # copy to 3 channels if not already there
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        # substract imagenet mean and scale imagenet std
        # x -= self.channel_mean
        # x /= self.channel_std
        x = self.ResNet(x)
        return F.log_softmax(x, dim=1)

    def freeze_except_fc(self):
        for name, param in self.named_parameters():
            if fnmatch(name, '*fc.*'):
                param.requires_grad = True
            else:
                param.requires_grad = False

    def unfreeze_all(self):
        for param in self.parameters():
            param.requires_grad = True


if __name__ == "__main__":
    Net = PretrainedResnetFrozen(2)
    for name, param in Net.named_parameters():
        if fnmatch(name, "ResNet.[b-c]*"):
            print(name)
    for i in range(1,6):
        print("######################################")
        for name, param in Net.named_parameters():
            if fnmatch(name, f"*.layer[{i}-5]*") or fnmatch(name, '*.fc.*'):
                print(name)
