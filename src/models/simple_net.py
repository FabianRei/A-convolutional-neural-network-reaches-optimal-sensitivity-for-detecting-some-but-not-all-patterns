import torch.nn as nn
import torch.nn.functional as F


class SimpleNet(nn.Module):
    def __init__(self, dimIn=12*12*12*6, dimOut=4):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(dimIn, 200)
        self.fc2 = nn.Linear(200, dimOut)

    def forward(self, x):
        x = F.sigmoid(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class RobustNet(nn.Module):
    def __init__(self, dimIn=12*12*12*6, dimOut=4, dropout=0.2):
        super(RobustNet, self).__init__()
        self.drop1 = nn.Dropout(p=dropout)
        self.fc1 = nn.Linear(dimIn, 200)
        self.fc2 = nn.Linear(200, dimOut)

    def forward(self, x):
        x = self.drop1(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
