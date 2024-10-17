
import torch.nn as nn
import torch.nn.functional as F
import torch

# Basic residual block, written by Kuangliu (https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py)
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()

        # Convolution 1
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        # Convolution 2
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        # Residual connection: by default identity
        self.shortcut = nn.Sequential()

        # Two cases when the convolution output does not match the initial C [=in-planes]
        # + the first convolution shrinks the filter size (stride != 1)
        # + initial C != intended output C (planes)
        # -> insert a convolution block that generates proper H,W and C
        # 
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


print(repr(BasicBlock(64, 64)))
