
import torch.nn as nn
import torch.nn.functional as F
import torch

# Basic residual block, written by Kuangliu (https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py)
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()

        # Convolution 1
        # VGSL: 
        # Cr3,3,<out_channels>  Gn<out_channels>
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        # Convolution 2
        # VGSL:
        # Cr3,3,<out_channels> Gn<out_channels>
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Residual connection: by default identity
        # VGSL:
        # (I ... )
        self.shortcut = nn.Sequential()

        # Two cases when the convolution output does not match the initial C [=in-planes]
        # + the first convolution shrinks the filter size (stride != 1)
        # + initial C != intended output C (planes)
        # -> insert a convolution block that generates proper H,W and C
        # 
        if stride != 1 or in_channels != self.expansion*out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

# VGSL BasicBlock
# Addtion pattern examples: A2,4 where 2 is the dimension to add and 4 is the chunk size
#
# '[1,5,6,3 ([I Cl1,1,64 Gn64] [Cr3,3,32 Gn32 Cl3,3,64 Gn64]) A3,64 Cr1,1,64]'
#
# Pattern: R64

class BasicBlockLayer(nn.Module):

    """A wrapper for a residual block.
    """
    def __init__(self, 
                 in_channels: int,
                 out_channels: int,
                 stride: int) -> None:
        super().__init_()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.bb = BasicBlock( in_channels, out_channels, stride )

    def forward( self,
                 inputs: torch.Tensor):
        self.bb( inputs )
