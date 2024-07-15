import torch
from torch import nn


class KeypointRCNNHeads(nn.Sequential):
    def __init__(self, in_channels, layers):
        d = []
        next_feature = in_channels
        for out_channels in layers:
            d.append(nn.Conv2d(next_feature, out_channels, 3, stride=1, padding=1))
            d.append(nn.ReLU(inplace=True))
            next_feature = out_channels
        super().__init__(*d)
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                nn.init.constant_(m.bias, 0)


class KeypointHead(nn.Sequential):
    def __init__(self, in_channels, layers):
        d = []
        next_feature = in_channels
        for out_channels in layers:
            d.append(ResidualBlock(next_feature, out_channels))
            next_feature = out_channels
        super().__init__(*d)


class ResidualBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, padding=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.inplanes = inplanes
        self.planes = planes
        self.stride = stride
        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride, padding)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride, padding)
        self.fixDim = nn.Conv2d(inplanes, planes, 1, stride)
        self.norm = nn.BatchNorm2d(self.planes)
        self.relu = nn.ReLU(inplace=True)
        self.assign_weights_and_bias()

    def assign_weights_and_bias(self):
        nn.init.kaiming_normal_(self.conv1.weight, mode="fan_out", nonlinearity="relu")
        nn.init.kaiming_normal_(self.conv2.weight, mode="fan_out", nonlinearity="relu")
        nn.init.constant_(self.conv1.bias, 0)
        nn.init.constant_(self.conv2.bias, 0)

    def F(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.conv1(x)  ## first convolution
        out = self.norm(out)  ## normalization
        out = self.relu(out)  ## activation function
        out = self.conv2(out)  ## second convolution
        out = self.norm(out)  ## normalization
        return out

    def G(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        ##if dimensions are different apply fixDim
        if self.inplanes != self.planes or self.stride > 1:
            out = self.fixDim(out)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.F(x) + self.G(x))