import torch
from torch import nn
from torchvision.models import resnet50
from torchvision.models.detection import KeypointRCNN
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.backbone_utils import _resnet_fpn_extractor
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

class AutomotiveKeypointDetector(KeypointRCNN):

    def __init__(self, kh_depth=4, dilation=2, num_classes=2, num_keypoints=20):
        anchor_generator = AnchorGenerator(sizes=(32, 64, 128, 256, 512), aspect_ratios=(0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0))
        backbone = resnet50(progress=True, norm_layer=nn.BatchNorm2d)
        backbone = _resnet_fpn_extractor(backbone, 3)
        self.add_dilation_to_backbone(backbone, dilation)
        box_predictor = FastRCNNPredictor(1024, num_classes)
        keypoint_head = KeypointHead(backbone.out_channels, tuple(512 for _ in range(kh_depth)))
        super().__init__(backbone,
                            num_keypoints=num_keypoints,
                            anchor_generator=anchor_generator,
                            box_predictor=box_predictor,
                            keypoint_head=keypoint_head)

    def add_dilation_to_backbone(self, backbone, dilation):
        for name, module in backbone.named_children():
            if 'layer4' in name:  # Modifica solo layer4 per questo esempio
                for sub_name, sub_module in module.named_children():
                    if isinstance(sub_module, nn.Conv2d):
                        sub_module.dilation = (dilation, dilation)
                        sub_module.padding = (dilation, dilation)
                    elif isinstance(sub_module, nn.Sequential):
                        for nn_name, nn_module in sub_module.named_children():
                            if isinstance(nn_module, nn.Conv2d):
                                nn_module.dilation = (dilation, dilation)
                                nn_module.padding = (dilation, dilation)


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