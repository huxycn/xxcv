import torch
import torch.nn as nn
import torch.nn.functional as F

from yolo.backbone import resnet


class detnet_bottleneck(nn.Module):
    # no expansion
    # dilation = 2
    # type B use 1x1 conv
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, block_type='A'):
        super(detnet_bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=2, bias=False,dilation=2)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes or block_type=='B':
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.downsample(x)
        out = F.relu(out)
        return out


class YOLO(nn.Module):
    def __init__(self, backbone, pretrained=True, feat_channels=2048, S=7, B=2, C=20):
        super().__init__()
        if 'resnet' in backbone:
            self.backbone = getattr(resnet, backbone)(pretrained=pretrained)
        self.layer5 = self._make_detnet_layer(in_channels=feat_channels)
        end_channels = B * 5 + C
        self.conv_end = nn.Conv2d(256, end_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_end = nn.BatchNorm2d(end_channels)

    def forward(self, x):
        x = self.backbone(x)
        x = self.layer5(x)
        x = self.conv_end(x)
        x = self.bn_end(x)
        x = torch.sigmoid(x)
        x = x.permute(0,2,3,1) #(-1,7,7,30)
        return x

    def _make_detnet_layer(self,in_channels):
        layers = []
        layers.append(detnet_bottleneck(in_planes=in_channels, planes=256, block_type='B'))
        layers.append(detnet_bottleneck(in_planes=256, planes=256, block_type='A'))
        layers.append(detnet_bottleneck(in_planes=256, planes=256, block_type='A'))
        return nn.Sequential(*layers)
