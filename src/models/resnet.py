"""
Adapted from <https://github.com/pytorch/vision/blob/master/torchvision/models/
resnet.py>.
"""

import torch
import torch.nn as nn
from torch.nn import init

from .custom_layers import SuppressContent


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """
    3x3 convolution with padding.
    """
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False,
                     dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """
    1x1 convolution.
    """
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                     bias=False)


class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ["downsample"]

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and "
                             "base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in "
                                      "BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when
        # stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4
    __constants__ = ["downsample"]

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when
        # stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, params, block, layers, zero_init_residual=False,
                 groups=1, width_per_group=64,
                 replace_stride_with_dilation=None, norm_layer=None):
        super(ResNet, self).__init__()

        self.clf_input = params.clf_input
        self.n_classes = params.n_classes
        self.ptc_fm = params.ptc_fm
        self.ptc_sz = params.ptc_sz
        self.drop_rate = params.drop_rate

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(
                                 replace_stride_with_dilation))

        self.groups = groups
        self.base_width = width_per_group

        # suppress the image content if clf_input != "rgb"
        if (self.clf_input != "rgb"):
            self.suppressor = SuppressContent(params)

        # concat low- / high-frequency if "+" in clf_input
        self.concat = False
        if "+" in self.dis_input:
            self.concat = True

        if self.concat:
            # concat low / high spatial frequency components
            if "fixed_hpf" in self.clf_input:
                self.ptc_fm += 1
            elif "finite_difference" in self.clf_input:
                self.ptc_fm += 6
            else:
                self.ptc_fm += 3
            self._forward_impl = self._forward_concat
        else:
            # low- / high-frequency
            if self.clf_input == "rgb":
                self.ptc_fm = 3
                self._forward_impl = self._forward_low
            else:
                if self.clf_input == "fixed_hpf":
                    self.ptc_fm = 1
                elif self.clf_input == "finite_difference":
                    self.ptc_fm = 6
                else:
                    self.ptc_fm = 3
                self._forward_impl = self._forward_high

        features = []
        if self.ptc_sz <= 32:
            # small inputs if images are 32x32, otherwise assumes images are
            # larger
            features += [nn.Conv2d(self.ptc_fm, self.inplanes, kernel_size=3,
                                   stride=1, padding=1, bias=False)]
        else:
            features += [nn.Conv2d(self.ptc_fm, self.inplanes, kernel_size=7,
                                   stride=2, padding=3, bias=False)]

        features += [norm_layer(self.inplanes)]
        features += [nn.ReLU(inplace=True)]
        features += [nn.MaxPool2d(kernel_size=3, stride=2, padding=1)]
        features += [self._make_layer(block, 64, layers[0])]
        features += [self._make_layer(block, 128, layers[1], stride=2,
                                      dilate=replace_stride_with_dilation[0])]
        features += [self._make_layer(block, 256, layers[2], stride=2,
                                      dilate=replace_stride_with_dilation[1])]
        features += [self._make_layer(block, 512, layers[3], stride=2,
                                      dilate=replace_stride_with_dilation[2])]
        if self.drop_rate > 0:
            features += [nn.Dropout(self.drop_rate)]
        features += [nn.AdaptiveAvgPool2d((1, 1))]

        self.features = nn.Sequential(*features)

        self.classifier = nn.Linear(512 * block.expansion, self.n_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode="fan_out",
                                     nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual
        # block behaves like an identity.
        # This improves the model by 0.2~0.3% according to <https://arxiv.org/
        # abs/1706.02677>
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample,
                            self.groups, self.base_width, previous_dilation,
                            norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width,
                                dilation=self.dilation, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_low(self, x):
        return torch.flatten(self.features(x), 1)

    def _forward_high(self, x):
        return torch.flatten(self.features(self.suppressor(x)), 1)

    def _forward_concat(self, x):
        x = torch.cat((x, self.suppressor(x)), 1)
        return torch.flatten(self.features(x), 1)

    def forward(self, x):
        x = self._forward_impl(x)
        return self.classifier(x)


def _resnet(params, block, layers):
    model = ResNet(params, block, layers)
    return model


def resnet18(params):
    return _resnet(params, BasicBlock, [2, 2, 2, 2])


def resnet34(params):
    return _resnet(params, BasicBlock, [3, 4, 6, 3])


def resnet50(params):
    return _resnet(params, Bottleneck, [3, 4, 6, 3])
