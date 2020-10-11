"""
Adapted from <https://github.com/pytorch/vision/blob/master/torchvision/models/
vgg.py>.
"""

import torch
import torch.nn as nn
from torch.nn import init

from .custom_layers import SuppressContent


class VGG(nn.Module):
    def __init__(self, params, cfg):
        super(VGG, self).__init__()
        self.clf_input = params.clf_input
        self.n_classes = params.n_classes
        self.ptc_fm = params.ptc_fm
        self.ptc_sz = params.ptc_sz
        self.drop_rate = params.drop_rate

        # suppress the image content if clf_input != "rgb"
        if (self.clf_input != "rgb"):
            self.suppressor = SuppressContent(params)

        # concat low- / high-frequency if "+" in clf_input
        self.concat = False
        if "+" in self.clf_input:
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

        features = self.make_layers(self.ptc_fm, cfg)

        self.features = nn.Sequential(*features)

        self.classifier = nn.Linear(512, self.n_classes)

        self._initialize_weights()

    def make_layers(self, ptc_fm, cfg):
        layers = []
        for i, v in enumerate(cfg):
            if v == "M":
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                # biases are set to False as the conv layers are followed by
                # batchnorm
                conv2d = nn.Conv2d(ptc_fm, v, kernel_size=3, padding=1,
                                   bias=False)
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                ptc_fm = v
        if self.drop_rate > 0:
            layers += [nn.Dropout(self.drop_rate)]
        layers += [nn.AdaptiveAvgPool2d((1, 1))]
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

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode="fan_out",
                                     nonlinearity="relu")
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, 0, 0.01)
                init.constant_(m.bias, 0)


cfgs = {
    "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "B": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512,
          "M"],
    "D": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M",
          512, 512, 512, "M"],
    "E": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512,
          512, "M", 512, 512, 512, 512, "M"],
}


def _vgg(params, cfg):
    model = VGG(params, cfgs[cfg])
    return model


def vgg11_bn(params):
    return _vgg(params, "A")


def vgg13_bn(params):
    return _vgg(params, "B")


def vgg16_bn(params):
    return _vgg(params, "D")


def vgg19_bn(params):
    return _vgg(params, "E")
