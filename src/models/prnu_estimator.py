"""
Adapted from <https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/
master/models/networks.py>.
"""

import functools
import torch.nn as nn


class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_bias):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer,
                                                use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_bias):
        conv_block = []
        p = 0
        if padding_type == "reflect":
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == "replicate":
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == "zero":
            p = 1
        else:
            raise NotImplementedError("padding [%s] is not implemented"
                                      % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p,
                                 bias=use_bias),
                       norm_layer(dim), nn.ReLU(True)]

        p = 0
        if padding_type == "reflect":
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == "replicate":
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == "zero":
            p = 1
        else:
            raise NotImplementedError("padding [%s] is not implemented"
                                      % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p,
                                 bias=use_bias),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


class ResnetEstimator(nn.Module):
    def __init__(self, params, norm_layer=nn.BatchNorm2d,
                 padding_type="reflect"):
        super(ResnetEstimator, self).__init__()
        self.ptc_fm = params.ptc_fm
        self.nef = params.nef
        self.n_blocks = params.n_blocks
        assert(self.n_blocks >= 0)

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(self.ptc_fm, self.nef, kernel_size=7, padding=0,
                           bias=use_bias),
                 norm_layer(self.nef),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(self.nef * mult, self.nef * mult * 2,
                                kernel_size=3, stride=2, padding=1,
                                bias=use_bias),
                      norm_layer(self.nef * mult * 2),
                      nn.ReLU(True)]

        mult = 2**n_downsampling
        for i in range(self.n_blocks):
            model += [ResnetBlock(self.nef * mult, padding_type=padding_type,
                                  norm_layer=norm_layer, use_bias=use_bias)]

        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [nn.ConvTranspose2d(self.nef * mult, int(self.nef *
                                                              mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(self.nef * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(self.nef, self.ptc_fm, kernel_size=7, padding=0)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)
