"""
Adapted from <https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/
master/models/networks.py>.
"""

import functools

import torch
import torch.nn as nn

from .custom_layers import Normalize, SuppressContent


class ResnetBlock(nn.Module):
    """
    Defines a ResNet block for use in a ResNet-based generator.
    """

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
                       norm_layer(dim),
                       nn.ReLU(True)]

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


class ResnetGenerator(nn.Module):
    """
    Defines a ResNet-based generator.
    """

    def __init__(self, params, norm_layer=nn.BatchNorm2d,
                 padding_type="reflect"):
        super(ResnetGenerator, self).__init__()
        self.ptc_fm = params.ptc_fm
        self.ngf = params.ngf
        self.n_classes = params.n_classes
        self.primary_gpu = params.primary_gpu
        self.n_blocks = params.n_blocks
        assert(self.n_blocks >= 0)

        self.normalizer = Normalize(params)

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(self.ptc_fm + self.n_classes, self.ngf,
                           kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(self.ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(self.ngf * mult, self.ngf * mult * 2,
                                kernel_size=3, stride=2, padding=1,
                                bias=use_bias),
                      norm_layer(self.ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2**n_downsampling
        for i in range(self.n_blocks):
            model += [ResnetBlock(self.ngf * mult, padding_type=padding_type,
                                  norm_layer=norm_layer, use_bias=use_bias)]

        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [nn.ConvTranspose2d(self.ngf * mult,
                                         int(self.ngf * mult / 2),
                                         kernel_size=3, stride=2, padding=1,
                                         output_padding=1, bias=use_bias),
                      norm_layer(int(self.ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(self.ngf, self.ptc_fm, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def label_to_condition(self, h_x, w_x, y):
        condition = torch.zeros((len(y), self.n_classes, h_x, w_x),
                                dtype=torch.float32).to(
            device=self.primary_gpu)
        condition[torch.arange(0, len(y), dtype=torch.long), y.to(torch.long),
                  :, :] += 1
        return condition

    def forward(self, x, y):
        # convert a label to a condition
        condition = self.label_to_condition(x.size(2), x.size(3), y)

        # normalize the input image
        x = self.normalizer.normalize(x)

        # concatenate the image and label condition
        x = torch.cat((x, condition), dim=1)

        # transform the input, then normalize the output
        x = self.model(x)
        return self.normalizer.reverse_normalize(x)


class NLayerDiscriminator(nn.Module):
    def __init__(self, params, norm_layer=nn.BatchNorm2d):
        """
        Defines an N-layered PatchGAN discriminator.
        """
        super(NLayerDiscriminator, self).__init__()

        self.dis_input = params.dis_input
        self.n_classes = params.n_classes
        self.ptc_fm = params.ptc_fm
        self.ndf = params.ndf
        self.drop_rate = params.dis_drop_rate
        self.n_dis_layers = params.n_dis_layers
        self.primary_gpu = params.primary_gpu

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1

        normalizer = Normalize(params)
        self.preprocessor = normalizer.normalize

        # suppress the image content if dis_input != "rgb"
        if (self.dis_input != "rgb"):
            self.suppressor = SuppressContent(params)

        # concat low- / high-frequency if "+" in dis_input
        self.concat = False
        if "+" in self.dis_input:
            self.concat = True

        if self.concat:
            # concat low / high spatial frequency components
            if "fixed_hpf" in self.dis_input:
                self.ptc_fm += 1
            elif "finite_difference" in self.dis_input:
                self.ptc_fm += 6
            else:
                self.ptc_fm += 3
            self._forward_impl = self._forward_concat
        else:
            # low- / high-frequency
            if self.dis_input == "rgb":
                self.ptc_fm = 3
                self._forward_impl = self._forward_low
            else:
                if self.dis_input == "fixed_hpf":
                    self.ptc_fm = 1
                elif self.dis_input == "finite_difference":
                    self.ptc_fm = 6
                else:
                    self.ptc_fm = 3
                self._forward_impl = self._forward_high

        preconditional_features = []
        preconditional_features += [nn.Conv2d(self.ptc_fm, self.ndf,
                                              kernel_size=kw, stride=2,
                                              padding=padw)]
        preconditional_features += [nn.LeakyReLU(0.2, True)]

        features = []

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, self.n_dis_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            if n == 1:
                features += [
                    nn.Conv2d(self.ndf * nf_mult_prev + self.n_classes,
                              self.ndf * nf_mult, kernel_size=kw, stride=2,
                              padding=padw, bias=use_bias),
                    norm_layer(self.ndf * nf_mult),
                    nn.LeakyReLU(0.2, True)
                ]
            else:
                features += [
                    nn.Conv2d(self.ndf * nf_mult_prev, self.ndf * nf_mult,
                              kernel_size=kw, stride=2, padding=padw,
                              bias=use_bias),
                    norm_layer(self.ndf * nf_mult),
                    nn.LeakyReLU(0.2, True)
                ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**self.n_dis_layers, 8)
        features += [
            nn.Conv2d(self.ndf * nf_mult_prev, self.ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(self.ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        if self.drop_rate > 0:
            features += [nn.Dropout(self.drop_rate)]

        self.preconditional_features = nn.Sequential(*preconditional_features)
        self.features = nn.Sequential(*features)

        classifier = []
        classifier += [nn.Conv2d(self.ndf * nf_mult, 1, kernel_size=kw,
                                 stride=1, padding=padw)]

        self.classifier = nn.Sequential(*classifier)

    def label_to_condition(self, h_x, w_x, y):
        condition = torch.zeros((len(y), self.n_classes, h_x, w_x),
                                dtype=torch.float32).to(
            device=self.primary_gpu)
        condition[torch.arange(0, len(y), dtype=torch.long), y.to(torch.long),
                  :, :] += 1
        return condition

    def _forward_low(self, x):
        return self.preconditional_features(x)

    def _forward_high(self, x):
        return self.preconditional_features(self.suppressor(x))

    def _forward_concat(self, x):
        x = torch.cat((x, self.suppressor(x)), 1)
        return self.preconditional_features(x)

    def forward(self, x, y):
        # compute initial features prior to inserting conditions
        if "prnu" not in self.dis_input:
            x = self.preprocessor(x)
        x = self._forward_impl(x)

        # convert a label to a condition
        condition = self.label_to_condition(x.size(2), x.size(3), y)

        # concatenate the image and label condition
        x = torch.cat((x, condition), dim=1)

        # classify
        x = self.features(x)
        return self.classifier(x)


class PixelDiscriminator(nn.Module):
    """
    Defines a 1x1 PatchGAN discriminator (pixelGAN).
    """

    def __init__(self, params, norm_layer=nn.BatchNorm2d):
        super(PixelDiscriminator, self).__init__()
        self.dis_input = params.dis_input
        self.n_classes = params.n_classes
        self.ptc_fm = params.ptc_fm
        self.ndf = params.ndf
        self.drop_rate = params.dis_drop_rate
        self.n_dis_layers = params.n_dis_layers
        self.primary_gpu = params.primary_gpu

        # no need to use bias as BatchNorm2d has affine parameters
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        normalizer = Normalize(params)
        self.preprocessor = normalizer.normalize

        # suppress the image content if dis_input != "rgb"
        if (self.dis_input != "rgb"):
            self.suppressor = SuppressContent(params)

        # concat low- / high-frequency if "+" in dis_input
        self.concat = False
        if "+" in self.dis_input:
            self.concat = True

        if self.concat:
            # concat low / high spatial frequency components
            if "fixed_hpf" in self.dis_input:
                self.ptc_fm += 1
            elif "finite_difference" in self.dis_input:
                self.ptc_fm += 6
            else:
                self.ptc_fm += 3

            self._forward_impl = self._forward_concat
        else:
            # low- / high-frequency
            if self.dis_input == "rgb":
                self.ptc_fm = 3
                self._forward_impl = self._forward_low
            else:
                if self.dis_input == "fixed_hpf":
                    self.ptc_fm = 1
                elif self.dis_input == "finite_difference":
                    self.ptc_fm = 6
                else:
                    self.ptc_fm = 3
                self._forward_impl = self._forward_high

        preconditional_features = []
        preconditional_features += [nn.Conv2d(self.ptc_fm, self.ndf,
                                              kernel_size=1, stride=1,
                                              padding=0)]
        preconditional_features += [nn.LeakyReLU(0.2, True)]

        features = []
        features += [
            nn.Conv2d(self.ndf + self.n_classes, self.ndf * 2, kernel_size=1,
                      stride=1, padding=0, bias=use_bias),
            norm_layer(self.ndf * 2),
            nn.LeakyReLU(0.2, True)]

        if self.drop_rate > 0:
            features += [nn.Dropout(self.drop_rate)]

        self.preconditional_features = nn.Sequential(*preconditional_features)
        self.features = nn.Sequential(*features)

        classifier = []
        classifier += [nn.Conv2d(self.ndf * 2 if not self.dual_stream
                                 else 2 * self.ndf * 2, 1, kernel_size=1,
                                 stride=1, padding=0, bias=use_bias)]

        self.classifier = nn.Sequential(*classifier)

    def label_to_condition(self, h_x, w_x, y):
        condition = torch.zeros((len(y), self.n_classes, h_x, w_x),
                                dtype=torch.float32).to(
            device=self.primary_gpu)
        condition[torch.arange(0, len(y), dtype=torch.long), y.to(torch.long),
                  :, :] += 1
        return condition

    def _forward_low(self, x):
        return self.preconditional_features(x)

    def _forward_high(self, x):
        return self.preconditional_features(self.suppressor(x))

    def _forward_concat(self, x):
        x = torch.cat((x, self.suppressor(x)), 1)
        return self.preconditional_features(x)

    def forward(self, x, y):
        # compute initial features prior to inserting conditions
        if "prnu" not in self.dis_input:
            x = self.preprocessor(x)
        x = self._forward_impl(x)

        # convert a label to a condition
        condition = self.label_to_condition(x.size(2), x.size(3), y)

        # concatenate the image and label condition
        x = torch.cat((x, condition), dim=1)

        # classify
        x = self.features(x)
        return self.classifier(x)
