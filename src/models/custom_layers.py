import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

from .utils import n_order_finite_difference


class SuppressContent(nn.Module):
    """
    Image content suppression layer.
    """

    def __init__(self, params):
        super(SuppressContent, self).__init__()
        try:
            self.net_input = params.clf_input
        except AttributeError:
            try:
                self.net_input = params.dis_input
            except AttributeError:
                # special case for estimator training
                self.net_input = params.estimator_output

        self.ptc_fm = params.ptc_fm

        if "con_conv" in self.net_input:
            self.suppressor = nn.Parameter(torch.FloatTensor(self.ptc_fm,
                                                             self.ptc_fm,
                                                             5,
                                                             5))
            n = 5 * 5 * self.ptc_fm
            init.normal_(self.suppressor.data, 0.0, math.sqrt(2. / n))
            self._forward = self._forward_other_high_pass
        elif "fixed_hpf" in self.net_input:
            self.suppressor = nn.Parameter(torch.FloatTensor(1,
                                                             self.ptc_fm,
                                                             5,
                                                             5),
                                           requires_grad=False)
            for i in range(self.ptc_fm):
                self.suppressor.data[:, i, :, :] = \
                    torch.FloatTensor([
                                      [-1, 2, -2, 2, -1],
                                      [2, -6, 8, -6, 2],
                                      [-2, 8, -12, 8, -2],
                                      [2, -6, 8, -6, 2],
                                      [-1, 2, -2, 2, -1]
                                      ]) * (1 / 12.)
            self._forward = self._forward_other_high_pass
        elif "finite_difference" in self.net_input:
            self.suppressor = n_order_finite_difference
            self._forward = self._forward_finite_difference
        else:
            self.suppressor = None
            self._forward = self._forward_prnu

    def _forward_prnu(self, x):
        return x

    def _forward_finite_difference(self, x):
        return self.suppressor(x)

    def _forward_other_high_pass(self, x):
        return F.conv2d(x, self.suppressor, padding=2)

    def forward(self, x):
        return self._forward(x)


class Normalize(nn.Module):
    """
    Normalization layer.
        "Normalize.normalize" rescales data from [0, 255] to [-1, 1].
        "Normalize.reverse_normalize" rescales data from [-1, 1] to [0, 255].
    """

    def __init__(self, params):
        super(Normalize, self).__init__()
        self.sub = torch.as_tensor([0.5, 0.5, 0.5]).to(params.primary_gpu).\
            view(-1, 1, 1)
        self.div = torch.as_tensor([0.5, 0.5, 0.5]).to(params.primary_gpu).\
            view(-1, 1, 1)

    def normalize(self, x):
        x = x / 255.
        return (x - self.sub) / self.div

    def reverse_normalize(self, x):
        x = (x * self.div) + self.sub
        return x * 255.
