import torch
import torch.nn as nn
import torch.nn.functional as F


def xent_loss(preds, targets):
    """
    Compute cross-entropy classification loss.
    """
    return F.cross_entropy(preds, targets)


def bce_logits_loss(preds, targets):
    """
    Compute binary cross-entropy classification loss.
    """
    return F.binary_cross_entropy_with_logits(preds, targets)


def mse_loss(preds, targets, reduction=True):
    """
    Compute mean squared error l2 loss.
    """
    if reduction:
        return F.mse_loss(preds, targets)
    else:
        return F.mse_loss(preds, targets, reduction="none")


def l1_loss(preds, targets):
    """
    Compute l1 loss.
    """
    return F.l1_loss(preds, targets)


class gan_loss(nn.Module):
    """
    Compute GAN loss.
    Binary cross-entropy with logits loss or least squares mean squared error
    l2 loss.
    """

    def __init__(self, use_lsgan=True, target_real_label=1.0,
                 target_fake_label=0.0):
        super(gan_loss, self).__init__()
        self.register_buffer("real_label", torch.tensor(target_real_label))
        self.register_buffer("fake_label", torch.tensor(target_fake_label))
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCEWithLogitsLoss()

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(input)

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)
