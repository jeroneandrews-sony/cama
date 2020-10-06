import functools

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


def n_order_finite_difference(imgs, n=3):
    """
    Compute n-th order horizontal and vertical derivatives (finite
    differences).
    """
    # images are of shape batch x channel x height x width
    b, c, h, w = imgs.shape
    dx_image = imgs
    dy_image = imgs

    # n-th order horizontal and vertical derivatives (finite differences)
    for i in range(n):
        dy_image = dy_image[:, :, 1:h, :] - dy_image[:, :, 0:h - 1, :]
        dx_image = dx_image[:, :, :, 1:w] - dx_image[:, :, :, 0:w - 1]
        dy_image = F.pad(dy_image, (0, 0, 1, 0, 0, 0, 0, 0), "constant", 0)
        dx_image = F.pad(dx_image, (1, 0, 0, 0, 0, 0, 0, 0), "constant", 0)
    return torch.cat((dx_image, dy_image), 1)


def update_con_conv(model, params):
    """
    Update the parameters of a constrained convolutional layer.
    """
    if len(params.gpu_devices) > 1:
        new_vals = model.module.suppressor.suppressor.data
    else:
        new_vals = model.suppressor.suppressor.data

    new_vals[:, :, 2, 2] = 0.
    new_vals = new_vals.view(3, 3, 1, 5 * 5)
    denom = new_vals.sum([-1]).view(3, 3, 1, 1)
    new_vals = new_vals / denom
    new_vals = new_vals.view(3, 3, 5, 5)
    new_vals[:, :, 2, 2] = -1.

    if len(params.gpu_devices) > 1:
        model.module.suppressor.suppressor.data = new_vals
    else:
        model.suppressor.suppressor.data = new_vals


def get_norm_layer(norm_type="batch"):
    """
    Get normalization layer.
    """
    if norm_type == "batch":
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == "instance":
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False,
                                       track_running_stats=False)
    elif norm_type == "none":
        norm_layer = None
    else:
        raise NotImplementedError("normalisation layer [%s] was not found"
                                  % norm_type)
    return norm_layer


def init_weights(net, init_type="normal", gain=0.02):
    """
    Initialize the weights of a layer.
    """
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, "weight") and (classname.find("Conv") != -1 or
                                     classname.find("Linear") != -1):
            if init_type == "normal":
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == "xavier":
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == "kaiming":
                init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
            elif init_type == "orthogonal":
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError("initialization method [%s] is not "
                                          "implemented" % init_type)
            if hasattr(m, "bias") and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find("BatchNorm2d") != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    net.apply(init_func)


def init_net(net, init_type="normal", init_gain=0.02):
    """
    Initialize a network (model).
    """
    init_weights(net, init_type, gain=init_gain)
    return net


def Classifier(params):
    """
    Returns a classifier based on the given parameters.
    """
    net = None

    if params.clf_architecture == "resnet18":
        from .resnet import resnet18
        net = resnet18(params)
    elif params.clf_architecture == "resnet34":
        from .resnet import resnet34
        net = resnet34(params)
    elif params.clf_architecture == "resnet50":
        from .resnet import resnet50
        net = resnet50(params)
    elif params.clf_architecture == "densenet40":
        from .densenet import densenet40
        net = densenet40(params)
    elif params.clf_architecture == "densenet100":
        from .densenet import densenet100
        net = densenet100(params)
    elif params.clf_architecture == "vgg11":
        from .vgg import vgg11_bn
        net = vgg11_bn(params)
    elif params.clf_architecture == "vgg13":
        from .vgg import vgg13_bn
        net = vgg13_bn(params)
    elif params.clf_architecture == "vgg16":
        from .vgg import vgg16_bn
        net = vgg16_bn(params)
    elif params.clf_architecture == "vgg19":
        from .vgg import vgg19_bn
        net = vgg19_bn(params)
    else:
        raise NotImplementedError("Architecture [%s] is not recognized"
                                  % params.clf_architecture)
    return net


def Estimator(params, norm="batch", init_type="normal", init_gain=0.02):
    """
    Returns an estimator based on the given parameters.
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    from .prnu_estimator import ResnetEstimator
    net = ResnetEstimator(params, norm_layer=norm_layer)
    return init_net(net, init_type, init_gain)


def Generator(params, norm="batch", init_type="normal", init_gain=0.02):
    """
    Returns a generator based on the given parameters.
    """
    from .cgan import ResnetGenerator
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    net = ResnetGenerator(params, norm_layer=norm_layer)
    return init_net(net, init_type, init_gain)


def Discriminator(params, norm="batch", init_type="normal", init_gain=0.02):
    """
    Returns a discriminator based on the given parameters.
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if params.dis_type == "patch":
        from .cgan import NLayerDiscriminator
        net = NLayerDiscriminator(params, norm_layer=norm_layer)
    elif params.dis_type == "pixel":
        from .cgan import PixelDiscriminator
        net = PixelDiscriminator(params, norm_layer=norm_layer)
    else:
        raise NotImplementedError("Discriminator model name [%s] is not "
                                  "recognized" % net)
    return init_net(net, init_type, init_gain)
