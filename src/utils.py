import argparse
import inspect
import os
import pickle
import random
import re
import subprocess
import sys
from datetime import datetime
from glob import glob
from logging import getLogger

import torch
import torch.nn as nn
from torch import optim

from .logger import create_logger
from .models.custom_layers import SuppressContent

sys.path.append("..")

FALSY_STRINGS = {"off", "false", "0"}
TRUTHY_STRINGS = {"on", "true", "1"}

MODELS_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                           "models")
if not os.path.exists(MODELS_PATH):
    subprocess.Popen("mkdir -p %s" % MODELS_PATH, shell=True).wait()

logger = getLogger()


def get_valid_input_names(mode="all"):
    """
    Get valid input names.
    """
    low_freq_inputs = ["rgb", "prnu_lp_low"]
    low_freq_inputs += ["rgb+" + x for x in ["con_conv", "finite_difference",
                                             "fixed_hpf"]]
    high_freq_inputs = ["con_conv", "finite_difference", "fixed_hpf",
                        "prnu_lp"]
    if mode == "all":
        return low_freq_inputs + high_freq_inputs
    elif mode == "low":
        return low_freq_inputs
    elif mode == "high":
        return high_freq_inputs
    else:
        raise Exception("Unknown get valid input name mode '%s'" % mode)


def glob_get_path(path):
    """
    Get reload path to the parameters of a model.
    """
    if path:
        path = glob(path)
        assert len(path) == 1, "multiple .pths (or folders) exist"
        path = path[0]
    return path


def reload_params(model_path):
    """
    Given a model path load the model parameters.
    """
    dirname, _ = os.path.split(model_path)
    with open(os.path.join(dirname, "params.pk"), "rb") as f:
        pkld_params = pickle.load(f)
    return pkld_params


def bool_flag(s):
    """
    Parse boolean arguments from the command line.
    """
    if s.lower() in FALSY_STRINGS:
        return False
    elif s.lower() in TRUTHY_STRINGS:
        return True
    else:
        raise argparse.ArgumentTypeError(
            "invalid value for a boolean flag. use 0 or 1")


def separate_bnorm_and_biases(model):
    """
    Separate batch norm / bias parameters from other parameters. Weight decay
    should not be applied to batch norm / bias parameters.
    """
    decay_suppressor = []
    decay = []
    no_decay = []
    for m in model.modules():
        if isinstance(m, nn.Linear):
            decay.append(m.weight)
            if m.bias is not None:
                no_decay.append(m.bias)
        elif isinstance(m, nn.Conv2d):
            decay.append(m.weight)
            if m.bias is not None:
                no_decay.append(m.bias)
        elif isinstance(m, nn.ConvTranspose2d):
            decay.append(m.weight)
            if m.bias is not None:
                no_decay.append(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            if m.bias is not None:
                no_decay.append(m.weight)
            if m.bias is not None:
                no_decay.append(m.bias)
        elif isinstance(m, nn.BatchNorm1d):
            if m.bias is not None:
                no_decay.append(m.weight)
            if m.bias is not None:
                no_decay.append(m.bias)
        elif isinstance(m, SuppressContent):
            # only decay a suppress image content layer if it requires a
            # gradient
            try:
                if m.suppressor.requires_grad:
                    decay_suppressor.append(m.suppressor)
                else:
                    no_decay.append(m.suppressor)
            except AttributeError:
                continue

    assert len(list(model.parameters())) == len(decay) + len(no_decay) + \
        len(decay_suppressor), "error separating batch norm / bias params"
    return decay_suppressor, decay, no_decay


def get_optimizer(model, init_params):
    """
    Parse optimizer parameters.
    init_params should be of the form:
        - "sgd,lr=0.01,weight_decay=0.0005"
        - "adagrad,lr=0.1,lr_decay=0.05"
    """
    s = init_params
    if "," in s:
        method = s[:s.find(",")]
        optim_params = {}
        for x in s[s.find(",") + 1:].split(","):
            split = x.split("=")
            assert len(split) == 2
            if split[0] == "nesterov":
                optim_params[split[0]] = bool_flag(split[1])
                continue
            else:
                assert re.match(r"^[+-]?(\d+(\.\d*)?|\.\d+)$",
                                split[1]) is not None
                optim_params[split[0]] = float(split[1])
    else:
        method = s
        optim_params = {}

    if method == "adadelta":
        optim_fn = optim.Adadelta
    elif method == "adagrad":
        optim_fn = optim.Adagrad
    elif method == "adam":
        optim_fn = optim.Adam
        optim_params["betas"] = (optim_params.get("beta1", 0.5),
                                 optim_params.get("beta2", 0.999))
        optim_params.pop("beta1", None)
        optim_params.pop("beta2", None)
    elif method == "adam_standard":
        optim_fn = optim.Adam
        optim_params["betas"] = (optim_params.get("beta", 0.9),
                                 optim_params.get("beta2", 0.999))
        optim_params.pop("beta1", None)
        optim_params.pop("beta2", None)
    elif method == "adamax":
        optim_fn = optim.Adamax
    elif method == "asgd":
        optim_fn = optim.ASGD
    elif method == "rmsprop":
        optim_fn = optim.RMSprop
    elif method == "rprop":
        optim_fn = optim.Rprop
    elif method == "sgd":
        optim_fn = optim.SGD
        assert "lr" in optim_params
    else:
        raise Exception("Unknown optimization method: '%s'" % method)

    # check that the optimizer is given valid parameters
    expected_args = inspect.getargspec(optim_fn.__init__)[0]
    assert expected_args[:2] == ["self", "params"]
    if not all(k in expected_args[2:] for k in optim_params.keys()):
        raise Exception("Unexpected parameters: expected '%s', got '%s'" % (
            str(expected_args[2:]), str(optim_params.keys())))

    if optim_params.get("weight_decay") is not None:
        decay_suppressor, decay, no_decay = separate_bnorm_and_biases(model)
        if decay_suppressor == []:
            model_params = [{"params": decay,
                             "weight_decay": optim_params.get("weight_decay")},
                            {"params": no_decay}]
        else:
            # for a trainable image content suppression layer, the initial
            # learning rate is 1/10th of the learning rate
            model_params = [{"params": decay_suppressor,
                             "weight_decay": optim_params.get("weight_decay"),
                             "lr": optim_params["lr"] / 10.},
                            {"params": decay,
                             "weight_decay": optim_params.get("weight_decay")},
                            {"params": no_decay}]

        optim_params.pop("weight_decay", None)
        return optim_fn(model_params, **optim_params)
    else:
        return optim_fn(model.parameters(), **optim_params)


def get_dump_path(params):
    """
    Create a directory to store the experiment.
    """
    assert os.path.isdir(MODELS_PATH)

    # create the sweep path if it does not exist
    sweep_path = os.path.join(MODELS_PATH, params.name)
    if not os.path.exists(sweep_path):
        subprocess.Popen("mkdir -p %s" % sweep_path, shell=True).wait()

    # create a name for the experiment based on current date and time
    now = datetime.now()
    chars = "abcdefghijklmnopqrstuvwxyz0123456789"
    while True:
        exp_id = now.strftime("%d%m%y_%H%M%S_") + \
            "".join(random.choice(chars) for _ in range(10))
        dump_path = os.path.join(MODELS_PATH, params.name, exp_id)
        if not os.path.isdir(dump_path):
            break

    # create the dump folder
    if not os.path.exists(dump_path):
        subprocess.Popen("mkdir -p %s" % dump_path, shell=True).wait()
    return dump_path


def initialize_exp(params, model_type):
    """
    Initialize the experiment.
    """
    # set the name of the experiment
    if ("classifiers" == model_type) or ("estimators" == model_type):
        top_lvl = os.path.join(model_type,
                               "adversary" if "adversary" in params.train_root
                               else "examiner")
        params.name = os.path.join(top_lvl,
                                   params.clf_architecture if model_type !=
                                   "estimators" else "",
                                   params.clf_input
                                   if model_type != "estimators" else "",
                                   params.estimator_output
                                   if "estimators" == model_type else "",
                                   "train")
    elif "gans" == model_type:
        top_lvl = os.path.join(model_type, "adversary")
        mid_lvl = os.path.join(params.clf_low_input
                               if params.clf_low_input is not None else "None",
                               params.clf_high_input
                               if params.clf_high_input is not None else "None"
                               )
        params.name = os.path.join(top_lvl, mid_lvl,
                                   params.gen_input + "_" + params.dis_input,
                                   "train")
    else:
        raise Exception("Unknown model type '%s'" % model_type)

    # dump parameters
    params.dump_path = get_dump_path(params)
    pickle.dump(params, open(os.path.join(params.dump_path,
                                          "params.pkl"), "wb"))

    # create a logger
    logger = create_logger(os.path.join(params.dump_path, "train.log"))
    if not params.resume:
        logger.info("============ initialized logger ============")
        logger.info("\n".join("%s: %s" % (k, str(v)) for k, v
                              in sorted(dict(vars(params)).items())))

        # store all trained model params in a common .csv file
        trained_model_params_file = os.path.join(
            MODELS_PATH, top_lvl, "training_params.txt")

        # write keys to the file if the file is empty
        if not os.path.exists(trained_model_params_file):
            with open(trained_model_params_file, "w+") as f:
                _ = f.write("".join("%s\t" % (k) for k, _ in sorted(
                    dict(vars(params)).items()))[:-1] + '\n')

        # write the params to the file
        with open(trained_model_params_file, "a+") as f:
            _ = f.write("".join("%s\t" % (str(v))
                                for _, v
                                in sorted(dict(vars(params)).items()))[:-1] +
                        "\n")
    else:
        # if resuming then we need to remove the last unfinished epoch's train
        # log entries
        with open(os.path.join(params.dump_path, "train.log"), "r") as f:
            lines = f.readlines()

        # find line where "end of epoch" was last logged
        keep_line = 0
        for i, line in enumerate(lines):
            if re.search("end of epoch %i." % (params.n_epoch_start - 1),
                         line):
                keep_line = i + 1

        with open(os.path.join(params.dump_path, "temporary_file.log"),
                  "w") as f:
            for i, line in enumerate(lines):
                if i < keep_line:
                    _ = f.write(line)
                else:
                    break

        # replace old train.log with the new temporary_file.log
        os.replace(os.path.join(params.dump_path, "temporary_file.log"),
                   os.path.join(params.dump_path, "train.log"))
    return logger


def initialize_test_exp(params, folder_name):
    """
    Initialize the test experiment.
    """
    # dump parameters
    params.dump_path = params.dump_path.replace("train", folder_name)
    # create the dump folder
    if not os.path.exists(params.dump_path):
        subprocess.Popen("mkdir -p %s" % params.dump_path, shell=True).wait()

    pickle.dump(params, open(os.path.join(params.dump_path, "params.pkl"),
                             "wb"))

    # create a logger
    logger = create_logger(os.path.join(params.dump_path, "%s.log"
                                        % folder_name))
    logger.info("============ initialized logger ============")
    logger.info("\n".join("%s: %s" % (k, str(v))
                          for k, v in sorted(dict(vars(params)).items())))
    return logger


def save_state(params, n_epoch, best_names, best_vals, checkpoint_name,
               model_names=[], models=[], optimizer_names=[], optimizers=[]):
    """
    Save the classifier (and optionally the optimizer) state(s)
    """
    assert len(model_names) == len(models), "number of model names and models "
    "differ"
    assert len(optimizer_names) == len(optimizers), "number of optimizer "
    "name and optimizers differ"
    assert len(best_names) == len(best_vals), "number of best names and best "
    "vals differ"

    path = os.path.join(params.dump_path, "%s.pth" % checkpoint_name)
    chkpt = {"n_epoch": n_epoch,
             "n_epochs": params.n_epochs}
    chkpt.update({best_names[i]: best_vals[i] for i in range(len(best_names))})

    # save optimizer
    if params.save_opt:
        # iterate over optimizer names / optimizers
        for name, opt in zip(optimizer_names, optimizers):
            chkpt[str(name)] = opt.state_dict()

    # save models
    if len(params.gpu_devices) > 1:
        # iterate over model names / dataparallel models
        for name, model in zip(model_names, models):
            chkpt[str(name)] = model.module.state_dict()
    else:
        # iterate over model names and models
        for name, model in zip(model_names, models):
            chkpt[str(name)] = model.state_dict()

    # save checkpoint
    print("Saving checkpoint to '%s'" % path)
    torch.save(chkpt, path)


def reload_state_dict(params, checkpoint_path, resume, model_names=[],
                      models=[], optimizer_names=[], optimizers=[]):
    """
    Reload a previously trained model's state.
    """
    assert len(model_names) == len(models), "number of model names and models "
    "differ"
    assert len(optimizer_names) == len(optimizers), "number of optimizer "
    "names and optimizers differ"

    # reload the models (and optimizers)
    assert os.path.isfile(checkpoint_path)
    to_reload = torch.load(checkpoint_path)

    if resume:
        assert params, "resume is 'True', but params were not given"
        params.n_epoch_start = to_reload["n_epoch"] + 1
        params.n_epochs = to_reload["n_epochs"]

    for name, model in zip(model_names, models):
        name = str(name)
        if (set([name]).intersection(set(to_reload.keys())) != set()):
            reloader(model, to_reload[name])

    for name, opt in zip(optimizer_names, optimizers):
        name = str(name)
        if (set([name]).intersection(set(to_reload.keys())) != set()):
            opt.load_state_dict(to_reload[name])


def reloader(current, saved):
    """
    Load saved state based on keys.
    """
    current_params = set(current.state_dict().keys())
    to_reload_params = set(saved.keys())
    assert current_params == to_reload_params, (current_params -
                                                to_reload_params,
                                                to_reload_params -
                                                current_params)

    # copy saved parameters
    for k in current.state_dict().keys():
        if current.state_dict()[k].size() != saved[k].size():
            raise Exception("Expected tensor {} of size {}, but got {}".format(
                k, current.state_dict()[k].size(),
                saved[k].size()
            ))
        current.state_dict()[k].copy_(saved[k])


def print_metric(values, accuracy=True):
    """
    Print metric.
    """
    assert all(len(x) == 2 for x in values)
    for name, value in values:
        if value == value:
            logger.info("{:<20}: {:>6}".format(name, "%.3f%%" %
                                               (100 * value)if accuracy
                                               else "%.3f" % (value)))
        else:
            # deal wih NaN
            logger.info("{:<20}: --".format(name))
    logger.info("")


def schedule_lr(optimizer):
    """
    Simple learning rate schedule, which divides the current learning rate by
    10
    """
    for params in optimizer.param_groups:
        params['lr'] /= 10.


def preprocess(data):
    """
    Rescale data from [0, 255] to [-1, 1].
    """
    return data.div(255.).sub(0.5).div(0.5)


def reverse_preprocess(data):
    """
    Rescale data from [-1, 1] to [0, 255].
    """
    return data.mul(0.5).add(0.5).mul(255.)


def generate_targets(n_classes, labels, seed=None):
    """
    Generate target labels different from the ground truth labels.
    """
    target_labels = torch.zeros_like(labels)
    if seed is None:
        for _ in range(len(labels)):
            target_labels[_] = random.sample(
                [x for x in range(n_classes) if x != labels[_]], 1)[0]
    else:
        rng = random.Random(seed)
        for _ in range(len(labels)):
            target_labels[_] = rng.sample(
                [x for x in range(n_classes) if x != labels[_]], 1)[0]
    return target_labels


def lambda_coeff(schedule, params):
    """
    Compute lambda coefficient for the loss.
    First argument of schedule: maximum of the loss feedback coefficient (0 to
    disable). Second argument of schedule: epochs to progressively increase
    the loss (0 to disable).
    """
    max_coeff = schedule[0]
    n_epochs = schedule[1]
    if n_epochs == 0:
        return max_coeff
    else:
        return max_coeff * float(min(params.n_iters_done, n_epochs)) / n_epochs
