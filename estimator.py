import argparse
import json
import os

import numpy as np
import torch
from data.config import centre_crop_size, preproc_img_dir
from src.evaluation import estimator_evaluate
from src.loader import dataset_loader
from src.models.utils import Estimator
from src.training import estimator_train
from src.utils import (bool_flag, get_optimizer, glob_get_path, initialize_exp,
                       print_metric, reload_params, reload_state_dict,
                       save_state, schedule_lr)
from torchsummary import summary

# parse parameters
parser = argparse.ArgumentParser(description="Camera model attribution "
                                 "PRNU estimator training")

# main parameters
parser.add_argument("--user", type=str, default="adversary",
                    help="Dataset (adversary / examiner)")
parser.add_argument("--ptc_sz", type=int, default=64,
                    help="Patch width/height")
parser.add_argument("--ptc_fm", type=int, default=3,
                    help="Number of input feature maps (channels)")
parser.add_argument("--expanded_cms", type=bool_flag, default=False,
                    help="Training with an expanded set of camera models"
                    "(only valid if user is examiner)")

# network architecture
parser.add_argument("--estimator_output", type=str, default="prnu_lp",
                    help="Estimate prnu_lp, i.e. prnu noise with linear "
                    "pattern, of an rgb image (prnu_lp)")
parser.add_argument("--nef", type=int, default=64,
                    help="Number of feature maps in the first and last "
                    "convolutional layers")
parser.add_argument("--n_blocks", type=int, default=2,
                    help="Number of residual blocks in the estimator")
parser.add_argument("--drop_rate", type=float, default=0,
                    help="Dropout rate in the residual blocks")

# training parameters
parser.add_argument("--batch_size", type=int, default=128,
                    help="Batch size (training)")
parser.add_argument("--test_batch_size", type=int, default=32,
                    help="Batch size (validation / testing)")
parser.add_argument("--rnd_crops", type=bool_flag, default=False,
                    help="Extract patches randomly (True) or from a "
                    "non-overlapping grid (False)")
parser.add_argument("--n_epochs", type=int, default=90,
                    help="Total number of epochs")
parser.add_argument("--n_samples_per_epoch", type=int, default=150000,
                    help="Number of training samples per epoch")
parser.add_argument("--optimizer", type=str, default="adam_standard,"
                    "weight_decay=0.0005",
                    help="Estimator optimizer (adam_standard,"
                    "weight_decay=0.0005 / sgd,lr=0.1,weight_decay=0.0005,"
                    "momentum=0.9,nesterov=True)")
parser.add_argument("--save_opt", type=bool_flag, default=True,
                    help="Save optimizer")
parser.add_argument("--lr_milestones", nargs='+', type=int, default=[],
                    help="Epochs to divide learning rate by 10")

# loaders / gpus
parser.add_argument("--n_workers", type=int, default=10,
                    help="Number of workers per data loader")
parser.add_argument("--pin_memory", type=bool_flag, default=True,
                    help="Pin memory of data loaders")
parser.add_argument("--gpu_devices", nargs='+', type=int, default=[0],
                    help="Which gpu devices to use")

# reload
parser.add_argument("--reload", type=str, default="",
                    help="Path to a pre-trained estimator (and optimizer if "
                    "saved)")
parser.add_argument("--resume", type=bool_flag, default=False,
                    help="Resume training")

# debug
parser.add_argument("--debug", type=bool_flag, default=False,
                    help="Debug mode (only use a subset of the available data)"
                    )

params = parser.parse_args()

# if resume, reload necessary parameters
if params.resume:
    assert params.reload, "tried to resume training, but no reload path was "
    "given"
    params.reload = glob_get_path(params.reload)
    reload_path = params.reload
    assert os.path.isfile(reload_path), "estimator reload file does not exist"
    params = reload_params(params.reload)
    params.resume = True
    params.reload = reload_path
    del reload_path
else:
    params.n_epoch_start = 1

assert not params.reload or os.path.isfile(params.reload), "estimator reload "
"file does not exist"

# set model type
params.model_type = "estimators"

# lower case
params.estimator_output = params.estimator_output.lower()
params.optimizer = params.optimizer.lower()

if centre_crop_size % params.ptc_sz != 0:
    assert params.rnd_crops, "(when rnd_crops is False) the spatial size of "
    "an image (%ix%i) must be divisible by " % (centre_crop_size,
                                                centre_crop_size)
    "the spatial size of the patch window (%ix%i)" % (params.ptc_sz,
                                                      params.ptc_sz)

assert params.user in ["adversary", "examiner"], "user must be adversary or "
"examiner"

# set data paths
params.train_root = os.path.join(preproc_img_dir, params.user, "train")
params.val_root = os.path.join(preproc_img_dir, params.user, "validation")
params.test_root = os.path.join(preproc_img_dir, "test")

if params.expanded_cms:
    assert params.user == "examiner", "if training with an expanded set of "
    "camera models, user must be equal to examiner"

# specify which gpu(s) to use
assert torch.cuda.is_available(), "torch cuda must be available"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "%s" % ("".join(str(gpu_) + ","
                                                     for gpu_ in
                                                     params.gpu_devices)
                                             )[:-1]
params.primary_gpu = 0

# enable cudnn benchmark mode
torch.backends.cudnn.benchmark = True

assert params.estimator_output in ["prnu_lp"], "invalid estimator "
"output (prnu_lp)"

# load train / validation / test dataset
train_dataset, n_classes, _ = dataset_loader(params, True).train
validation_dataset, n_classes_valid, _ = dataset_loader(params, False).test(
    params.val_root)
test_dataset, n_classes_test, _ = dataset_loader(params, False).test(
    params.test_root)
params.n_classes = n_classes
assert n_classes == n_classes_valid == n_classes_test, "the number of "
"training classes must equal the number of validation / testing classes"

# build the model / optimizer
estimator = Estimator(params)
optimizer = get_optimizer(estimator, params.optimizer)

# reload state
if params.reload and params.resume:
    reload_state_dict(params, params.reload, params.resume, ["estimator"],
                      [estimator], ["optimizer"], [optimizer])
elif params.reload:
    reload_state_dict(params, params.reload, params.resume, ['estimator'],
                      [estimator], [], [])

# move the model to DataParallel
if len(params.gpu_devices) > 1:
    estimator = torch.nn.DataParallel(estimator,
                                      [gpu_for gpu_ in range(len(
                                          params.gpu_devices
                                      ))])

# move the model to cuda
estimator = estimator.to(device=params.primary_gpu)

# best accuracy / best rmse
best_acc = -1e12
best_rmse = 1e12

# initialize logger and dump params
logger = initialize_exp(params,
                        model_type=params.model_type
                        if not params.expanded_cms else params.model_type +
                        "_exp")


if params.resume is False:
    logger.info(summary(estimator,
                        (params.ptc_fm, params.ptc_sz, params.ptc_sz)))
    logger.info(optimizer)

# commence training
for n_epoch in range(params.n_epoch_start, params.n_epochs + 1):

    logger.info("start of epoch %i..." % n_epoch)

    # initialize stats
    stats = []

    # estimator in train mode
    estimator.train()

    # estimator train
    estimator_train(n_epoch, estimator, optimizer, train_dataset, params,
                    stats)

    # estimator in eval mode
    estimator.eval()

    # evaluate estimator on validation / test data
    val_err = estimator_evaluate(estimator, validation_dataset, params)
    test_err = estimator_evaluate(estimator, test_dataset, params)

    # log estimator validation / test rmse
    log_err = [("val_err", np.nanmean(val_err).tolist()),
               ("test_err", np.nanmean(test_err).tolist())]

    for err, n_class in zip(val_err, range(params.n_classes)):
        log_err.append(("val_err_%s" % n_class, err.tolist()))

    for err, n_class in zip(test_err, range(params.n_classes)):
        log_err.append(("test_err_%s" % n_class, err.tolist()))

    # print rmse
    logger.info("estimator rmse:")
    print_metric(log_err, False)

    # JSON log
    logger.debug("__log__:%s"
                 % json.dumps(dict([("n_epoch", n_epoch)] + log_err)))

    # save best model based on validation rmse
    if np.nanmean(val_err) < best_rmse:
        best_rmse = np.nanmean(val_err)
        logger.info("best validation average rmse: %.3f" % best_rmse)

        save_state(params, n_epoch, ["best_acc", "best_rmse"],
                   [best_acc, best_rmse], "best_rmse",
                   ["estimator"], [estimator], ["optimizer"], [optimizer])

    logger.info("end of epoch %i.\n" % n_epoch)

    # update learning rate if milestones are set
    if n_epoch in params.lr_milestones:
        # basic schedule which divides the current learning rate by 10
        schedule_lr(optimizer)
