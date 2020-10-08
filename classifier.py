import argparse
import json
import os

import numpy as np
import torch
from data.config import centre_crop_size, preproc_img_dir
from src.evaluation import Classifier_Evaluator
from src.loader import dataset_loader
from src.models.utils import Classifier, Estimator
from src.training import Classifier_Trainer
from src.utils import (bool_flag, get_optimizer, get_valid_input_names,
                       glob_get_path, initialize_exp, print_metric,
                       reload_params, reload_state_dict, save_state,
                       schedule_lr)
from torchsummary import summary

# parse parameters
parser = argparse.ArgumentParser(description="Camera model attribution "
                                 "classifier training")

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
parser.add_argument("--clf_input", type=str, default="rgb",
                    help="Classifier input (prnu_lp / rgb / con_conv / "
                    "prnu_lp_low / prnu_lp_low+prnu_lp)")
parser.add_argument("--clf_architecture", type=str, default="resnet18",
                    help="Classifier architecture (vgg11 / vgg13 / vgg16 / "
                    "vgg19 / resnet18 / resnet34 / resnet50 / densenet40 / "
                    "densenet100)")
parser.add_argument("--drop_rate", type=float, default=0.5,
                    help="Dropout in the classifier")
parser.add_argument("--efficient", type=bool_flag, default=False,
                    help="Memory efficient (but slower) training for DenseNet "
                    "models")

# training parameters
parser.add_argument("--batch_size", type=int, default=128,
                    help="Batch size (training)")
parser.add_argument("--test_batch_size", type=int, default=16,
                    help="Batch size (validation / testing)")
parser.add_argument("--rnd_crops", type=bool_flag, default=False,
                    help="Extract patches randomly (True) or from a "
                    "non-overlapping grid (False)")
parser.add_argument("--n_epochs", type=int, default=90,
                    help="Total number of epochs")
parser.add_argument("--n_samples_per_epoch", type=int, default=150000,
                    help="Number of training samples per epoch")
parser.add_argument("--optimizer", type=str, default="sgd,lr=0.1,"
                    "weight_decay=0.0005,momentum=0.9,nesterov=True",
                    help="Classifier optimizer (sgd,lr=0.1,"
                    "weight_decay=0.0005,momentum=0.9,nesterov=True / "
                    "adagrad,lr=0.1,lr_decay=0.05")
parser.add_argument("--save_opt", type=bool_flag, default=True,
                    help="Save optimizer")
parser.add_argument("--lr_milestones", nargs='+', type=int, default=[45, 68],
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
                    help="Path to a pre-trained classifier (and optimizer if "
                    "saved)")
parser.add_argument("--est_reload", type=str, default="",
                    help="Path to a a pre-trained PRNU estimator (trained "
                    "with estimator.py)")
parser.add_argument("--resume", type=bool_flag, default=False,
                    help="Resume training")

# debug
parser.add_argument("--debug", type=bool_flag, default=False,
                    help="Debug mode (only use a subset of the available "
                    "data)")

params = parser.parse_args()

# if resume, reload necessary parameters
if params.resume:
    assert params.reload, "tried to resume training, but no reload path was "
    "given"
    params.reload = glob_get_path(params.reload)
    reload_path = params.reload
    assert os.path.isfile(reload_path), "classifier reload file does not exist"
    params = reload_params(params.reload)
    params.resume = True
    params.reload = reload_path
    del reload_path
else:
    params.n_epoch_start = 1

assert not params.reload or os.path.isfile(params.reload), "classifier reload "
"file does not exist"

# set model type
params.model_type = "classifiers"

# lower case
params.clf_input = params.clf_input.lower()
params.optimizer = params.optimizer.lower()
params.clf_architecture = params.clf_architecture.lower()

if centre_crop_size % params.ptc_sz != 0:
    assert params.rnd_crops, "(when rnd_crops is False) the spatial size of "
    "an image (%ix%i) must be divisible by " % (centre_crop_size,
                                                centre_crop_size)
    "the spatial size of the patch window (%ix%i)" % (params.ptc_sz,
                                                      params.ptc_sz)

assert params.user in ["adversary", "examiner"], "user must be adversary or "
"examiner"
assert params.clf_input in get_valid_input_names(), "invalid classifier input"

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

# get estimator path (if necessary)
if "prnu_lp" not in params.clf_input:
    params.est_reload = ""
else:
    params.est_reload = glob_get_path(params.est_reload)
    assert os.path.isfile(params.est_reload), "estimator reload file "
    "does not exist and is required"

# load train / validation / test dataset
train_dataset, n_classes, _ = dataset_loader(params, True).train
validation_dataset, n_classes_valid, _ = dataset_loader(params, False).test(
    params.val_root)
test_dataset, n_classes_test, _ = dataset_loader(params, False).test(
    params.test_root)
params.n_classes = n_classes
assert n_classes == n_classes_valid == n_classes_test, "the number of "
"training classes must equal the number of validation / testing classes"

# build the models / optimizer
if params.est_reload:
    estimator_params = reload_params(params.est_reload)
    estimator = Estimator(estimator_params)
else:
    estimator_params, estimator = None, None

classifier = Classifier(params)
optimizer = get_optimizer(classifier, params.optimizer)

# reload states
if params.reload and params.resume:
    reload_state_dict(params, params.reload, params.resume,
                      ["classifier"], [classifier],
                      ["optimizer"], [optimizer])
elif params.reload:
    reload_state_dict(params, params.reload, params.resume,
                      ["classifier"], [classifier],
                      [], [])
if estimator is not None:
    reload_state_dict(None, params.est_reload, False,
                      ["estimator"], [estimator],
                      [], [])

# move model to DataParallel
if len(params.gpu_devices) > 1:
    gpu_list = [gpu_ for gpu_ in range(len(params.gpu_devices))]
    classifier = torch.nn.DataParallel(classifier, gpu_list)
    if estimator is not None:
        estimator = torch.nn.DataParallel(estimator, gpu_list)

# move model to cuda
classifier = classifier.to(device=params.primary_gpu)

if estimator is not None:
    estimator = estimator.to(device=params.primary_gpu).eval()
    for p_ in estimator.parameters():
        p_.requires_grad = False

# best accuracy
best_acc = -1e12
best_loss = 1e12

# initialize logger and dump params
logger = initialize_exp(params,
                        model_type=params.model_type
                        if not params.expanded_cms else params.model_type +
                        "_exp")

# log architectural details / model parameters using torchsummary's summary
if not params.resume:
    logger.info(summary(classifier,
                        (2 * params.ptc_fm if ("+prnu_lp" in params.clf_input)
                         else params.ptc_fm, params.ptc_sz, params.ptc_sz)))
    logger.info(optimizer)

trainer = Classifier_Trainer(params.clf_input)
evaluator = Classifier_Evaluator(params.clf_input)

# commence training
for n_epoch in range(params.n_epoch_start, params.n_epochs + 1):

    logger.info("start of epoch %i..." % n_epoch)

    # initialize stats
    stats = []

    # classifier in train mode
    classifier.train()

    # estimator in eval mode
    if estimator is not None:
        estimator.eval()

    # classifier train
    trainer.train(n_epoch, classifier, estimator, optimizer, train_dataset,
                  params, stats)

    # classifier in eval mode
    classifier.eval()

    # evaluate classifier on validation / test data
    val_acc, val_loss = evaluator.classifier_accuracy(classifier, estimator,
                                                      validation_dataset,
                                                      params)
    test_acc, test_loss = evaluator.classifier_accuracy(classifier, estimator,
                                                        test_dataset,
                                                        params)

    # log classifier validation / test accuracy
    log_acc = [("val_acc", np.nanmean(val_acc).tolist()),
               ("test_acc", np.nanmean(test_acc).tolist())]
    for acc, n_class in zip(val_acc, range(params.n_classes)):
        log_acc.append(("val_acc_%s" % n_class, acc.tolist()))

    for acc, n_class in zip(test_acc, range(params.n_classes)):
        log_acc.append(("test_acc_%s" % n_class, acc.tolist()))

    log_xent = [("val_loss", np.nanmean(val_loss).tolist()),
                ("test_loss", np.nanmean(test_loss).tolist())]
    for err, n_class in zip(val_loss, range(params.n_classes)):
        log_xent.append(("val_loss_%s" % n_class, err.tolist()))
    for err, n_class in zip(test_loss, range(params.n_classes)):
        log_xent.append(("test_loss_%s" % n_class, err.tolist()))

    # print classifier accuracy
    logger.info("classifier accuracy:")
    print_metric(log_acc, True)

    # print cross-entropy loss
    logger.info("classifier loss:")
    print_metric(log_xent, False)

    # JSON log
    logger.debug("__log__:%s" % json.dumps(dict([("n_epoch", n_epoch)]
                                                + log_acc + log_xent)))

    # save best model based on validation accuracy
    if ((np.nanmean(val_acc) >= best_acc) and
            (np.nanmean(val_loss) < best_loss)):
        best_acc = np.nanmean(val_acc)
        best_loss = np.nanmean(val_loss)
        logger.info("best validation average accuracy: %.3f"
                    % (best_acc * 100))
        save_state(params, n_epoch, ["best_acc", "best_loss"],
                   [best_acc, best_loss], "best", ["classifier"], [classifier],
                   ["optimizer"], [optimizer])

    logger.info("end of epoch %i.\n" % n_epoch)

    # update learning rate if milestones are set
    if n_epoch in params.lr_milestones:
        # basic schedule which divides the current learning rate by 10
        schedule_lr(optimizer)
