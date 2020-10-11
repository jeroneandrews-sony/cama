import argparse
import os
import sys

import models
import torch
from data.config import centre_crop_size, preproc_img_dir
from src.evaluation import GAN_Evaluator
from src.loader import dataset_loader
from src.models.utils import Classifier, Discriminator, Estimator, Generator
from src.training import GAN_Trainer
from src.utils import (bool_flag, get_valid_input_names, glob_get_path,
                       initialize_exp, reload_params, reload_state_dict)

# parse parameters
parser = argparse.ArgumentParser(description="Conditional GAN training for "
                                 "camera model anonymization")

# main parameters
parser.add_argument("--ptc_sz", type=int, default=64,
                    help="Patch width/height")
parser.add_argument("--ptc_fm", type=int, default=3,
                    help="Number of input feature maps (channels)")

# generator architecture
parser.add_argument("--gen_input", type=str, default="rgb",
                    help="Generator input (rgb / remosaic)")
parser.add_argument("--ngf", type=int, default=64,
                    help="Number of feature maps in the generator's first and "
                    "last convolutional layers")
parser.add_argument("--n_blocks", type=int, default=2,
                    help="Number of residual blocks in the generator")

# discriminator architecture
parser.add_argument("--dis_input", type=str, default="con_conv",
                    help="Classifier input (prnu_lp / rgb / con_conv / "
                    "prnu_lp_low+prnu_lp)")
parser.add_argument("--ndf", type=int, default=64,
                    help="Number of feature maps in the discriminator's first "
                    "convolutional layer")
parser.add_argument("--use_lsgan", type=bool_flag, default=True,
                    help="Least squares GAN")
parser.add_argument("--dis_type", type=str, default="patch",
                    help="Discriminator type (patch / pixel)")
parser.add_argument("--n_dis_layers", type=int, default=2,
                    help="Number of discriminator layers (only used if the "
                    "discriminator is a patch discriminator)")
parser.add_argument("--dis_drop_rate", type=float, default=0.0,
                    help="Dropout in the discriminator")

# training parameters
parser.add_argument("--pixel_loss", type=str, default="l1",
                    help="Pixel-wise loss (l1 / l2)")
parser.add_argument("--pixel_loss_schedule", nargs='+', type=float,
                    default=[10.0, 0],
                    help="First argument: pixel loss feedback coefficient (0 "
                    "to disable). Second argument: epochs to progressively "
                    "increase the pixel loss coefficient (0 to disable).")
parser.add_argument("--adv_loss_schedule", nargs='+', type=float,
                    default=[1.0, 0],
                    help="First argument: adversarial loss feedback "
                    "coefficient (0 to disable). Second argument: epochs to "
                    "progressively increase the adversarial loss coefficient "
                    "(0 to disable).")
parser.add_argument("--clf_low_loss_schedule", nargs='+', type=float,
                    default=[0.005, 0],
                    help="First argument: low-frequency classifier loss feedback "
                    "coefficient (0 to disable). Second argument: epochs to "
                    "progressively increase the rgb classifier loss "
                    "coefficient (0 to disable).")
parser.add_argument("--clf_high_loss_schedule", nargs='+', type=float,
                    default=[0.005, 0],
                    help="First argument: high-frequency classifier loss feedback "
                    "coefficient (0 to disable). Second argument: epochs to "
                    "progressively increase the noise classifier loss "
                    "coefficient (0 to disable).")
parser.add_argument("--batch_size", type=int, default=32,
                    help="Batch size (training)")
parser.add_argument("--test_batch_size", type=int, default=16,
                    help="Batch size (validation)")
parser.add_argument("--rnd_crops", type=bool_flag, default=False,
                    help="Extract patches randomly (True) or from a "
                    "non-overlapping grid (False)")
parser.add_argument("--n_epochs", type=int, default=2,
                    help="Total number of epochs")
parser.add_argument("--n_samples_per_epoch", type=int, default=10000,
                    help="Number of training samples per epoch")
parser.add_argument("--gen_optimizer", type=str, default="adam,lr=0.0002",
                    help="Generator optimizer (adam,lr=0.0002)")
parser.add_argument("--dis_optimizer", type=str, default="adam,lr=0.0002",
                    help="Discriminator optimizer (adam,lr=0.0002)")
parser.add_argument("--save_opt", type=bool_flag, default=True,
                    help="Save optimizers")
parser.add_argument("--lr_milestones", nargs='+', type=int, default=[],
                    help="Epochs to divide learning rate by 10")

# visualization parameters
parser.add_argument("--padding", type=int, default=20,
                    help="Amount of padding (in pixels) between images in a "
                    "single plot")

# loaders / gpus
parser.add_argument("--n_workers", type=int, default=8,
                    help="Number of workers per data loader")
parser.add_argument("--pin_memory", type=bool_flag, default=True,
                    help="Pin memory of data loaders")
parser.add_argument("--gpu_devices", nargs='+', type=int, default=[0],
                    help="Which gpu devices to use")

# reload
parser.add_argument("--reload", type=str, default="",
                    help="Path to a pre-trained conditional GAN (and "
                    "optimizer if saved)")
parser.add_argument("--clf_low_reload", type=str, default="",
                    help="Path to a pre-trained low-frequency classifier "
                    "(trained with classifier.py)")
parser.add_argument("--clf_high_reload", type=str, default="",
                    help="Path to a pre-trained high-frequency classifier "
                    "(trained with classifier.py)")
parser.add_argument("--est_reload", type=str, default="",
                    help="Path to a pre-trained PRNU estimator (trained with "
                    "estimator.py)")
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
    assert os.path.isfile(reload_path), "conditional GAN reload file does not "
    "exist"
    params = reload_params(params.reload)
    params.resume = True
    params.reload = reload_path
    del reload_path
else:
    params.n_epoch_start = 1
    params.n_iters_done = 0

assert not params.reload or os.path.isfile(params.reload), "conditional GAN "
"reload file does not exist"

# set model type
params.model_type = "gans"
# set user
params.user = "adversary"

# lower case
params.dis_optimizer = params.dis_optimizer.lower()
params.gen_optimizer = params.gen_optimizer.lower()
params.dis_type = params.dis_type.lower()
params.gen_input = params.gen_input.lower()
params.dis_input = params.dis_input.lower()

if centre_crop_size % params.ptc_sz != 0:
    assert params.rnd_crops, "(when rnd_crops is False) the spatial size of "
    "an image (%ix%i) must be divisible by " % (centre_crop_size,
                                                centre_crop_size)
    "the spatial size of the patch window (%ix%i)" % (params.ptc_sz,
                                                      params.ptc_sz)

# set data paths
params.train_root = os.path.join(preproc_img_dir, params.user, "train")
params.val_root = os.path.join(preproc_img_dir, params.user, "validation")

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

# add PerceptualSimilarity modules to path so as to import
sys.path.insert(0, os.path.join(os.getcwd(), "PerceptualSimilarity"))

# check parameters
assert params.gen_input in ["rgb", "remosaic"], "invalid generator input (rgb "
"/ remosaic)"
assert params.dis_type in ["patch", "pixel"], "invalid discriminator type "
"(patch / pixel)"
assert params.dis_input in get_valid_input_names(mode="all"), "invalid "
"discriminator input"
assert params.pixel_loss in ["l1", "l2"], "invalid pixel loss (l1 / l2)"

if "prnu_lp" in params.dis_input:
    assert os.path.isfile(params.est_reload), "discriminator with input "
    "%s " % params.dis_input
    "requires a prnu estimator, but prnu estimator reload file does not "
    "exist"

assert params.pixel_loss_schedule[0] >= 0, "first argument of the pixel loss "
"feedback must be greater than or equal to 0 (0 to disable)"
assert params.adv_loss_schedule[0] >= 0, "first argument of the adversarial "
"loss feedback must be greater than or equal to 0 (0 to disable)"
assert params.clf_low_loss_schedule[0] >= 0, "first argument of the "
"low-frequency classifier loss feedback must be greater than or equal to 0 "
"(0 to disable)"
assert params.clf_high_loss_schedule[0] >= 0, "first argument of the "
"high-frequency classifier loss feedback must be greater than or equal to 0 "
"(0 to disable)"

assert params.pixel_loss_schedule[1] >= 0, "second argument of the pixel loss "
"feedback must be greater than or equal to 0 (0 to disable)"
assert params.adv_loss_schedule[1] >= 0, "second argument of the adversarial "
"loss feedback must be greater than or equal to 0 (0 to disable)"
assert params.clf_low_loss_schedule[1] >= 0, "second argument of the "
"low-frequency classifier loss feedback must be greater than or equal to 0 "
"(0 to disable)"
assert params.clf_high_loss_schedule[1] >= 0, "second argument of the "
"high-frequency classifier loss feedback must be greater than or equal to 0 "
"(0 to disable)"

# load train / validation dataset
train_dataset, n_classes, _ = dataset_loader(params, True).train
validation_dataset, n_classes_valid, _ = dataset_loader(params, False).test(
    params.val_root)
params.n_classes = n_classes
assert n_classes == n_classes_valid, "the number of training classes must "
"equal the number of validation classes"

# if resuming, calculate total iterations already completed
if params.resume:
    params.n_iters_done = params.n_epoch_start * len(train_dataset)

# convert the epoch-based schedules to iteration-based schedules
params.pixel_loss_schedule[1] *= len(train_dataset)
params.adv_loss_schedule[1] *= len(train_dataset)
params.clf_low_loss_schedule[1] *= len(train_dataset)
params.clf_high_loss_schedule[1] *= len(train_dataset)

# build the models
if params.clf_low_reload:
    params.clf_low_reload = glob_get_path(params.clf_low_reload)
    assert os.path.isfile(params.clf_low_reload), "(low-frequency) classifier "
    "reload file does not exist"
    clf_low_params = reload_params(params.clf_low_reload)
    assert clf_low_params.clf_input in get_valid_input_names(mode="low"), \
        "(low-frequency) classifier input must be in %s" \
        % get_valid_input_names(mode="low")
    params.clf_low_input = clf_low_params.clf_input
    clf_low = Classifier(clf_low_params)
else:
    params.clf_low_input, clf_low_params, clf_low = None, None, None
    params.clf_low_loss_schedule = [0., 0.]

if params.clf_high_reload:
    params.clf_high_reload = glob_get_path(params.clf_high_reload)
    assert os.path.isfile(params.clf_high_reload), "(high-frequency) "
    "classifier reload file does not exist"
    clf_high_params = reload_params(params.clf_high_reload)
    assert clf_high_params.clf_input in get_valid_input_names(mode="high"), \
        "(high-frequency) classifier input must be in %s" \
        % get_valid_input_names(mode="high")
    params.clf_high_input = clf_high_params.clf_input
    if "prnu_lp" in clf_high_params.clf_input:
        assert os.path.isfile(params.est_reload), "estimator reload file "
        "does not exist and is required"
    clf_high = Classifier(clf_high_params)
else:
    params.clf_high_input, clf_high_params, clf_high = None, None, None
    params.clf_high_loss_schedule = [0., 0.]

if params.est_reload:
    params.est_reload = glob_get_path(params.est_reload)
    assert os.path.isfile(params.est_reload), "estimator reload file "
    "does not exist"
    est_params = reload_params(params.est_reload)
    try:
        params.estimator_output = est_params.estimator_output
    except AttributeError:
        params.estimator_output = None
    est = Estimator(est_params)
else:
    params.estimator_output, est_params, est = None, None, None

gen, dis = Generator(params), None
if params.adv_loss_schedule[0] > 0:
    dis = Discriminator(params)

# reload states
if params.reload:
    reload_state_dict(params, params.reload, params.resume,
                      ["generator"] + (["discriminator"]
                                       if dis is not None else []),
                      [gen] + ([dis] if dis is not None else []),
                      [], [])

if clf_low is not None:
    reload_state_dict(None, params.clf_low_reload, False,
                      ["classifier"], [clf_low], [], [])

if clf_high is not None:
    reload_state_dict(None, params.clf_high_reload, False,
                      ["classifier"], [clf_high], [], [])

if est is not None:
    reload_state_dict(None, params.est_reload, False,
                      ["estimator"], [est], [], [])

# move model to DataParallel
if len(params.gpu_devices) > 1:
    gpu_list = [gpu_ for gpu_ in range(len(params.gpu_devices))]
    gen = torch.nn.DataParallel(gen, gpu_list)
    if dis is not None:
        dis = torch.nn.DataParallel(dis, gpu_list)
    if clf_low is not None:
        clf_low = torch.nn.DataParallel(clf_low, gpu_list)
    if clf_high is not None:
        clf_high = torch.nn.DataParallel(clf_high, gpu_list)
    if est is not None:
        est = torch.nn.DataParallel(est, gpu_list)

# move model to cuda
gen = gen.to(device=params.primary_gpu)
if dis is not None:
    dis = dis.to(device=params.primary_gpu)
if clf_low is not None:
    clf_low = clf_low.to(device=params.primary_gpu).eval()
if clf_high is not None:
    clf_high = clf_high.to(device=params.primary_gpu).eval()
if est is not None:
    est = est.to(device=params.primary_gpu).eval()

# construct lpips model for evaluating the distortion of the generated images
# (using multiple gpu_ids causes an error, most likely an inconsistency with
# pytorch versions)
percept_model = models.PerceptualLoss(model="net-lin", net="alex",
                                      use_gpu=True,
                                      gpu_ids=[params.primary_gpu])

# initialize logger and dump params
logger = initialize_exp(params, model_type=params.model_type)

trainer = GAN_Trainer(gen, dis, clf_low, clf_high, est, train_dataset, params)
evaluator = GAN_Evaluator(gen, clf_low, clf_high, est, percept_model,
                          validation_dataset, params)

# commence training
for n_epoch in range(params.n_epoch_start, params.n_epochs + 1):

    logger.info("start of epoch %i..." % n_epoch)

    # gan train
    trainer.gan_train(n_epoch)

    # evaluate gan on validation data / log gan evaluation metrics
    to_log = evaluator.evaluate(n_epoch)

    # save best model based on validation metrics
    trainer.save_best_periodic(to_log)

    logger.info("end of epoch %i.\n" % n_epoch)
