import argparse
import os
import sys

import models
import torch
from data.config import centre_crop_size, preproc_img_dir
from src.loader import dataset_loader
from src.models.utils import Classifier, Estimator, Generator
from src.testing import GAN_Tester
from src.utils import (bool_flag, glob_get_path, initialize_test_exp,
                       reload_params, reload_state_dict)

# parse parameters
parser = argparse.ArgumentParser(description="Conditional GAN testing for "
                                 "camera model anonymization")

# main parameters
parser.add_argument("--ptc_fm", type=int, default=3,
                    help="Number of input feature maps (channels)")

# testing parameters
parser.add_argument("--test_batch_size", type=int, default=16,
                    help="Batch size (testing)")
parser.add_argument("--in_dist", type=bool_flag, default=True,
                    help="Are the test images in-distribution, i.e. captured "
                    "by camera models known to conditional GAN?")
parser.add_argument("--comp_distortion", type=bool_flag, default=False,
                    help="Compute the distortion of the generator's outputs?")
parser.add_argument("--quantize", type=bool_flag, default=False,
                    help="Perform quantization?")

# visualization parameters
parser.add_argument("--visualize", type=int, default=0,
                    help="Number of transformation visualizations to save")
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
                    help="Path to a pre-trained conditional GAN")
parser.add_argument("--clf_low_reload", type=str, default="",
                    help="Path to a pre-trained low-frequency classifier")
parser.add_argument("--clf_high_reload", type=str, default="",
                    help="Path to a pre-trained high-frequency classifier")
parser.add_argument("--est_reload", type=str, default="",
                    help="Path to a pre-trained PRNU estimator")
parser.add_argument("--transformed_imgs_reload", type=str, default="",
                    help="Path to pre-computed transformed images '.pth' file")

params = parser.parse_args()

# reload necessary parameters
assert params.reload, "no conditional GAN reload path was given"
params.reload = glob_get_path(params.reload)
assert os.path.isfile(params.reload), "conditional GAN reload file does not "
"exist"

# set user
params.user = "adversary"

# set debug mode to "False" (in the dataset loader)
params.debug = False

# set the image spatial size
params.centre_crop_size = centre_crop_size

# set data path
params.test_root = os.path.join(preproc_img_dir,
                                "test" if params.in_dist else "test_outdist")


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

# get some necessary additional params from the generator file
gen_params = reload_params(params.reload)
params.n_classes = gen_params.n_classes
params.gen_input = gen_params.gen_input
params.dump_path = gen_params.dump_path

# load test dataset
test_dataset, n_classes_test, n_samples_test = dataset_loader(params, False)\
    .test(params.test_root)
params.n_classes_test = n_classes_test
params.n_samples_test = n_samples_test

# build the models
if params.clf_low_reload:
    params.clf_low_reload = glob_get_path(params.clf_low_reload)
    assert os.path.isfile(params.clf_low_reload), "(low-frequency) classifier "
    "reload file does not exist"
    clf_low_params = reload_params(params.clf_low_reload)
    params.clf_low_input = clf_low_params.clf_input
    clf_low = Classifier(clf_low_params)
else:
    params.clf_low_input, clf_low_params, clf_low = None, None, None

if params.clf_high_reload:
    params.clf_high_reload = glob_get_path(params.clf_high_reload)
    assert os.path.isfile(params.clf_high_reload), "(high-frequency) "
    "classifier reload file does not exist"
    clf_high_params = reload_params(params.clf_high_reload)
    params.clf_high_input = clf_high_params.clf_input
    if "prnu_lp" in clf_high_params.clf_input:
        assert os.path.isfile(params.est_reload), "estimator reload file "
        "does not exist and is required"
    clf_high = Classifier(clf_high_params)
else:
    params.clf_high_input, clf_high_params, clf_high = None, None, None

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

gen = Generator(gen_params)
del gen_params

# reload states
if params.reload:
    reload_state_dict(None, params.reload, False,
                      ["generator"], [gen], [], [])

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
    if clf_low is not None:
        clf_low = torch.nn.DataParallel(clf_low, gpu_list)
    if clf_high is not None:
        clf_high = torch.nn.DataParallel(clf_high, gpu_list)
    if est is not None:
        est = torch.nn.DataParallel(est, gpu_list)

# move model to cuda
gen = gen.to(device=params.primary_gpu)
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


if params.transformed_imgs_reload:
    if "_ood" in params.transformed_imgs_reload:
        assert params.in_dist is False, "transformed images reload path [%s] "
        "corresponds to out-of-distribution data, but 'in_dist' is set to "
        "'True'"
        # if out-of-distribution data is used, then it's not possible to
        # compute the untargeted success rate
        params.comp_untgted = False
    else:
        params.comp_untgted = True
else:
    # if no precomputed transformed data is given
    if params.in_dist:
        params.comp_untgted = True
    else:
        params.comp_untgted = False

# initialize logger and dump params
logger = initialize_test_exp(params, "test")

if not params.transformed_imgs_reload:
    # transform the data in advance
    tester = GAN_Tester(gen, None, None, None, None, test_dataset, params)
    transformed_data = tester.transform_and_save()
else:
    # load precomputed transformed data
    transformed_data = torch.load(params.transformed_imgs_path)

# remove the last element from transformed_data (list), which corresponds
# to gen_input ("rgb" / "remosaic")
if len(transformed_data) > 3:
    transformed_data.pop()

# perform quantization (256 levels)
if params.quantize:
    transformed_data[0] = torch.round(transformed_data[0])
    transformed_data[0] = torch.clamp(transformed_data[0], 0, 255)

# if a classifier was trained on an expanded set of camera models, then we need
# to increment the class labels in the case of out-of-distribution data
# (out-of-distribution w.r.t. the classes known to the conditional GAN)
if ("_exp" in params.clf_low_reload) or ("_exp" in params.clf_high_reload):
    if "_ood" in params.transformed_imgs_reload:
        transformed_data[1] += params.n_classes
    # in this case, it's always possible to compute the untargeted success rate
    params.comp_untgted = True

tester = GAN_Tester(None, clf_low, clf_high, est, percept_model,
                    [transformed_data, test_dataset], params)

# commence testing
logger.info("start of testing ...")

log_psnr, log_lpips = [], []
if (params.comp_distortion) or (params.visualize > 0):
    log_psnr, log_lpips = tester.transform_distortion()

tester.transform_accuracy(log_psnr, log_lpips)

# end of testing
logger.info("end of testing.\n")
