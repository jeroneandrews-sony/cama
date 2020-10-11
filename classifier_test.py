import argparse
import json
import os

import numpy as np
import torch
from data.config import centre_crop_size, preproc_img_dir
from src.evaluation import Classifier_Blur_Evaluator as Classifier_Evaluator
from src.loader import test_loader, train_loader
from src.models.utils import construct_classifier, construct_prnu_estimator
from src.training import Classifier_Trainer
from src.utils import (bool_flag, get_dataset_mean_std, get_optimizer,
                       get_valid_input_names, glob_get_path, initialize_exp,
                       print_metric, reload_params, reload_state_dict,
                       save_state, schedule_lr)
from torchsummary import summary

# parse parameters
parser = argparse.ArgumentParser(description="Camera model attribution\
                                 classifier training")

# main parameters
parser.add_argument("--user", type=str, default="adversary",
                    help="Dataset split to use {'adversary', 'examiner'}")
parser.add_argument("--ptc_sz", type=int, default=64,
                    help="Patch size either width or height (images are\
                    assumed to be square)")
parser.add_argument("--ptc_fm", type=int, default=3,
                    help="Number of input feature maps (rgb channels)")
parser.add_argument("--expanded_cms", type=bool_flag, default=False,
                    help="Training with an expanded set of camera models (only valid if user == examiner)")

# network architecture
parser.add_argument("--standardize", type=bool_flag, default=False,
                    help="standardize input channels")
parser.add_argument("--clf_input", type=str, default="con_conv_low",
                    help="Classifier mode {'prnu', 'prnu_lp', 'rgb', 'rgb+prnu_lp'}")
parser.add_argument("--clf_architecture", type=str, default="resnet18",
                    help="Classifier architecture to use {'vgg11', 'vgg13', 'vgg16', 'vgg19', 'resnet18', 'resnet34', 'resnet50', 'densenet40', 'densenet100'}")
parser.add_argument("--drop_rate", type=float, default=0.5,
                    help="Dropout in the classifier")
parser.add_argument("--efficient", type=bool_flag, default=False,
                    help="Memory efficient (but slower) training for DenseNet models")

# training parameters
parser.add_argument("--batch_size", type=int, default=128,
                    help="Batch size for training data")
parser.add_argument("--test_batch_size", type=int, default=16,
                    help="Batch size for validation / test data")
parser.add_argument("--rnd_crops", type=bool_flag, default=False,
                    help="If False crops are extracted from a non-overlapping grid; if True crops are extracted randomly")
parser.add_argument("--n_epochs", type=int, default=90,
                    help="Total number of epochs")
parser.add_argument("--n_samples_per_epoch", type=int, default=150000,
                    help="Number of training samples per epoch")
parser.add_argument("--optimizer", type=str, default="sgd,lr=0.1,weight_decay=0.0005,momentum=0.9,nesterov=True",
                    help="Classifier optimizer. E.g. {'sgd,lr=0.1,weight_decay=0.0005,momentum=0.9,nesterov=True', 'adagrad,lr=0.1,lr_decay=0.05'}")
parser.add_argument("--save_opt", type=bool_flag, default=True,
                    help="Save optimizer?")
parser.add_argument("--lr_milestones", nargs='+', type=int, default=[45, 68],
                    help="Learning rate divided by 10 schedule")
# loaders / gpus
parser.add_argument("--n_workers", type=int, default=10,
                    help="Number of workers per data loader")
parser.add_argument("--pin_memory", type=bool_flag, default=True,
                    help="Pin memory of data loaders")
parser.add_argument("--gpu_devices", nargs='+', type=int, default=[0],
                    help="Which gpu ids to use")

# reload
parser.add_argument("--est_prnu_reload", type=str, default="",
                    help="Path to a pretrained prnu estimator")
parser.add_argument("--reload", type=str, default="/home/jandrews/Documents/new_models/classifiers/adversary/resnet18/con_conv/train/180520_063455_wma6ao7saz/best.pth",
                    help="Path to a pretrained classifier (and optimizer if saved)")
parser.add_argument("--resume", type=bool_flag, default=False,
                    help="Resume training from the pretrained classifier's last checkpoint?")

# debug
parser.add_argument("--n_debug_samples", type=int, default=0,
                    help="Debug mode samples n_debug_samples (0 to turn off)")

params = parser.parse_args()

if centre_crop_size % params.ptc_sz != 0:
    assert rnd_crops, 'image size %i not divisible by patch size %i, with rnd_crops set to False. change rnd_crops to False if this behaviour is required.' % (centre_crop_size, params.ptc_sz)

assert params.user in ['adversary', 'examiner'], 'invalid "user". choose either "adversary" or "examiner"'
params.train_root = os.path.join(preproc_img_dir, params.user, 'train')  # path to training data folders (to images not prnus; path to prnus will be inferred from the path to images)
params.val_root = os.path.join(preproc_img_dir, params.user, 'validation')  # path to validation data folders (to images not prnus; path to prnus will be inferred from the path to images)
params.test_root = os.path.join(preproc_img_dir, 'test')  # path to test data folders (to images not prnus; path to prnus will be inferred from the path to images)
if params.expanded_cms:
    assert params.user == 'examiner', 'training with an expanded set of camera models only valid when user is "examiner"'

# specify which GPU(s) to use
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "%s" % ("".join(str(gpu_) + "," for gpu_ in params.gpu_devices))[:-1]
params.primary_gpu = 0


torch.backends.cudnn.benchmark = True

# # use glob to get full path name
# if params.reload: params.reload = glob_get_path(params.reload)
# params = reload_params(params.reload) # reload training params


# lower case
params.clf_input = params.clf_input.lower()
params.optimizer = params.optimizer.lower()
params.clf_architecture = params.clf_architecture.lower()

if params.est_prnu_reload:
    params.est_prnu_reload = glob_get_path(params.est_prnu_reload)
if 'prnu' not in params.clf_input:
    params.est_prnu_reload = ''


# check parameters
assert torch.cuda.is_available(), 'torch cuda not available'
assert params.clf_input in get_valid_input_names(), 'invalid classifier input mode'
assert not params.reload or os.path.isfile(params.reload), 'params reload file given, but does not exist'

assert not params.est_prnu_reload or os.path.isfile(params.est_prnu_reload), 'params prnu estimator reload file given, but does not exist'

if 'prnu' in params.clf_input:
    assert os.path.isfile(params.est_prnu_reload), 'classifier uses prnu, but prnu estimator does not exist'

if params.n_debug_samples:
    params.n_samples_per_epoch = params.n_debug_samples

# load train / validation / test dataset
train_dataset, n_classes = train_loader(params)
params.n_classes = n_classes

validation_dataset, n_classes_valid = test_loader(params, params.val_root)
assert n_classes == n_classes_valid, 'number of training and validation classes differ'

test_dataset, n_classes_test = test_loader(params, params.test_root)
assert n_classes == n_classes_test, 'number of training and test classes differ'

# construct the models
est_prnu_params = reload_params(params.est_prnu_reload) if params.est_prnu_reload else None

est_prnu = construct_prnu_estimator(est_prnu_params) if est_prnu_params is not None else None

# get channel mean and std for data preprocessing
params.rgb_mu, params.rgb_std, params.prnu_mu, params.prnu_std = get_dataset_mean_std(params)

# construct the model
classifier = construct_classifier(params)

# # construct optimizer
# optimizer = get_optimizer(classifier, params.optimizer)

# reload parameters if file given and define epoch number
reload_state_dict(params, params.reload, False, ['classifier'], [classifier], [], [])
if est_prnu is not None:
    reload_state_dict(None, params.est_prnu_reload, False, ['estimator'], [est_prnu], [], [])

# move model to DataParallel
if len(params.gpu_devices) > 1:
    classifier = torch.nn.DataParallel(classifier, [gpu_ for gpu_ in range(len(params.gpu_devices))])
    est_prnu = torch.nn.DataParallel(est_prnu, gpu_list) if est_prnu is not None else None

# move model to cuda
classifier = classifier.to(device=params.primary_gpu).eval()
for p_ in classifier.parameters():
    p_.requires_grad = False

if est_prnu is not None:
    est_prnu = est_prnu.to(device=params.primary_gpu).eval()
    for p_ in est_prnu.parameters():
        p_.requires_grad = False


evaluator = Classifier_Evaluator(params.clf_input)

# classifier in train mode
classifier.eval()

# if prnu estimator set to eval mode
if est_prnu is not None:
    est_prnu.eval()

test_acc, test_loss = evaluator.classifier_accuracy(classifier, est_prnu, test_dataset, params)

# log classifier validation and test accuracy
log_acc = [('test_acc', np.nanmean(test_acc).tolist())]

for acc, n_class in zip(test_acc, range(params.n_classes)):
    log_acc.append(('test_acc_%s' % n_class, acc.tolist()))

log_xent = [('test_loss', np.nanmean(test_loss).tolist())]
for err, n_class in zip(test_loss, range(params.n_classes)):
    log_xent.append(('test_loss_%s' % n_class, err.tolist()))

# print classifier accuracy
print('classifier accuracy:')
print(log_acc)

# print xent loss
print('classifier loss:')
print(log_xent)
print(hola)
