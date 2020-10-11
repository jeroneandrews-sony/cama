#!/usr/bin/env bash

# evaluators for Cama
python classifier.py --user adversary --clf_architecture resnet18 --clf_input prnu_lp \
--est_reload models/estimators/adversary/prnu_lp/train/**/best_rmse.pth

python classifier.py --user adversary --clf_architecture resnet18 --clf_input prnu_lp_low \
--est_reload models/estimators/adversary/prnu_lp/train/**/best_rmse.pth

# evaluator for SpoC
python classifier.py --user adversary --clf_architecture resnet18 --clf_input rgb+finite_difference

# evaluator for MISL
python classifier.py --user adversary --clf_architecture resnet18 --clf_input con_conv


# non-interactive black-box classifiers with resnet-18 architectures
python classifier.py --user examiner --clf_architecture resnet18 --clf_input rgb

python classifier.py --user examiner --clf_architecture resnet18 --clf_input prnu_lp \
--est_reload /home/jandrews/Documents/new_models/estimators/examiner/prnu_lp/train/**/best_rmse.pth

python classifier.py --user examiner --clf_architecture resnet18 --clf_input con_conv

python classifier.py --user examiner --clf_architecture resnet18 --clf_input finite_difference

python classifier.py --user examiner --clf_architecture resnet18 --clf_input fixed_hpf

python classifier.py --user examiner --clf_architecture resnet18 --clf_input rgb+finite_difference


# non-interactive black-box classifiers with resnet-50 architectures
python classifier.py --user examiner --clf_architecture resnet50 --clf_input rgb

python classifier.py --user examiner --clf_architecture resnet50 --clf_input prnu_lp \
--est_reload /home/jandrews/Documents/new_models/estimators/examiner/prnu_lp/train/**/best_rmse.pth

python classifier.py --user examiner --clf_architecture resnet50 --clf_input con_conv

python classifier.py --user examiner --clf_architecture resnet50 --clf_input finite_difference

python classifier.py --user examiner --clf_architecture resnet50 --clf_input fixed_hpf

python classifier.py --user examiner --clf_architecture resnet50 --clf_input rgb+finite_difference

# non-interactive black-box classifiers with vgg-16 architectures
python classifier.py --user examiner --clf_architecture vgg16 --clf_input rgb

python classifier.py --user examiner --clf_architecture vgg16 --clf_input prnu_lp \
--est_reload /home/jandrews/Documents/new_models/estimators/examiner/prnu_lp/train/**/best_rmse.pth

python classifier.py --user examiner --clf_architecture vgg16 --clf_input con_conv

python classifier.py --user examiner --clf_architecture vgg16 --clf_input finite_difference

python classifier.py --user examiner --clf_architecture vgg16 --clf_input fixed_hpf

python classifier.py --user examiner --clf_architecture vgg16 --clf_input rgb+finite_difference


# non-interactive black-box classifiers with densenet-100 architectures
python classifier.py --user examiner --clf_architecture densenet100 --clf_input rgb

python classifier.py --user examiner --clf_architecture densenet100 --clf_input prnu_lp \
--est_reload /home/jandrews/Documents/new_models/estimators/examiner/prnu_lp/train/**/best_rmse.pth

python classifier.py --user examiner --clf_architecture densenet100 --clf_input con_conv

python classifier.py --user examiner --clf_architecture densenet100 --clf_input finite_difference

python classifier.py --user examiner --clf_architecture densenet100 --clf_input fixed_hpf

python classifier.py --user examiner --clf_architecture densenet100 --clf_input rgb+finite_difference


# non-interactive black-box classifiers with resnet-18 architectures (expanded camera models)
python classifier.py --user examiner --clf_architecture resnet18 --clf_input rgb --expanded_cms True

python classifier.py --user examiner --clf_architecture resnet18 --clf_input prnu_lp --expanded_cms True \
--est_reload /home/jandrews/Documents/new_models/estimators/examiner/prnu_lp/train/**/best_rmse.pth

python classifier.py --user examiner --clf_architecture resnet18 --clf_input con_conv --expanded_cms True

python classifier.py --user examiner --clf_architecture resnet18 --clf_input finite_difference --expanded_cms True

python classifier.py --user examiner --clf_architecture resnet18 --clf_input fixed_hpf --expanded_cms True

python classifier.py --user examiner --clf_architecture resnet18 --clf_input rgb+finite_difference --expanded_cms True


# non-interactive black-box classifiers with resnet-50 architectures (expanded camera models)
python classifier.py --user examiner --clf_architecture resnet50 --clf_input rgb --expanded_cms True

python classifier.py --user examiner --clf_architecture resnet50 --clf_input prnu_lp --expanded_cms True \
--est_reload /home/jandrews/Documents/new_models/estimators/examiner/prnu_lp/train/**/best_rmse.pth

python classifier.py --user examiner --clf_architecture resnet50 --clf_input con_conv --expanded_cms True

python classifier.py --user examiner --clf_architecture resnet50 --clf_input finite_difference --expanded_cms True

python classifier.py --user examiner --clf_architecture resnet50 --clf_input fixed_hpf --expanded_cms True

python classifier.py --user examiner --clf_architecture resnet50 --clf_input rgb+finite_difference --expanded_cms True

# non-interactive black-box classifiers with vgg-16 architectures (expanded camera models)
python classifier.py --user examiner --clf_architecture vgg16 --clf_input rgb --expanded_cms True

python classifier.py --user examiner --clf_architecture vgg16 --clf_input prnu_lp --expanded_cms True \
--est_reload /home/jandrews/Documents/new_models/estimators/examiner/prnu_lp/train/**/best_rmse.pth

python classifier.py --user examiner --clf_architecture vgg16 --clf_input con_conv --expanded_cms True

python classifier.py --user examiner --clf_architecture vgg16 --clf_input finite_difference --expanded_cms True

python classifier.py --user examiner --clf_architecture vgg16 --clf_input fixed_hpf --expanded_cms True

python classifier.py --user examiner --clf_architecture vgg16 --clf_input rgb+finite_difference --expanded_cms True


# non-interactive black-box classifiers with densenet-100 architectures (expanded camera models)
python classifier.py --user examiner --clf_architecture densenet100 --clf_input rgb --expanded_cms True

python classifier.py --user examiner --clf_architecture densenet100 --clf_input prnu_lp --expanded_cms True \
--est_reload /home/jandrews/Documents/new_models/estimators/examiner/prnu_lp/train/**/best_rmse.pth

python classifier.py --user examiner --clf_architecture densenet100 --clf_input con_conv --expanded_cms True

python classifier.py --user examiner --clf_architecture densenet100 --clf_input finite_difference --expanded_cms True

python classifier.py --user examiner --clf_architecture densenet100 --clf_input fixed_hpf --expanded_cms True

python classifier.py --user examiner --clf_architecture densenet100 --clf_input rgb+finite_difference --expanded_cms True
