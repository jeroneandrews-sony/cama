#!/usr/bin/env bash

# train Cama
python train.py --dis_input con_conv \
--clf_low_reload models/classifiers/adversary/resnet18/prnu_lp_low/train/**/best.pth \
--clf_high_reload models/classifiers/adversary/resnet18/prnu_lp/train/**/best.pth \
--est_reload models/estimators/adversary/prnu_lp/train/**/best_rmse.pth

# train SpoC
python train.py --dis_input rgb+finite_difference \
--adv_loss_schedule 0.3 0 --clf_low_loss_schedule 0.0 0 --clf_high_loss_schedule 0.01 0 \
--clf_high_reload models/classifiers/adversary/resnet18/rgb+finite_difference/train/**/best.pth

# train MISL
python train.py --dis_input con_conv --clf_low_loss_schedule 0.0 0 --clf_high_loss_schedule 0.01 0 \
--clf_high_reloadmodels/classifiers/adversary/resnet18/con_conv/train/**/best.pth


