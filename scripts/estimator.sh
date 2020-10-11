#!/usr/bin/env bash
#estimators
python estimator.py --user adversary --estimator_output prnu_lp --expanded_cms False --n_epochs 1

python estimator.py --user examiner --estimator_output prnu_lp --expanded_cms False --n_epochs 1

python estimator.py --user examiner --estimator_output prnu_lp --expanded_cms True --n_epochs 1
