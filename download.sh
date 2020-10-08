#!/usr/bin/env bash

# install cama requirements
pip install -r requirements.txt

# cama main directory
base_dir=${PWD}

# cama data directory
data_dir=${PWD}/data

# clone the perceptual similarity repository
percep_sim_dir=${PWD}/PerceptualSimilarity
git clone https://github.com/richzhang/PerceptualSimilarity.git
cd ${percep_sim_dir}
# install perceptual similarity requirements
pip install -r requirements.txt

# go to the cama data directory
cd ${data_dir}

# download the data
python download_dataset.py

# preprocess the data
python preprocess.py

# compute the dataset stats
python dataset_statistics.py

# go back to the cama main directory
cd ${base_dir}
