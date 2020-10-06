import os

# CHANGE THE BELOW PARAMETERS IF YOU WISH
# centrally crop images to a common resolution
centre_crop_size = 512
# proportion of in-distribution data used for training
train_prop = 0.9
# extract and save prnu images
prnu_extract = True
# remosaic and save remosaiced images
remosaic = True

# DO NOT CHANGE THE BELOW PARAMETERS
# base directory for saving the dataset
base_dir = os.path.join(os.path.dirname(__file__), "dataset")
# dataset folder path
data_dir = os.path.join(base_dir, "dresden")
# preprocessed dataset folder path
preproc_dir = os.path.join(base_dir, "dresden_preprocessed")
# preprocessed dataset of "rgb" images folder path
preproc_img_dir = os.path.join(preproc_dir, "rgb")
# prnu type to extract from "rgb" images
prnu_type = "prnu_lp"
# where to save the dataset info as a .txt file
dataset_info_savedir = preproc_dir
# if you alter "n_unique_in" or "n_unique_out" then you must alter the
# "preprocess.py" file
# each in-distribution camera model must have "n_unique_in" camera devices
n_unique_in = 5
# each in-distribution camera model must have "n_unique_out" camera devices
n_unique_out = 3
