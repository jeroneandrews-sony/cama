import pickle
import random
import subprocess
from glob import glob
from multiprocessing import Pool, cpu_count
from os.path import exists, join

import config
import numpy as np
import torch
from colour_demosaicing import (demosaicing_CFA_Bayer_Menon2007,
                                mosaicing_CFA_Bayer)
from PIL import Image
from prnu import extract_prnu_lp
from torchvision.transforms import CenterCrop, ToTensor
from tqdm import tqdm


def remosaic(rgb_image, pattern="RGGB"):
    """
    Remosaic an rgb image using a Bayer pattern.
    """
    cfa = mosaicing_CFA_Bayer(rgb_image, pattern=pattern)
    return demosaicing_CFA_Bayer_Menon2007(cfa, pattern=pattern)


def merge_lists(list1, list2):
    """
    Merge two python lists into a list of tuples.
    """
    return list(map(lambda x, y: (x, y), list1, list2))


def preprocess_images():
    """
    Preprocess images in the dataset.
    """
    # create directory to store the files in, if it doesn't already exists
    if not exists(config.preproc_dir):
        subprocess.Popen("mkdir -p %s" % config.preproc_dir, shell=True).wait()

    # (step 1) in distribution camera model images

    # camera model folders
    cm_folders = glob(join(config.data_dir, "**"))

    # number of devices per camera model
    devices = np.zeros((len(cm_folders), ))

    # compute the number of unique devices for each camera model
    for i in range(len(cm_folders)):
        # image files are .JPG (e.g. "Pentax_OptioW60_0_32803.JPG")
        # filename format "cameraBrand_cameraModel_device#_image#.JPG"
        files = glob(join(cm_folders[i], "*.JPG"))
        for file_ in files:
            # force device indices to start at 1 instead of 0
            devices[i] = np.maximum(devices[i], int(file_.split("_")[-2]) + 1)

    # find camera models with n_unique devices defined in the config module
    cm_ids = np.where(devices == config.n_unique_in)[0]
    cms = []
    for i in range(len(cm_ids)):
        cms.append(cm_folders[cm_ids[i]])

    # number of camera models with n_unique devices
    n_cms = len(cms)

    # count the total number of images
    n_imgs = 0
    for i in range(len(cms)):
        files = glob(join(cms[i], "*.JPG"))
        n_imgs += len(files)

    # create dict with camera model names
    cm_names = [cms[i].split("/")[-1] for i in range(len(cms))]
    cm_names_dict = {key: value for key, value in zip(range(len(cms)),
                                                      cm_names)}
    # save dict using pickle to .txt file
    with open(join(config.preproc_dir,
                   "indist_cm_name_dict.txt"), "wb") as handle:
        pickle.dump(cm_names_dict, handle)

    img_path = []
    img_dataset = []
    img_device = []
    img_cm = []
    for i in range(n_cms):
        # file paths for images captured by camera model i
        files = glob(join(cms[i], "*.JPG"))
        for file_ in files:
            # capture device for the current image (device indices start at 0)
            curr_img_device = int(file_.split("_")[-2])
            img_device.append(curr_img_device)

            # capture camera model for the current image
            img_cm.append(i)

            # split images into three datasets: {adversary, examiner or test}
            if (curr_img_device == 0) or (curr_img_device == 1):
                # devices 0 and 1
                img_dataset.append("adversary")
            elif (curr_img_device == 2) or (curr_img_device == 3):
                # devices 2 and 3
                img_dataset.append("examiner")
            else:
                # device 4
                img_dataset.append("test")

            # add image path to list
            img_path.append(file_)

    # create folders in advance for the adversary and examiner sets with train
    # / validation partitions
    data_types = ["rgb"]
    if config.prnu_extract:
        data_types.append("prnu_lp")
    if config.remosaic:
        data_types.append("remosaic")

    dataset_to_split = ["adversary", "examiner"]
    partition = ["train", "validation"]
    for i in range(n_cms):
        for dataset in dataset_to_split:
            for part in partition:
                for data_type in data_types:
                    folder_to_make = join(config.preproc_dir, data_type,
                                          dataset, part, cm_names_dict[i])
                    if not exists(folder_to_make):
                        subprocess.Popen("mkdir -p %s" % folder_to_make,
                                         shell=True).wait()

    # create folders in advance for the test set
    for i in range(n_cms):
        for data_type in data_types:
            folder_to_make = join(config.preproc_dir, data_type, "test",
                                  cm_names_dict[i])
            if not exists(folder_to_make):
                subprocess.Popen("mkdir -p %s" % folder_to_make,
                                 shell=True).wait()

    dataset_to_split = ["adversary", "examiner", "test"]

    save_names_in = []
    img_paths_in = []
    # centrally crop images to a common resolution and save images as torch
    # .pth files
    for dataset in dataset_to_split:
        dataset_ids = [i for i, j in enumerate(img_dataset) if j == dataset]
        random.seed(7789)
        random.shuffle(dataset_ids)

        # select a random sample of 100 images per camera model for the test
        # set based on camera model counts
        if dataset == "test":
            counts = np.zeros((len(cms)), dtype="uint8")

        # number of images to use for the training set
        n_train = int(config.train_prop * len(dataset_ids))

        for i in range(len(dataset_ids)):
            idx = dataset_ids[i]
            label = img_cm[idx]

            fname = img_path[idx].split("/")
            if dataset != "test":
                # train / validation sets
                if i <= n_train:
                    save_name = join(config.preproc_dir, "rgb", dataset,
                                     "train", fname[-2],
                                     fname[-1].replace("JPG", "pth"))
                else:
                    save_name = join(config.preproc_dir, "rgb", dataset,
                                     "validation", fname[-2],
                                     fname[-1].replace("JPG", "pth"))
            else:
                # test sets: 100 images per camera model
                if counts.sum() >= (100 * len(counts)):
                    break
                if counts[label].sum() >= 100:
                    continue
                counts[label] += 1
                save_name = join(config.preproc_dir, "rgb", dataset,
                                 fname[-2], fname[-1].replace("JPG", "pth"))

            save_names_in.append(save_name)
            img_paths_in.append(img_path[idx])

    # (step 2) out of distribution camera model images

    # find camera models != n_unique devices
    cm_ids = np.where(devices == config.n_unique_out)[0]
    cms = []
    for i in range(len(cm_ids)):
        cms.append(cm_folders[cm_ids[i]])

    # number of camera models != n_unique devices
    n_cms = len(cms)

    # count the total number of images
    n_imgs = 0
    for i in range(len(cms)):
        files = glob(join(cms[i], "*.JPG"))
        n_imgs += len(files)

    # create dict with camera model names
    cm_names = [cms[i].split("/")[-1] for i in range(len(cms))]
    cm_names_dict = {key: value for key, value in zip(range(len(cms)),
                                                      cm_names)}
    # save dict using pickle to .txt file
    with open(join(config.preproc_dir, "outofdist_cm_name_dict.txt"),
              "wb") as handle:
        pickle.dump(cm_names_dict, handle)

    img_path = []
    img_dataset = []
    img_device = []
    img_cm = []
    for i in range(n_cms):
        # file paths for images captured by camera model i
        files = glob(join(cms[i], "*.JPG"))
        for file_ in files:
            # capture device for the current image (device indexes start at 0)
            curr_img_device = int(file_.split("_")[-2])
            # append device
            img_device.append(curr_img_device)
            # capture camera model for the current image
            img_cm.append(i)
            # add image path to list
            img_path.append(file_)
            # for out-of-dist test set only use images from a single device,
            # else use for training
            if (curr_img_device == 2):
                # device 2
                img_dataset.append("test_outdist")
            else:
                img_dataset.append("train_outdist")

    # create folders in advance for adversary and examiner sets with train /
    # validations partitions
    data_types = ["rgb"]
    if config.prnu_extract:
        data_types.append("prnu_lp")

    if config.remosaic:
        data_types.append("remosaic")

    # create folders in advance for the test set
    for i in range(n_cms):
        for data_type in data_types:
            folder_to_make = join(config.preproc_dir, data_type,
                                  "test_outdist", cm_names_dict[i])
            if not exists(folder_to_make):
                subprocess.Popen("mkdir -p %s" % folder_to_make,
                                 shell=True).wait()

    # create folders in advance for train / validation out-of-distribution sets
    partition = ["train", "validation"]
    for i in range(n_cms):
        for part in partition:
            for data_type in data_types:
                folder_to_make = join(config.preproc_dir, data_type,
                                      "train_outdist", part, cm_names_dict[i])
                if not exists(folder_to_make):
                    subprocess.Popen("mkdir -p %s" % folder_to_make,
                                     shell=True).wait()

    save_names_out = []
    img_paths_out = []

    datasets = ["train_outdist", "test_outdist"]
    for dataset in datasets:
        # centrally crop images to a common resolution and save images as
        # torch .pth files
        dataset_ids = [i for i, j in enumerate(img_dataset) if j == dataset]
        random.seed(7789)
        random.shuffle(dataset_ids)

        # select a random sample of 100 images per camera model for the test
        # set based on camera model counts
        if dataset == "test_outdist":
            counts = np.zeros((len(cms)), dtype="uint8")

        n_train = int(config.train_prop * len(dataset_ids))

        for i in range(len(dataset_ids)):
            idx = dataset_ids[i]
            label = img_cm[idx]

            fname = img_path[idx].split("/")
            if dataset != "test_outdist":
                if i <= n_train:
                    save_name = join(config.preproc_dir, "rgb", dataset,
                                     "train", fname[-2],
                                     fname[-1].replace("JPG", "pth"))
                else:
                    save_name = join(config.preproc_dir, "rgb", dataset,
                                     "validation", fname[-2],
                                     fname[-1].replace("JPG", "pth"))
            else:
                if counts.sum() >= (100 * len(counts)):
                    break
                if counts[label].sum() >= 100:
                    continue
                counts[label] += 1
                save_name = join(config.preproc_dir, "rgb", dataset,
                                 fname[-2], fname[-1].replace("JPG", "pth"))

            save_names_out.append(save_name)
            img_paths_out.append(img_path[idx])

    return merge_lists(img_paths_in, save_names_in) + merge_lists(
        img_paths_out, save_names_out)


def write_data(info):
    """
    Write data to disk by iterating through the image list..
    """
    img_path, save_name = info

    # open image with PIL
    img = Image.open(img_path)
    # centre crop and convert to tensor of dtype uint8
    img = CenterCrop(config.centre_crop_size)(img)
    img = ToTensor()(img) * 255
    img = img.type(torch.uint8)
    # save image
    torch.save(img, save_name)

    # save remosaiced image
    if (config.remosaic) and ("examiner" not in save_name):
        remosaiced = remosaic(np.array(img).transpose(1, 2, 0)).\
            transpose(2, 0, 1)
        remosaiced = np.clip(remosaiced, 0, 255)
        remosaiced = torch.from_numpy(remosaiced)
        # save remosaiced
        torch.save(remosaiced, save_name.replace("rgb", "remosaic"))

    # save prnu_lp (with linear pattern)
    if config.prnu_extract:
        prnu = extract_prnu_lp(np.array(img).transpose(1, 2, 0))
        prnu = prnu.transpose(2, 0, 1)
        prnu = torch.from_numpy(prnu)
        torch.save(prnu, save_name.replace("rgb", "prnu_lp"))


if __name__ == "__main__":

    # get a list of tuples: (img_path, save_path)
    data_info = preprocess_images()

    # use multiprocessing Pool to speed things up
    with Pool(cpu_count() - 1) as p:
        list(tqdm(p.imap(write_data, data_info), total=len(data_info)))
