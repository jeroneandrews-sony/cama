import os
import subprocess
from multiprocessing import Pool, cpu_count

import pandas as pd
import requests
from config import base_dir, data_dir
from tqdm import tqdm

DATA_PATH = data_dir
if not os.path.exists(DATA_PATH):
    subprocess.Popen("mkdir -p %s" % DATA_PATH, shell=True).wait()

# read csv file with links to images
INFO_PATH = os.path.join(base_dir, "dresden_info.csv")
if not os.path.isfile(INFO_PATH):
    info = pd.read_csv(os.path.join(base_dir, "dresden_img_urls.csv"),
                       index_col=0)
    # whether the image has been downloaded or not
    info["downloaded"] = False
else:
    info = pd.read_csv(INFO_PATH, index_col=0)

# unique camera models
mdls = info["model"].unique()

# create folders for each unique camera model
for mdl in mdls:
    mdl_path = os.path.join(DATA_PATH, mdl)
    if not os.path.exists(mdl_path):
        subprocess.Popen("mkdir -p %s" % mdl_path, shell=True).wait()

# number of images to download
n_imgs = len(info)


def write_image(i_):
    """
    Write downloaded images to disk by iterating through the existing
    dataframe and downloading images not yet downloaded. If an image has
    already been downloaded, we'll skip trying to re-download it.
    """
    if info.loc[i_, "downloaded"]:
        return True
    url = info.loc[i_, "img_url"]
    model = info.loc[i_, "model"]
    fname = url.split("/")[-1]
    file_path = os.path.join(DATA_PATH, model, fname)
    if info.loc[i_, "downloaded"]:
        return True
    try:
        img_data = requests.get(url).content
        with open(file_path, "wb") as handler:
            handler.write(img_data)
        return True
    except requests.exceptions.RequestException as error:
        print("error: %s" % error)
        return False


if __name__ == "__main__":

    # use multiprocessing Pool to speed things up
    with Pool(cpu_count() - 1) as p:
        info["downloaded"] = list(tqdm(p.imap(write_image, range(n_imgs)),
                                       total=n_imgs))

    print("successfully downloaded [%i / %i] images"
          % (info["downloaded"].sum(), n_imgs))

    print("saving dresden dataframe with info related to whether images were "
          "successfully downloaded\nfile location: '[%s]'" % INFO_PATH)

    if info["downloaded"].sum() != n_imgs:
        print("rerun this file to try and download images that we failed to "
              "retrieve in the initial run.")

    # save dataframe locally
    info.to_csv(INFO_PATH, index=True)
