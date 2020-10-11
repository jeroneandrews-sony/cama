import os
import pickle
from glob import glob
from shutil import rmtree

import pandas as pd
from config import data_dir, dataset_info_savedir

if __name__ == "__main__":
    # create in-distribution dataframe
    indist_name_dict = pickle.load(open(os.path.join(dataset_info_savedir,
                                                     "indist_cm_name_dict.txt"
                                                     ), "rb"))
    dataset_to_split = ["adversary", "examiner", "test"]
    all_ims_per_model = []
    for dset in dataset_to_split:
        ims_per_model = []
        for i in range(len(indist_name_dict)):
            if dset != "test":
                files_ = glob(os.path.join(dataset_info_savedir,
                                           "rgb/%s/**/%s/*.pth"
                                           % (dset, indist_name_dict[i])))
            else:
                files_ = glob(os.path.join(dataset_info_savedir,
                                           "rgb/%s/%s/*.pth"
                                           % (dset, indist_name_dict[i])))
            ims_per_model.append(len(files_))
        all_ims_per_model.append(ims_per_model)

    df_in = pd.DataFrame({"model": list(indist_name_dict.values()),
                          "adversary": all_ims_per_model[0],
                          "examiner": all_ims_per_model[1],
                          "test": all_ims_per_model[2]})

    # create out-of-distribution expanded camera models for examiner training
    # dataframe
    outofdist_name_dict = pickle.load(open(os.path.join(dataset_info_savedir,
                                                        "outofdist_cm_name"
                                                        "_dict.txt"), "rb"))
    dset = "examiner_outdist"
    ims_per_model = []
    for i in range(len(outofdist_name_dict)):
        files_ = glob(os.path.join(dataset_info_savedir, "rgb/%s/**/%s/*.pth"
                                   % (dset, outofdist_name_dict[i])))
        ims_per_model.append(len(files_))

    df_out = pd.DataFrame({"model": list(outofdist_name_dict.values()),
                           "examiner_outdist": ims_per_model})

    # create out-of-distribution dataframe
    outofdist_name_dict = pickle.load(open(os.path.join(dataset_info_savedir,
                                                        "outofdist_cm_name"
                                                        "_dict.txt"), "rb"))
    dset = "test_outdist"
    ims_per_model = []
    for i in range(len(outofdist_name_dict)):
        files_ = glob(os.path.join(dataset_info_savedir, "rgb/%s/%s/*.pth"
                                   % (dset, outofdist_name_dict[i])))
        ims_per_model.append(len(files_))

    df_out["test_outdist"] = ims_per_model

    # camera model labels, indexing starts from 0
    df_out.index = range(len(df_in), len(df_in) + len(df_out))

    # save paths
    df_in_savepath = os.path.join(dataset_info_savedir, "indist_summary.csv")
    df_out_savepath = os.path.join(dataset_info_savedir, "outdist_summary.csv")

    # print image counts and save dataframes to .csv files
    print("in-distribution image counts (below). saved to ['%s']\n"
          % df_in_savepath)
    print(df_in)
    print("in-distribution column sum\n")
    print(df_in.sum()[1:])
    df_in.to_csv(df_in_savepath, index=False)

    print("out-of-distribution image counts (below). saved to ['%s']\n"
          % df_out_savepath)
    print(df_out)
    print("out-of-distribution column sum")
    print(df_out.sum()[1:])
    df_out.to_csv(df_out_savepath, index=False)

    # remove unpreprocessed dresden data
    if os.path.isdir(data_dir):
        rmtree(data_dir)
