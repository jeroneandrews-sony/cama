import random

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader
from torchvision.datasets import DatasetFolder

from .sampler import ClassBasedRandomDatasetSampler, ImbalancedDatasetSampler


class sample_loader(object):
    """
    Load Dresden images.
    """

    def __init__(self, model_type, gen_input):
        """
        Initialize sample loader based on type of neural net model being used.
        """
        self.model_type = model_type
        self.gen_input = gen_input

        if model_type == "classifiers":
            self.loader = self.basic_loader
        elif model_type == "estimators":
            self.loader = self.estimator_loader
        elif model_type == "gans":
            if self.gen_input == "remosaic":
                self.loader = self.gan_remosaic_loader
            else:
                self.loader = self.basic_loader
        else:
            raise Exception('Unknown model type: "%s"' % model_type)

    def basic_loader(self, path):
        """
        Returns an image.
        """
        return torch.load(path).float()

    def estimator_loader(self, path):
        """
        Returns an image and its PRNU (with linear pattern) noise.
        """
        path_prnu = path.replace("rgb", "prnu_lp")
        return torch.load(path).float(), torch.load(path_prnu).float()

    def gan_remosaic_loader(self, path):
        """
        Returns an image and its remosaic.
        """
        remosaic_path = path.replace("rgb", "remosaic")
        return torch.load(path).float(), torch.load(remosaic_path).float()

    def load(self, path):
        return self.loader(path)


class transforms(object):
    """
    Training data transformations (dihedral group 4 and cropping).
    """

    def __init__(self, rnd_crops, ptc_sz, model_type, gen_input):
        """
        Initialize type of transformations applied.
        """
        self.ptc_sz = ptc_sz
        self.model_type = model_type
        if rnd_crops:
            self.crop = self.random_crop
        else:
            self.crop = self.grid_crop

        if model_type == "estimators":
            self.transformer = self.estimator_transforms
        elif model_type == "gans":
            if gen_input == "remosaics":
                self.transformer = self.gan_transforms
            else:
                self.transformer = self.gan_basic_transforms
        elif model_type == "classifiers":
            self.transformer = self.basic_transforms
        else:
            raise Exception("Unknown model type: '%s'" % model_type)

    def dihedral_group4(self, sample):
        """
        Random Dihedral group 4 transformations.
        """
        # random vertical and horizontal flipping
        if random.random() > 0.5:
            sample = torch.flip(sample, (1,))
        if random.random() > 0.5:
            sample = torch.flip(sample, (2,))
        # random rotation by 0, 90, 180 or 270 degrees
        n_rots = random.randint(0, 3)
        sample = torch.rot90(sample, n_rots, [1, 2])
        return sample

    def random_crop(self, sample):
        """
        Crop a patch randomly from an image.
        """
        ptc_sz = self.ptc_sz
        x_crop, y_crop = (random.randint(0, sample.shape[1] - ptc_sz),
                          random.randint(0, sample.shape[2] - ptc_sz))
        sample = sample[:, x_crop:x_crop + ptc_sz, y_crop:y_crop + ptc_sz]
        return sample

    def grid_crop(self, sample):
        """
        Crop a patch from a non-overlapping image grid.
        """
        ptc_sz = self.ptc_sz
        x_crop, y_crop = (random.randint(0, (sample.shape[1] // ptc_sz) - 1) *
                          ptc_sz, random.randint(0, (sample.shape[2] // ptc_sz)
                                                 - 1) * ptc_sz)
        sample = sample[:, x_crop:x_crop + ptc_sz, y_crop:y_crop + ptc_sz]
        return sample

    def basic_transforms(self, sample):
        """
        Crop and perform Dihedral group 4 transformations.
        """
        ptc_sz = self.ptc_sz
        sample = self.crop(sample, ptc_sz)
        return self.dihedral_group4(sample)

    def estimator_transforms(self, sample):
        """
        Concatenate channel-wise a paired sample (rgb, prnu), then crop and
        perform Dihedral group 4 transformations. Return rgb and prnu
        components separately.
        """
        ptc_sz = self.ptc_sz
        sample = torch.cat((sample[0], sample[1]), 0)
        sample = self.crop(sample, ptc_sz)
        sample = self.dihedral_group4(sample)
        return sample[:3], sample[3:]

    def gan_basic_transforms(self, sample):
        """
        Crop and perform Dihedral group 4 transformations. Return transformed
        sample and an empty list.
        """
        ptc_sz = self.ptc_sz
        sample = self.crop(sample, ptc_sz)
        return self.dihedral_group4(sample), []

    def gan_transforms(self, sample):
        """
        Concatenate channel-wise a paired sample (rgb, remosaic), then crop
        and perform Dihedral group 4 transformations. Return rgb and
        remosaiced components separately.
        """
        ptc_sz = self.ptc_sz
        sample = torch.cat((sample[0], sample[1]), 0)
        sample = self.crop(sample, ptc_sz)
        sample = self.dihedral_group4(sample)
        return sample[:3], sample[3:]

    def transform(self, sample):
        return self.transformer(sample)


class dataset_loader(object):
    """
    Dataset loader.
    """

    def __init__(self, params, train_mode):
        self.params = params
        self.train_mode = train_mode

    def dataset(self, data_root, increment):
        """
        Load a dataset.
        """
        params = self.params
        try:
            params.gen_input = params.gen_input
        except AttributeError:
            params.gen_input = None

        loader = sample_loader(params.model_type, params.gen_input).load

        # initialize transformations
        if self.train_mode:
            transformer = transforms(params.rnd_crops, params.ptc_sz,
                                     params.model_type,
                                     params.gen_input).transform
        else:
            transformer = None

        # load dataset folder
        dataset_folder = DatasetFolder(root=data_root, transform=transformer,
                                       loader=loader, extensions=(".pth"))

        n_classes = dataset_folder[-1][1] + 1
        n_samples = len(dataset_folder.samples)

        # increment the labels if using additional camera models
        if increment:
            dataset_folder.targets = [x + increment
                                      for x in dataset_folder.targets]
            for key_ in dataset_folder.class_to_idx.keys():
                dataset_folder.class_to_idx[key_] += increment
            for i in range(len(dataset_folder.samples)):
                t = list(dataset_folder.samples[i])
                t[1] += increment
                dataset_folder.samples[i] = tuple(t)
        return dataset_folder, n_classes, n_samples

    def train(self):
        """
        Construct training data loader using an imbalanced dataset sampler.
        """
        params = self.params
        dataset, n_classes, n_samples = self.dataset(params, params.train_root)

        # incorporate additional data captured by supplemental camera models
        if params.expanded_cms:
            assert "examiner" in params.train_root, "training with an "
            "expanded set of camera models (only valid if user is examiner)"
            exp_root = params.train_root.replace("examiner", "examiner_outdist")
            dataset_exp, n_classes_exp, n_samples_exp = self.dataset(
                params,
                exp_root,
                n_classes)

            dataset = ConcatDataset((dataset, dataset_exp))
            n_classes += n_classes_exp
            n_samples += n_samples_exp

        # if debug mode, use a subset of the data (approx 10%)
        if params.debug:
            params.n_samples_per_epoch = int((n_samples // n_classes) * 0.1)
            params.n_samples_per_epoch = np.maximum(params.n_samples_per_epoch,
                                                    1)
        # construct data loader with an imbalanced dataset sampler
        data_loader = DataLoader(dataset,
                                 batch_size=params.batch_size,
                                 shuffle=False,
                                 drop_last=True,
                                 sampler=ImbalancedDatasetSampler(
                                     dataset=dataset,
                                     n_samples=params.n_samples_per_epoch,
                                     debug=params.debug),
                                 pin_memory=params.pin_memory,
                                 num_workers=params.n_workers)
        return data_loader, n_classes, n_samples

    def test(self, test_root):
        """
        Construct  validation / testing data loader.
        """
        params = self.params
        dataset, n_classes, n_samples = self.dataset(params, test_root)

        # incorporate additional data captured by supplemental camera models
        if params.expanded_cms:
            if "validation" in test_root:
                assert "examiner" in test_root, "validating with an expanded "
                "set of camera models (only valid if user is examiner)"
                exp_root = test_root.replace("examiner", "examiner_outdist")
                dataset_exp, n_classes_exp, n_samples_exp = self.dataset(
                    params,
                    exp_root,
                    n_classes)
            else:
                exp_root = test_root.replace("test", "test_outdist")
                dataset_exp, n_classes_exp, n_samples_exp = self.dataset(
                    params,
                    exp_root,
                    n_classes)
            dataset = ConcatDataset((dataset, dataset_exp))
            n_classes += n_classes_exp
            n_samples += n_samples_exp

        # if debug mode is "True" use a subset of the data (approx 10%) using
        # a class based dataset sampler
        if not params.debug:
            sampler = None
        else:
            params.n_samples_per_class = int((n_samples // n_classes) * 0.1)
            params.n_samples_per_class = np.maximum(params.n_samples_per_class,
                                                    1)
            sampler = ClassBasedRandomDatasetSampler(
                dataset=dataset,
                n_samples_per_class=params.n_samples_per_class)

        # construct data loader
        data_loader = DataLoader(dataset,
                                 batch_size=params.test_batch_size,
                                 shuffle=False,
                                 drop_last=False,
                                 sampler=sampler,
                                 pin_memory=params.pin_memory,
                                 num_workers=0)
        return data_loader, n_classes, n_samples
