import random

import numpy as np
import torch
from torch.utils.data import ConcatDataset
from torch.utils.data.sampler import Sampler
from torchvision.datasets import DatasetFolder, ImageFolder


class ImbalancedDatasetSampler(Sampler):
    """Samples elements randomly from a given list of indices for imbalanced
    dataset
    Arguments:
        indices (list, optional): a list of indices
        n_samples (int, optional): number of samples to draw
    """

    def __init__(self, dataset, indices=None, n_samples=None, debug=False):
        self.debug = debug

        # if indices is not provided,
        # all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) \
            if indices is None else indices

        # if n_samples is not provided,
        # draw `len(indices)` samples in each iteration
        self.n_samples = len(self.indices) \
            if n_samples is None else n_samples

        # debug mode: ensure that n_samples is <= number of indices for
        # multinomial sampling without replacement
        if self.debug:
            if self.n_samples > len(self.indices):
                self.n_samples = len(self.indices)

        # distribution of classes in the dataset
        label_to_count = {}
        for idx in self.indices:
            label = self._get_label(dataset, idx)
            if label in label_to_count:
                label_to_count[label] += 1
            else:
                label_to_count[label] = 1

        # weight for each sample
        weights = [1.0 / label_to_count[self._get_label(dataset, idx)]
                   for idx in self.indices]
        self.weights = torch.DoubleTensor(weights)

    def _get_label(self, dataset, idx):
        dataset_type = type(dataset)
        if dataset_type is ImageFolder:
            return dataset.imgs[idx][1]
        elif dataset_type is DatasetFolder or dataset_type is ConcatDataset:
            return dataset[idx][1]
        else:
            raise NotImplementedError

    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(
            self.weights, self.n_samples, replacement=False
            if self.debug else True))

    def __len__(self):
        return self.n_samples


class ClassBasedRandomDatasetSampler(Sampler):
    """
    Arguments:
        n_samples_per_class (int, optional): number of samples per class draw
    """

    def __init__(self, dataset, n_samples_per_class):

        # distribution of classes in the dataset
        label_to_count = {}
        labels = []
        for idx in list(range(len(dataset))):
            label = self._get_label(dataset, idx)
            labels.append(label)
            if label in label_to_count:
                label_to_count[label] += 1
            else:
                label_to_count[label] = 1

        labels = np.array(labels).reshape((-1,))
        min_samples = np.minimum(np.min(list(label_to_count.values())),
                                 n_samples_per_class)
        idxs_per_class = [np.argwhere(labels == x).reshape((-1,))
                          for x in range(len(label_to_count))]

        for _ in range(len(label_to_count)):
            random.shuffle(idxs_per_class[_])
            idxs_per_class[_] = idxs_per_class[_][:min_samples]

        self.indices = []
        for _ in range(min_samples):
            for i in range(len(label_to_count)):
                self.indices.append(idxs_per_class[i][0])
                idxs_per_class[i] = idxs_per_class[i][1:]

        self.n_samples = len(self.indices)

    def _get_label(self, dataset, idx):
        dataset_type = type(dataset)
        if dataset_type is ImageFolder:
            return dataset.imgs[idx][1]
        elif dataset_type is DatasetFolder or dataset_type is ConcatDataset:
            return dataset[idx][1]
        else:
            raise NotImplementedError

    def __iter__(self):
        return (self.indices[i] for i in range(self.n_samples))

    def __len__(self):
        return self.n_samples
