import json
import os
import random
import subprocess
from logging import getLogger
from math import ceil, log10

import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.image import imsave
from torchvision.utils import make_grid

from .utils import preprocess

logger = getLogger()


class GAN_Tester(object):
    """
    Evaluate the gan's performance on the test dataset.
    """

    def __init__(self, gen, clf_low, clf_high, est, percept_model,
                 test_dataset, params):
        """
        Initialize gan testing.
        """
        # data / parameters
        self.test_dataset = test_dataset
        self.params = params

        # models
        self.gen = gen
        self.clf_low = clf_low
        self.clf_high = clf_high
        self.est = est
        self.percept_model = percept_model

        # type of input to the generator (remosaic / rgb)
        if params.gen_input == "remosaic":
            self.transform_samples = self.transform_from_remosaic
        else:
            self.transform_samples = self.transform_from_rgb

        # low-frequency input / classifier
        if self.clf_low is not None:
            self.clf_forward = self.classify_low
            if "prnu_lp_low" in params.clf_low_input:
                self.get_clf_low_input = self._prnu_low
            else:
                self.get_clf_low_input = self._rgb
        else:
            self.clf_forward = self.null_classify

        # high-frequency input / classifier
        if self.clf_high is not None:
            self.clf_forward = self.classify_high
            if "prnu_lp" in params.clf_high_input:
                self.get_clf_high_input = self._prnu
            else:
                self.get_clf_high_input = self._rgb
        else:
            self.clf_forward = self.null_classify

    def _rgb(self, data):
        """
        Returns the rgb data.
        """
        return data

    def _prnu(self, data):
        """
        Returns PRNU estimates of the given data.
        """
        return self.est(data)

    def _prnu_low(self, data):
        """
        Returns the low-frequency components (data minus their corresponding
        PRNU estimates).
        """
        return data - self.est(data)

    def _prnu_low_high(self, data):
        """
        Returns the low-frequency components concatenated with their
        corresponding high-frequency PRNU estimates.
        """
        prnu_hat = self.est(data)
        return torch.cat((data - prnu_hat, prnu_hat), 1)

    def transform_from_remosaic(self, data_rgb, data_remosaic, labels,
                                target_labels):
        """
        Generate fake images from remosaiced inputs.
        """
        # move data_remosaic to cuda
        data_remosaic = data_remosaic.to(device=self.params.primary_gpu)

        # targeted transformation
        return self.gen(data_remosaic, target_labels)

    def transform_from_rgb(self, data_rgb, data_remosaic, labels,
                           target_labels):
        """
        Generate fake images from rgb inputs.
        """
        # targeted transformation
        return self.gen(data_rgb, target_labels)

    def classify_low(self, data, labels, target_labels, conf_mat_tgted,
                     conf_mat_untgted):
        """
        Low-frequency classifier.
        """
        # get classifier input
        data = self.get_clf_low_input(data)

        # classify data
        logits = self.clf_low(data)

        # predictions
        _, preds = torch.max(logits, 1)

        # add results to the confusion matrix
        for t, l, p in zip(target_labels.view(-1), labels.view(-1),
                           preds.view(-1)):
            conf_mat_tgted[t.long(), p.long()] += 1
            if self.params.comp_untgted:
                conf_mat_untgted[l.long(), p.long()] += 1
        return conf_mat_tgted, conf_mat_untgted

    def classify_high(self, data, labels, target_labels, conf_mat_tgted,
                      conf_mat_untgted):
        """
        High-frequency classifier.
        """
        # get classifier input
        data = self.get_clf_high_input(data)

        # classify data
        logits = self.clf_high(data)

        # predictions
        _, preds = torch.max(logits, 1)

        # add results to the confusion matrix
        for t, l, p in zip(target_labels.view(-1), labels.view(-1),
                           preds.view(-1)):
            conf_mat_tgted[t.long(), p.long()] += 1
            if self.params.comp_untgted:
                conf_mat_untgted[l.long(), p.long()] += 1
        return conf_mat_tgted, conf_mat_untgted

    def null_classify(self, data, labels, target_labels, conf_mat_tgted,
                      conf_mat_untgted):
        """
        Null classifier.
        """
        return conf_mat_tgted, conf_mat_untgted

    def min_max_normalize(self, images):
        """
        Min-max normalize images to [0,1].
        """
        for i in range(images.shape[0]):
            images[i] = images[i].sub(images[i].min()).div(
                images[i].max() - images[i].min())
        return images

    def save_images(self, ims_to_plot, n_versions, sample_num):
        """
        Save a grid of images.
        """
        # image grid
        grid_img = make_grid(ims_to_plot, nrow=n_versions + 1,
                             padding=self.params.padding, pad_value=1,
                             normalize=False, scale_each=False)
        grid_img = grid_img.detach().cpu().numpy().transpose((1, 2, 0))

        # save image
        output_path = os.path.join(self.image_path, "img_%s.png" % sample_num)
        imsave(output_path, grid_img)
        print("saved visualization '%s' ..." % output_path)

    def compute_distortion(self, data_rgb, gen_outputs, target_labels, psnrs,
                           lpips, samples_per_class):
        """
        Compute the distortion introduced by transforming inputs conditioned
        on target labels.
        """
        # batch size
        bs = data_rgb.shape[0]

        # rescale all pixel values from [0,255] to [-1,1] for lpips computation
        data_rgb, gen_outputs = preprocess(data_rgb), preprocess(gen_outputs)

        # get indices for each class in the batch
        target_labels = target_labels.long()
        idxs, counts = target_labels.unique(return_counts=True)
        counts = counts.float().cpu()
        samples_per_class[idxs] += counts

        # compute lpips / stop gradient computation
        with torch.no_grad():
            lpips_ = self.percept_model.forward(
                pred=gen_outputs, target=data_rgb, normalize=False)

        # compute psnr
        data_rgb, gen_outputs = data_rgb.view(bs, -1), gen_outputs.view(bs, -1)
        mse = F.mse_loss(gen_outputs, data_rgb, reduction="none").mean(1)

        psnrs_ = torch.zeros_like(mse)
        for i in range(psnrs_.shape[0]):
            psnrs_[i] = 10.0 * log10(2.**2 / mse[i])

        # add distortion metrics to per class tensor
        for t in range(len(idxs)):
            where_ = torch.where(target_labels == idxs[t])[0]
            psnrs[idxs[t]] += psnrs_[where_].sum()
            lpips[idxs[t]] += lpips_[where_].sum()
        return psnrs, lpips

    def transform_distortion(self):
        """
        Compute the distortion precomputed transformed images.
        """
        params = self.params
        transformed_data = self.test_dataset[0][0]
        target_labels = self.test_dataset[0][2]
        dataset = self.test_dataset[1]

        # possible target labels depend on whether the test images are in- or
        # out-of-distribution
        range_ = range(1 if params.in_dist else 0, params.n_classes)

        groundtruth_imgs = torch.zeros([params.n_samples_test * (len(range_)),
                                        params.ptc_fm,
                                        params.centre_crop_size,
                                        params.centre_crop_size],
                                       dtype=torch.float32)

        idx = 0
        for r_ in range_:
            # iterate over the test dataset
            for n_iter, (data_rgb, labels) in enumerate(dataset):
                data_rgb = data_rgb[0]
                n_imgs = data_rgb.shape[0]
                groundtruth_imgs[idx:idx + n_imgs] = data_rgb
                idx += len(labels)

        n_imgs = transformed_data.shape[0]

        # visualizations
        if params.visualize > 0:
            # create a dummy image to ensure the plot looks nice
            dummy_img = torch.ones_like(groundtruth_imgs[0:1]).float()

            # create a visualization image folder dump
            self.image_path = params.vis_output_path
            if not os.path.exists(self.image_path):
                subprocess.Popen("mkdir -p %s" %
                                 self.image_path, shell=True).wait()
            print("saving transformation visualizations to '%s' ..."
                  % self.image_path)

            # randomly select samples to visualize
            vis_idxs = list(range(params.n_samples_test))
            random.shuffle(vis_idxs)
            vis_idxs = vis_idxs[:params.visualize]
            for v in range(params.visualize):
                # get the correct indices of the samples
                smpl_idxs = range(vis_idxs[v], n_imgs, params.n_samples_test)

                # calculate the perturbation in additive terms
                delta = (transformed_data[smpl_idxs] -
                         groundtruth_imgs[smpl_idxs])
                delta = self.min_max_normalize(delta)

                # concat the necessary images to be plotted
                ims_to_plot = torch.cat((
                    groundtruth_imgs[smpl_idxs][0:1] / 255.,
                    transformed_data[smpl_idxs] / 255.,
                    dummy_img,
                    delta))

                # save the images
                self.save_images(ims_to_plot, len(range_), v)

        # initialize logs
        log_psnr = []
        log_lpips = []

        # compute distortion
        if params.comp_distortion:
            # initialize the metrics
            psnrs = torch.zeros(params.n_classes).to(torch.float32)
            lpips = torch.zeros(params.n_classes).to(torch.float32)
            samples_per_class = torch.zeros(params.n_classes).to(torch.float32)

            n_batches = int(ceil(n_imgs / params.test_batch_size))

            for i in range(n_batches):
                j = i * params.test_batch_size
                k = min(j + params.test_batch_size, n_imgs)
                psnrs, lpips = self.compute_distortion(
                    groundtruth_imgs[j:k].to(device=params.primary_gpu),
                    transformed_data[j:k].to(device=params.primary_gpu),
                    target_labels[j:k].to(device=params.primary_gpu),
                    psnrs,
                    lpips,
                    samples_per_class)

            # aggregate
            psnrs = [x.cpu().numpy() for x in psnrs / samples_per_class]
            lpips = [x.cpu().numpy() for x in lpips / samples_per_class]

            logger.info("")

            # image quality metrics
            log_psnr += [("mean_psnr", np.nanmean(psnrs).tolist())]
            for err, n_class in zip(psnrs, range(params.n_classes)):
                log_psnr.append(("psnr_%s" % n_class, err.tolist()))

            log_lpips += [("mean_lpips", np.nanmean(lpips).tolist())]
            for err, n_class in zip(lpips, range(params.n_classes)):
                log_lpips.append(("lpips_%s" % n_class, err.tolist()))

        return log_psnr, log_lpips

    def transform_accuracy(self, log_psnr, log_lpips):
        """
        Compute the distortion precomputed transformed images.
        """
        params = self.params
        transformed_data = self.test_dataset[0][0]
        labels = self.test_dataset[0][1]
        target_labels = self.test_dataset[0][2]

        # initialize the confusion matrices
        conf_mat_tgted = torch.zeros(params.n_classes_test,
                                     params.n_classes_test).to(torch.float32)
        conf_mat_untgted = torch.zeros(params.n_classes_test,
                                       params.n_classes_test).to(torch.float32)

        # classifiers
        n_imgs = transformed_data.shape[0]
        n_batches = int(ceil(n_imgs / params.test_batch_size))

        for i in range(n_batches):
            j = i * params.test_batch_size
            k = min(j + params.test_batch_size, n_imgs)

            conf_mat_tgted, conf_mat_untgted = self.clf_forward(
                transformed_data[j:k].to(device=params.primary_gpu),
                labels[j:k].to(device=params.primary_gpu),
                target_labels[j:k].to(device=params.primary_gpu),
                conf_mat_tgted,
                conf_mat_untgted)

        # aggregate
        conf_mat_tgted = [x.cpu().numpy() for x in conf_mat_tgted.diag() /
                          conf_mat_tgted.sum(1)]
        conf_mat_untgted = [x.cpu().numpy()
                            for x in (conf_mat_untgted.sum(1) -
                                      conf_mat_untgted.diag()) /
                            conf_mat_untgted.sum(1)]

        # initialize logs
        log_clf_tgted = []
        log_clf_untgted = []

        # classifier accuracies
        log_clf_tgted += [("mean_clf_tgted_acc",
                           np.nanmean(conf_mat_tgted).tolist())]
        for acc, n_class in zip(conf_mat_tgted, range(params.n_classes_test)):
            log_clf_tgted.append(("clf_tgted_acc_%s" % n_class, acc.tolist()))

        if params.comp_untgted:
            log_clf_untgted += [("mean_clf_untgted_acc",
                                 np.nanmean(conf_mat_untgted).tolist())]
            for acc, n_class in zip(conf_mat_untgted,
                                    range(params.n_classes_test)):
                log_clf_untgted.append(("clf_untgted_acc_%s"
                                        % n_class, acc.tolist()))

        # JSON log
        to_log = dict([] + log_psnr + log_lpips + log_clf_tgted +
                      log_clf_untgted)
        print("__log__:%s" % json.dumps(to_log))
        logger.debug("__log__:%s" % json.dumps(to_log))

    def transform_and_save(self):
        """
        Transform and then save the data to disk.
        """
        params = self.params

        # generator in eval mode
        self.gen.eval()

        # possible target labels depend on whether the test images are in- or
        # out-of-distribution
        range_ = range(1 if params.in_dist else 0, params.n_classes)

        # initialize arrays for storing transformed images / labels
        groundtruth_labels = torch.zeros([params.n_samples_test *
                                          (len(range_)), ],
                                         dtype=torch.float32)
        target_labels = torch.zeros([params.n_samples_test * (len(range_)), ],
                                    dtype=torch.float32)
        transformed_imgs = torch.zeros([params.n_samples_test * (len(range_)),
                                        params.ptc_fm,
                                        params.centre_crop_size,
                                        params.centre_crop_size],
                                       dtype=torch.float32)

        idx = 0

        # stop gradient computation
        with torch.no_grad():
            for r_ in range_:
                # iterate over the test dataset
                for n_iter, (data, labels) in enumerate(self.test_dataset):
                    # split data into rgb and corresponding remosaiced
                    # versions.
                    # Note: if the generator's input is "rgb" then data[1] = []
                    data_rgb, data_remosaic = data[0], data[1]

                    # move data to cuda except data_remosaic which may be []
                    data_rgb = data_rgb.to(device=params.primary_gpu)
                    labels = labels.to(device=params.primary_gpu)

                    # number of images in the batch
                    n_imgs = len(labels)

                    # generate target labels different from the ground truth
                    # labels
                    if params.in_dist:
                        batch_target_labels = (labels + r_) % params.n_classes
                    else:
                        batch_target_labels = torch.zeros_like(labels) + r_

                    # transform the data conditioned on their corresponding
                    # target labels
                    gen_outputs = self.transform_samples(data_rgb,
                                                         data_remosaic,
                                                         labels,
                                                         batch_target_labels)

                    # store data / labels in pre-initialized arrays
                    groundtruth_labels[idx:idx + n_imgs] = labels.detach()\
                        .cpu()
                    target_labels[idx:idx + n_imgs] = batch_target_labels\
                        .detach().cpu()
                    transformed_imgs[idx:idx + n_imgs] = gen_outputs.detach()\
                        .cpu()
                    idx += n_imgs

        if params.save_transformed_imgs:
            save_fname = "transformed_imgs"
            save_fname += "_id.pth" if params.in_dist else "_ood.pth"
            save_name = os.path.join(params.dump_path, save_fname)
            torch.save([transformed_imgs, groundtruth_labels, target_labels,
                        params.gen_input], save_name)
            print("saving transformed images, groundtruth labels and target "
                  "labels to '%s'" % save_name)
        return [transformed_imgs, groundtruth_labels, target_labels,
                params.gen_input]
