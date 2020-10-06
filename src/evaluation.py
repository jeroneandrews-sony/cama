import json
import os
import subprocess
from logging import getLogger
from math import log10

import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.image import imsave
from torchvision.utils import make_grid

from .utils import preprocess, print_metric

logger = getLogger()


def estimator_evaluate(estimator, test_dataset, params):
    """
    Evaluate a PRNU estimator.
    """
    # estimator in eval mode
    estimator.eval()

    # initialize rmse tensor
    rmses = torch.zeros(params.n_classes).to(torch.float32)
    samples_per_class = torch.zeros(params.n_classes).to(torch.float32)

    # stop gradient computation
    with torch.no_grad():
        # iterate over the validation / test dataset
        for n_iter, ((data_img, data_prnu), labels) in enumerate(test_dataset):
            # move data / labels to cuda
            data_img = data_img.to(device=params.primary_gpu)
            data_prnu = data_prnu.to(device=params.primary_gpu)
            labels = labels.to(device=params.primary_gpu)

            # batch size
            bs = data_img.shape[0]

            # estimate prnu
            prnu_hat = estimator(data_img)

            # compute rmse
            prnu_hat = prnu_hat.view(bs, -1)
            data_prnu = data_prnu.view(bs, -1)
            rmse = torch.sqrt(F.mse_loss(prnu_hat, data_prnu,
                                         reduction="none").mean(1))

            # get indices for each class in the batch
            idxs, counts = labels.unique(return_counts=True)
            counts = counts.float().cpu()
            samples_per_class[idxs] += counts

            # add rmse to per class tensor
            for t in range(len(idxs)):
                where_ = torch.where(labels == idxs[t])[0]
                rmses[idxs[t]] += rmse[where_].sum()

    rmses = [x.cpu().numpy() for x in rmses / samples_per_class]
    return rmses


class Classifier_Evaluator(object):
    """
    Evaluate a classifier.
    """

    def __init__(self, clf_input):
        """
        Initialize type of classifier input.
        """
        if "prnu_lp_low+prnu_lp" in clf_input:
            self.get_clf_input = self._prnu_low_high
        elif "prnu_lp_low" in clf_input:
            self.get_clf_input = self._prnu_low
        elif "prnu_lp" in clf_input:
            self.get_clf_input = self._prnu
        else:
            self.get_clf_input = self._rgb

    def _rgb(self, est_prnu, data):
        """
        Returns the rgb data.
        """
        return data

    def _prnu(self, est_prnu, data):
        """
        Returns PRNU estimates of the given data.
        """
        return est_prnu(data)

    def _prnu_low(self, est_prnu, data):
        """
        Returns the low-frequency components (data minus their corresponding
        PRNU estimates).
        """
        return data - est_prnu(data)

    def _prnu_low_high(self, est_prnu, data):
        """
        Returns the low-frequency components concatenated with their
        corresponding high-frequency PRNU estimates.
        """
        prnu_hat = est_prnu(data)
        return torch.cat((data - prnu_hat, prnu_hat), 1)

    def classifier_accuracy(self, classifier, est_prnu, test_dataset, params):
        """
        Evaluate the classifier's performance.
        """
        # classifier in evaluation mode
        classifier.eval()
        if est_prnu is not None:
            est_prnu.eval()

        # initialize the confusion matrix
        conf_mat = torch.zeros(
            params.n_classes, params.n_classes).to(torch.float32)
        losses = torch.zeros(params.n_classes).to(torch.float32)
        samples_per_class = torch.zeros(params.n_classes).to(torch.float32)

        # stop gradient computation
        with torch.no_grad():
            # iterate over the validation / test dataset
            for n_iter, (data, labels) in enumerate(test_dataset):
                # move data / labels to cuda
                data = data.to(device=params.primary_gpu)
                labels = labels.to(device=params.primary_gpu)

                # get classifier input
                data = self.get_clf_input(est_prnu, data)

                # classify data
                logits = classifier(data)

                # predictions
                _, preds = torch.max(logits, 1)

                # add batch results to the confusion matrix
                for t, p in zip(labels.view(-1), preds.view(-1)):
                    conf_mat[t.long(), p.long()] += 1

                loss = F.cross_entropy(logits, labels, reduction="none")

                # get indices for each class in the batch
                idxs, counts = labels.unique(return_counts=True)
                counts = counts.float().cpu()
                samples_per_class[idxs] += counts

                # add l2 estimation errors to per class tensor
                for t in range(len(idxs)):
                    where_ = torch.where(labels == idxs[t])[0]
                    losses[idxs[t]] += loss[where_].sum().item()

        conf_mat = [x.cpu().numpy() for x in conf_mat.diag() / conf_mat.sum(1)]
        losses = [x.cpu().numpy() for x in losses / samples_per_class]
        return conf_mat, losses


class GAN_Evaluator(object):
    """
    Evaluate the gan's performance.
    """

    def __init__(self, gen, clf_low, clf_high, est, percept_model,
                 validation_dataset, params):
        """
        Initialize gan evaluation.
        """
        # data / parameters
        self.validation_dataset = validation_dataset
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
            self.clf_low_forward = self.classify_low
            if "prnu_lp_low" in params.clf_low_input:
                self.get_clf_low_input = self._prnu_low
            else:
                self.get_clf_low_input = self._rgb
        else:
            self.clf_low_forward = self.null_classify

        # high-frequency input / classifier
        if self.clf_high is not None:
            self.clf_high_forward = self.classify_high
            if "prnu_lp" in params.clf_high_input:
                self.get_clf_high_input = self._prnu
            else:
                self.get_clf_high_input = self._rgb
        else:
            self.clf_high_forward = self.null_classify

        # create transformed / real image folder dump
        self.params.image_path = os.path.join(params.dump_path, "images")
        if not os.path.exists(self.params.image_path):
            subprocess.Popen("mkdir -p %s" %
                             self.params.image_path, shell=True).wait()

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

    def classify_low(self, data, target_labels, conf_mat):
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
        for t, p in zip(target_labels.view(-1), preds.view(-1)):
            conf_mat[t.long(), p.long()] += 1
        return conf_mat

    def classify_high(self, data, target_labels, conf_mat):
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
        for t, p in zip(target_labels.view(-1), preds.view(-1)):
            conf_mat[t.long(), p.long()] += 1
        return conf_mat

    def null_classify(self, data, target_labels, conf_mat):
        """
        Null classifier.
        """
        return conf_mat

    def min_max_normalize(self, images):
        """
        Min-max normalize images to [0,1].
        """
        for i in range(images.shape[0]):
            images[i] = images[i].sub(images[i].min()).div(
                images[i].max() - images[i].min())
        return images

    def save_images(self, imgs_to_plot, nrow, image_name):
        """
        Save a grid of images.
        """
        padding = self.params.padding
        image_path = self.params.image_path

        # image grid
        grid_img = make_grid(imgs_to_plot, nrow=nrow, padding=padding,
                             pad_value=1.0, normalize=False, scale_each=False)
        grid_img = grid_img.detach().cpu().numpy().transpose((1, 2, 0))

        # remove outer padding
        grid_img = grid_img[padding:-padding, padding:-padding, :]

        # save grid of images
        output_path = os.path.join(image_path, image_name)
        imsave(output_path, grid_img)

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
        idxs, counts = target_labels.unique(return_counts=True)
        counts = counts.float().cpu()
        samples_per_class[idxs] += counts

        # compute lpips
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

    def eval_distortion_and_accuracy(self, n_epoch):
        """
        Compute the distortion and accuracy of the generator's outputs.
        """
        params = self.params

        # models in eval mode
        self.gen.eval()
        if self.clf_low is not None:
            self.clf_low.eval()
        if self.clf_high is not None:
            self.clf_high.eval()
        if self.est is not None:
            self.est.eval()
        self.percept_model.eval()

        # initialize the metrics
        psnrs = torch.zeros(params.n_classes).to(torch.float32)
        lpips = torch.zeros(params.n_classes).to(torch.float32)
        samples_per_class = torch.zeros(params.n_classes).to(torch.float32)

        # initialize the confusion matrices for the targeted success rates
        conf_mat_low = torch.zeros(params.n_classes,
                                   params.n_classes).to(
            torch.float32) if self.clf_low is not None else None
        conf_mat_high = torch.zeros(params.n_classes,
                                    params.n_classes).to(
            torch.float32) if self.clf_high is not None else None

        # stop gradient computation
        with torch.no_grad():
            # iterate over the validation / test dataset
            for n_iter, (data, labels) in enumerate(self.validation_dataset):
                range_ = range(1, params.n_classes)
                # split data into rgb and corresponding remosaiced versions.
                # Note: if the generator's input is "rgb" then data[1] = []
                data_rgb, data_remosaic = data[0], data[1]

                # move data to cuda except data_remosaic which may be []
                data_rgb = data_rgb.to(device=params.primary_gpu)
                labels = labels.to(device=params.primary_gpu)

                for r_ in range_:
                    # generate target labels different from the ground truth
                    # labels
                    target_labels = (labels + r_) % params.n_classes

                    # transform the data conditioned on their corresponding
                    # target labels
                    gen_outputs = self.transform_samples(
                        data_rgb, data_remosaic, labels, target_labels)

                    # compute psnr / lpips
                    psnrs, lpips = self.compute_distortion(
                        data_rgb, gen_outputs, target_labels, psnrs, lpips,
                        samples_per_class)

                    # classify data (low-frequency classifier)
                    conf_mat_low = self.clf_low_forward(
                        gen_outputs, target_labels, conf_mat_low)

                    # classify data (high-frequency classifier)
                    conf_mat_high = self.clf_high_forward(
                        gen_outputs, target_labels, conf_mat_high)

                    # save first batch of images and transformed versions
                    if (n_iter == 0) and (r_ == min(range_)):
                        imgs_to_plot = torch.cat((
                            data_rgb / 255.,
                            gen_outputs / 255.,
                            self.min_max_normalize(
                                gen_outputs - data_rgb)),
                            0)
                        self.save_images(imgs_to_plot, data_rgb.shape[0],
                                         "targeted_%i.png" % n_epoch)

        # aggregate psnr / lpips / targeted success rates
        psnrs = [x.cpu().numpy() for x in psnrs / samples_per_class]
        lpips = [x.cpu().numpy() for x in lpips / samples_per_class]
        if conf_mat_low is not None:
            conf_mat_low = [x.cpu().numpy()
                            for x in conf_mat_low.diag() /
                            conf_mat_low.sum(1)]
        if conf_mat_high is not None:
            conf_mat_high = [x.cpu().numpy()
                             for x in conf_mat_high.diag() /
                             conf_mat_high.sum(1)]
        return psnrs, lpips, conf_mat_low, conf_mat_high

    def evaluate(self, n_epoch):
        """
        Evaluate the generator and log the results.
        """
        params = self.params

        logger.info("")

        # evaluation metrics
        psnrs, lpips, conf_mat_low, conf_mat_high = \
            self.eval_distortion_and_accuracy(n_epoch)

        # initialize the logs
        log_psnr = []
        log_lpips = []
        log_clf_low = [] if self.clf_low is not None else None
        log_clf_high = [] if self.clf_high is not None else None

        # image quality metrics
        log_psnr += [("psnr", np.nanmean(psnrs).tolist())]
        for err, n_class in zip(psnrs, range(params.n_classes)):
            log_psnr.append(("psnr_%s" % n_class, err.tolist()))
        logger.info("psnr:")
        print_metric(log_psnr, False)

        log_lpips += [("lpips", np.nanmean(lpips).tolist())]
        for err, n_class in zip(lpips, range(params.n_classes)):
            log_lpips.append(("lpips_%s" % n_class, err.tolist()))
        logger.info("lpips:")
        print_metric(log_lpips, False)

        # targeted success rates
        if log_clf_low is not None:
            log_clf_low += [("clf_low_acc", np.nanmean(conf_mat_low).tolist())]
            for acc, n_class in zip(conf_mat_low, range(params.n_classes)):
                log_clf_low.append(("clf_low_acc_%s" % n_class, acc.tolist()))
            logger.info("clf. rgb acc.:")
            print_metric(log_clf_low, True)
        else:
            log_clf_low = []

        if log_clf_high is not None:
            log_clf_high += [("clf_high_acc",
                              np.nanmean(conf_mat_high).tolist())]
            for acc, n_class in zip(conf_mat_high, range(params.n_classes)):
                log_clf_high.append(("clf_high_acc_%s" %
                                     n_class, acc.tolist()))
            logger.info("clf. noise acc.:")
            print_metric(log_clf_high, True)
        else:
            log_clf_high = []

        # JSON log
        to_log = dict([
            ("n_epoch", n_epoch),
        ] + log_psnr + log_lpips + log_clf_low + log_clf_high)
        logger.debug("__log__:%s" % json.dumps(to_log))
        return to_log
