import json
import random
from logging import getLogger

import numpy as np
import torch
from torchsummary import summary

from .models.losses import (bce_logits_loss, gan_loss, l1_loss, mse_loss,
                            xent_loss)
from .models.utils import update_con_conv
from .utils import (generate_targets, get_optimizer, lambda_coeff, preprocess,
                    reload_state_dict, reverse_preprocess, save_state,
                    schedule_lr)

logger = getLogger()


def estimator_train(n_epoch, estimator, optimizer, train_dataset, params,
                    stats):
    """
    Train a PRNU estimator.
    """
    # estimator in train mode
    estimator.train()

    # iterate over the training dataset
    for n_iter, ((data_img, data_prnu), labels) in enumerate(train_dataset):
        # move data / labels to cuda
        data_img = data_img.to(device=params.primary_gpu)
        data_prnu = data_prnu.to(device=params.primary_gpu)
        labels = labels.to(device=params.primary_gpu)

        # estimate prnu
        prnu_hat = estimator(data_img)

        # compute loss
        loss = mse_loss(prnu_hat, data_prnu)

        # check NaN
        if (loss != loss).detach().any():
            logger.error("NaN detected")
            exit()

        # append loss to stats list
        stats.append(loss.item())

        # optimize estimator
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # average loss
        if len(stats) >= 25:
            logger.info("%06i - train loss: %.5f" % (n_iter + 1,
                                                     np.mean(stats)))
            stats = []


class Classifier_Trainer(object):
    """
    Train a classifier.
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
        with torch.no_grad():
            return est_prnu(data)

    def _prnu_low(self, est_prnu, data):
        """
        Returns the low-frequency components (data minus their corresponding
        PRNU estimates).
        """
        with torch.no_grad():
            return data - est_prnu(data)

    def _prnu_low_high(self, est_prnu, data):
        """
        Returns the low-frequency components concatenated with their
        corresponding high-frequency PRNU estimates.
        """
        with torch.no_grad():
            prnu_hat = est_prnu(data)
        return torch.cat((data - prnu_hat, prnu_hat), 1)

    def train(self, n_epoch, classifier, est_prnu, optimizer, train_dataset,
              params, stats):
        """
        Train the classifier for one epoch by iterating over the
        training dataset.
        """
        # classifier in train mode
        classifier.train()

        # estimator in eval mode
        if est_prnu is not None:
            est_prnu.eval()

        # iterate over the training dataset
        for n_iter, (data, labels) in enumerate(train_dataset):
            # move data / labels to cuda
            data = data.to(device=params.primary_gpu)
            labels = labels.to(device=params.primary_gpu)

            # get classifier input
            data = self.get_clf_input(est_prnu, data)

            # classify data
            logits = classifier(data)

            # compute loss
            loss = xent_loss(logits, labels)

            # predictions
            _, preds = torch.max(logits, 1)
            acc = preds.eq(labels).sum() / float(len(labels))

            # check NaN
            if (loss != loss).detach().any():
                logger.error("NaN detected")
                exit()

            # append loss / accuracy to stats list
            stats.append([loss.item(), acc.item()])

            # optimize classifier
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # average loss / accuracy
            if len(stats) >= 25:
                stats = np.array(stats)
                logger.info("%06i - train loss: %.5f - train acc: %.5f"
                            % (n_iter + 1, np.mean(stats[:, 0]),
                               np.mean(stats[:, 1])))
                stats = []

            # update constrained convolutional layer at each iteration
            if "con_conv" in params.clf_input:
                update_con_conv(classifier, params)


class GAN_Trainer(object):
    """
    Train a gan.
    """

    def __init__(self, gen, dis, clf_low, clf_high, est, train_dataset,
                 params):
        """
        Initialize gan training.
        """
        # data / parameters
        self.train_dataset = train_dataset
        self.params = params

        # models
        self.gen = gen
        self.dis = dis
        self.clf_low = clf_low
        self.clf_high = clf_high
        self.est = est

        # losses
        if params.pixel_loss == "l1":
            self.pixel_wise_loss = l1_loss
        else:
            self.pixel_wise_loss = mse_loss

        if self.dis is not None:
            self.gan_loss = gan_loss(use_lsgan=params.use_lsgan).to(
                device=params.primary_gpu)

        if (self.clf_low is not None) or (self.clf_high is not None):
            self.xent_loss = xent_loss

        # optimizers
        self.gen_optimizer = get_optimizer(gen, params.gen_optimizer)
        self.dis_optimizer = None
        if self.dis is not None:
            self.dis_optimizer = get_optimizer(dis,
                                               params.dis_optimizer)

        # reload saved optimizers
        if params.reload:
            reload_state_dict(params, params.reload, params.resume, [],
                              [], ["gen_optimizer"],
                              [self.gen_optimizer])
            if self.dis_optimizer is not None:
                reload_state_dict(params, params.reload, params.resume, [],
                                  [], ["dis_optimizer"],
                                  [self.dis_optimizer])

        # log architectural details / model parameters / optimizers using
        # torchsummary's summary
        logger.info("generator architecture. ")
        logger.info(summary(gen, [(params.ptc_fm, params.ptc_sz,
                                   params.ptc_sz), (1, 1, 1)]))
        if self.dis is not None:
            logger.info("discriminator architecture. ")
            logger.info(summary(dis,
                                [(2 * params.ptc_fm
                                  if ("+prnu" in params.dis_input)
                                  else params.ptc_fm,
                                  params.ptc_sz, params.ptc_sz), (1, 1, 1)]))

        logger.info("generator optimizer. ")
        logger.info(self.gen_optimizer)
        if self.dis_optimizer is not None:
            logger.info("discriminator optimizer. ")
            logger.info(self.dis_optimizer)

        # training statistics
        self.stats = {}
        self.stats["pixel_loss"] = []
        self.stats["adv_loss"] = []
        self.stats["dis_loss"] = []
        self.stats["clf_low_loss"] = []
        self.stats["clf_high_loss"] = []
        self.stats["total_gen_loss"] = []

        self.stats["d(x,y)"] = []
        self.stats["d(x,y')"] = []
        self.stats["d(g(x,y'),y')"] = []

        # best accuracy / image quality metrics
        self.best_acc = -1e12
        self.best_psnr = -1e12
        self.best_lpips = 1e12

        # valid (real) / invalid (fake) labels
        self.valid = torch.FloatTensor(1, 1).fill_(
            1.0).to(device=params.primary_gpu)
        self.invalid = torch.FloatTensor(1, 1).fill_(
            0.0).to(device=params.primary_gpu)

        # type of input to the generator (remosaic / rgb)
        if params.gen_input == "remosaic":
            self.transform_samples = self.transform_from_remosaic
        else:
            self.transform_samples = self.transform_from_rgb

        # discriminator's loss is based on the adversarial loss schedule
        if params.adv_loss_schedule[0] > 0:
            self.dis_forward = self.discriminate
            self.dis_backward = self.backward

            if "prnu_lp_low+prnu_lp" in params.dis_input:
                self.get_dis_input = self._prnu_low_high
            elif "prnu_lp_low" in params.dis_input:
                self.get_dis_input = self._prnu_low
            elif "prnu_lp" in params.dis_input:
                self.get_dis_input = self._prnu
            else:
                self.get_dis_input = self._rgb
        else:
            self.dis_forward = self.null_discriminate
            self.dis_backward = self.null_backward

        # initialize the number of classifiers (evaluators)
        self.n_clfs = 0.0

        # low-frequency classifier's loss is based on the provided schedule
        if ((params.clf_low_loss_schedule[0] > 0) and
                (self.clf_low is not None)):
            self.n_clfs += 1
            self.clf_low_forward = self.classify_low
            if "prnu_lp_low" in params.clf_low_input:
                self.get_clf_low_input = self._prnu_low
            else:
                self.get_clf_low_input = self._rgb
        else:
            self.clf_low_forward = self.null_classify

        # high-frequency classifier's loss is based on the provided schedule
        if ((params.clf_high_loss_schedule[0] > 0) and
                (self.clf_high is not None)):
            self.n_clfs += 1
            self.clf_high_forward = self.classify_high
            if "prnu_lp" in params.clf_high_input:
                self.get_clf_high_input = self._prnu
            else:
                self.get_clf_high_input = self._rgb
        else:
            self.clf_high_forward = self.null_classify

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

    def classify_low(self, data, target_labels):
        """
        Low-frequency classifier.
        """
        # get classifier input
        data = self.get_clf_low_input(data)

        # classify data
        logits = self.clf_low(data)

        # compute loss
        loss = (lambda_coeff(self.params.clf_low_loss_schedule, self.params) *
                self.xent_loss(logits, target_labels))

        # append loss to statistics
        self.stats["clf_low_loss"].append(loss.item())
        return loss

    def classify_high(self, data, target_labels):
        """
        High-frequency classifier.
        """
        # get classifier input
        data = self.get_clf_high_input(data)

        # classify data
        logits = self.clf_high(data)

        # compute loss
        loss = (lambda_coeff(self.params.clf_high_loss_schedule, self.params) *
                self.xent_loss(logits, target_labels))

        # append loss to statistics
        self.stats["clf_high_loss"].append(loss.item())
        return loss

    def null_classify(self, *args):
        """
        Null classifier.
        """
        loss = 0.0
        return loss

    def discriminate(self, data_rgb, gen_outputs, labels, target_labels):
        """
        Train the discriminator.
        """
        # get inputs for the discriminator
        data_rgb = self.get_dis_input(data_rgb)
        gen_outputs = self.get_dis_input(gen_outputs)

        # generator loss from the discriminator
        adv_loss = lambda_coeff(self.params.adv_loss_schedule, self.params) * \
            self.gan_loss(self.dis(gen_outputs, target_labels), self.valid)
        self.stats["adv_loss"].append(adv_loss.item())

        # discriminator loss: real data, true labels (valid)
        dis_valid_output = self.dis(data_rgb, labels)
        dis_valid_loss = self.gan_loss(dis_valid_output, self.valid)
        self.stats["d(x,y)"].append(dis_valid_output.mean().item())

        # discriminator loss: real data, false labels (invalid)
        dis_invalid1_output = self.dis(data_rgb, target_labels)
        dis_invalid1_loss = self.gan_loss(dis_invalid1_output, self.invalid)
        self.stats["d(x,y')"].append(dis_invalid1_output.mean().item())

        # discriminator loss: generated data (invalid)
        dis_invalid2_output = self.dis(gen_outputs.detach(), target_labels)
        dis_invalid2_loss = self.gan_loss(dis_invalid2_output, self.invalid)
        self.stats["d(g(x,y'),y')"].append(dis_invalid2_output.mean().item())

        # total discriminator loss
        dis_loss = 0.5 * (dis_valid_loss +
                          (0.5 * (dis_invalid1_loss + dis_invalid2_loss)))
        self.stats["dis_loss"].append(dis_loss.item())

        # check NaN
        if (dis_loss != dis_loss).detach().any():
            logger.error("NaN detected (discriminator)")
            exit()
        return adv_loss, dis_loss

    def null_discriminate(self, *args):
        """
        Null discriminator.
        """
        adv_loss, dis_loss = 0.0, 0.0
        return adv_loss, dis_loss

    def backward(self, optimizer, loss):
        """
        Backward step.
        """
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    def null_backward(self, *args):
        """
        Null backward step.
        """
        pass

    def gan_train(self, n_epoch):
        """
        Train the GAN.
        """
        params = self.params

        # generator / discriminator in train mode
        self.gen.train()
        if self.dis is not None:
            self.dis.train()

        # classifier(s) / estimator in eval mode
        if self.clf_low is not None:
            self.clf_low.eval()
        if self.clf_high is not None:
            self.clf_high.eval()
        if self.est is not None:
            self.est.eval()

        # iterate over the training dataset
        for n_iter, (data, labels) in enumerate(self.train_dataset):
            # split data into rgb and corresponding remosaiced versions. Note:
            # if the generator's input is "rgb" then data[1] = []
            data_rgb, data_remosaic = data[0], data[1]

            # generate target labels different from the ground truth labels
            target_labels = generate_targets(params.n_classes, labels)

            # move data / labels to cuda, except data_remosaic which may be []
            data_rgb = data_rgb.to(device=params.primary_gpu)
            labels = labels.to(device=params.primary_gpu)
            target_labels = target_labels.to(device=params.primary_gpu)

            # transform the data conditioned on their corresponding target
            # labels
            gen_outputs = self.transform_samples(
                data_rgb, data_remosaic, labels, target_labels)

            # compute low-frequency classifier loss
            clf_low_loss = self.clf_low_forward(gen_outputs, target_labels)

            # compute high-frequency classifier loss
            clf_high_loss = self.clf_high_forward(gen_outputs, target_labels)

            # compute adversarial / discriminator losses
            adv_loss, dis_loss = self.dis_forward(
                data_rgb, gen_outputs, labels, target_labels)

            # rescale all pixel values from [0,255] to [-1,1]
            data_rgb, = preprocess(data_rgb)
            gen_outputs = preprocess(gen_outputs)

            # compute pixel-wise image content loss
            pixel_loss = (lambda_coeff(params.pixel_loss_schedule, params) *
                          self.pixel_wise_loss(gen_outputs, data_rgb))
            self.stats["pixel_loss"].append(pixel_loss.item())

            # sum generator losses
            gen_loss = pixel_loss + adv_loss + clf_low_loss + clf_high_loss
            self.stats["total_gen_loss"].append(gen_loss.item())

            # check NaN
            if (gen_loss != gen_loss).detach().any():
                logger.error("NaN detected (generator)")
                exit()

            # optimize generator / discriminator
            self.backward(self.gen_optimizer, gen_loss)
            self.backward(self.dis_optimizer, dis_loss)

            # update the discriminator's constrained convolutional layer at
            # each iteration
            if "con_conv" in params.dis_input:
                update_con_conv(self.dis, params)

            # print training statistics
            self.step(n_iter + 1)

        # update learning rate if milestones are set
        if n_epoch in params.lr_milestones:
            # basic schedule which divides the current learning rate by 10
            schedule_lr(self.gen_optimizer)
            if self.dis_optimizer is not None:
                schedule_lr(self.dis_optimizer)

    def step(self, n_iter):
        """
        Print training statistics.
        """
        # compute the mean of the current set of statistics
        if len(self.stats['pixel_loss']) >= 25:
            mean_stats = [
                ("Pix.", "pixel_loss"),
                ("Adv.", "adv_loss"),
                ("Dis.", "dis_loss"),
                ("Clf(low).", "clf_low_loss") if len(
                    self.stats["clf_low_loss"]) >= 25 else (),
                ("Clf(high).", "clf_high_loss") if len(
                    self.stats["clf_high_loss"]) >= 25 else (),
                ("Tot.(gen.)", "total_gen_loss"),
                ("d(x,y)", "d(x,y)") if len(
                    self.stats["d(x,y)"]) >= 25 else (),
                ("d(x,y')", "d(x,y')") if len(
                    self.stats["d(x,y')"]) >= 25 else (),
                ("d(g(x,y'),y')", "d(g(x,y'),y')") if len(
                    self.stats["d(g(x,y'),y')"]) >= 25 else (),
            ]

            # remove empty tuples
            mean_stats = [x for x in mean_stats if x]

            logger.info(("%06i - " % n_iter) +
                        " / ".join(["%s : %.5f" % (a, np.mean(self.stats[b]))
                                    for a, b in mean_stats
                                    if len(self.stats[b]) > 0]))

            del self.stats["pixel_loss"][:]
            del self.stats["adv_loss"][:]
            del self.stats["dis_loss"][:]
            del self.stats["clf_low_loss"][:]
            del self.stats["clf_high_loss"][:]
            del self.stats["total_gen_loss"][:]
            del self.stats["d(x,y)"][:]
            del self.stats["d(x,y')"][:]
            del self.stats["d(g(x,y'), y')"][:]

        # number of iterations completed
        self.params.n_iters_done += 1

    def save_best_periodic(self, to_log):
        """
        Save the best model states / periodically save the model states.
        """
        # save the current state if the current accuracy is the best
        # historically
        mean_acc = 0.0
        if (self.params.clf_low_loss_schedule[0] > 0):
            mean_acc += to_log["clf_low_acc"]
        if (self.params.clf_high_loss_schedule[0] > 0):
            mean_acc += to_log["clf_high_acc"]
        mean_acc /= np.maximum(self.n_clfs, 1.0)

        if ((self.params.clf_low_loss_schedule[0] > 0) or
                (self.params.clf_high_loss_schedule[0] > 0)):
            if mean_acc > self.best_acc:
                self.best_acc = mean_acc
                logger.info("best (mean) acc: %.3f%%" % (self.best_acc * 100))

                save_state(self.params, to_log["n_epoch"],
                           ["psnr", "lpips", "acc"],
                           [to_log["psnr"], to_log["lpips"], mean_acc],
                           "best_acc",
                           ["generator"] + (["discriminator"]
                                            if self.dis is not None else []),
                           [self.gen] + ([self.dis]
                                         if self.dis is not None else []),
                           ["gen_optimizer"] + (["dis_optimizer"]
                                                if self.dis is not None
                                                else []),
                           [self.gen_optimizer] + ([self.dis_optimizer]
                                                   if self.dis is not None
                                                   else []))
        else:
            mean_acc = None

        # save the current state if the current psnr is the best historically
        if to_log["psnr"] > self.best_psnr:
            self.best_psnr = to_log["psnr"]
            logger.info("best psnr: %.3f" % self.best_psnr)
            save_state(self.params, to_log["n_epoch"],
                       ["psnr", "lpips", "acc"],
                       [to_log["psnr"], to_log["lpips"], mean_acc],
                       "best_psnr",
                       ["generator"] + (["discriminator"]
                                        if self.dis is not None else []),
                       [self.gen] + ([self.dis]
                                     if self.dis is not None else []),
                       ["gen_optimizer"] + (["dis_optimizer"]
                                            if self.dis is not None else []),
                       [self.gen_optimizer] + ([self.dis_optimizer]
                                               if self.dis is not None
                                               else []))

        # save the current state if the current lpips is the best historically
        if to_log["lpips"] < self.best_lpips:
            self.best_lpips = to_log["lpips"]
            logger.info("best lpips: %.5f" % self.best_lpips)
            save_state(self.params, to_log["n_epoch"],
                       ["psnr", "lpips", "acc"],
                       [to_log["psnr"], to_log["lpips"], mean_acc],
                       "best_lpips",
                       ["generator"] + (["discriminator"]
                                        if self.dis is not None else []),
                       [self.gen] + ([self.dis]
                                     if self.dis is not None else []),
                       ["gen_optimizer"] + (["dis_optimizer"]
                                            if self.dis is not None else []),
                       [self.gen_optimizer] + ([self.dis_optimizer]
                                               if self.dis is not None
                                               else []))

        # save the state every 5 epochs regardless of the metrics
        if (to_log["n_epoch"]) % 5 == 0 and (to_log["n_epoch"]) > 1:
            save_state(self.params, to_log["n_epoch"],
                       ["psnr", "lpips", "acc"],
                       [to_log["psnr"], to_log["lpips"], mean_acc],
                       "periodic-%i" % to_log["n_epoch"],
                       ["generator"] + (["discriminator"]
                                        if self.dis is not None else []),
                       [self.gen] + ([self.dis]
                                     if self.dis is not None else[]),
                       ["gen_optimizer"] + (["dis_optimizer"]
                                            if self.dis is not None else []),
                       [self.gen_optimizer] + ([self.dis_optimizer]
                                               if self.dis is not None
                                               else []))
