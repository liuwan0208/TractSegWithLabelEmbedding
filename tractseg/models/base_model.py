
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import glob
from os.path import join
import importlib
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adamax
from torch.optim import Adam
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn.functional as F


try:
    from apex import amp
    APEX_AVAILABLE = True
except ImportError:
    APEX_AVAILABLE = False
    pass

from tractseg.libs import pytorch_utils
from tractseg.libs import exp_utils
from tractseg.libs import metric_utils


class BaseModel:
    def __init__(self, Config, inference=False):
        self.Config = Config

        # Do not use during inference because uses a lot more memory
        if not inference:
            torch.backends.cudnn.benchmark = True

        if self.Config.NR_CPUS > 0:
            torch.set_num_threads(self.Config.NR_CPUS)

        if self.Config.SEG_INPUT == "Peaks" and self.Config.TYPE == "single_direction":
            NR_OF_GRADIENTS = self.Config.NR_OF_GRADIENTS
        elif self.Config.SEG_INPUT == "Peaks" and self.Config.TYPE == "combined":
            self.Config.NR_OF_GRADIENTS = 3 * self.Config.NR_OF_CLASSES
        else:
            self.Config.NR_OF_GRADIENTS = 33

        if self.Config.LOSS_FUNCTION == "soft_sample_dice":
            self.criterion = pytorch_utils.soft_sample_dice
        elif self.Config.LOSS_FUNCTION == "soft_batch_dice":
            self.criterion = pytorch_utils.soft_batch_dice
        elif self.Config.EXPERIMENT_TYPE == "peak_regression":
            if self.Config.LOSS_FUNCTION == "angle_length_loss":
                self.criterion = pytorch_utils.angle_length_loss
            elif self.Config.LOSS_FUNCTION == "angle_loss":
                self.criterion = pytorch_utils.angle_loss
            elif self.Config.LOSS_FUNCTION == "l2_loss":
                self.criterion = pytorch_utils.l2_loss
        elif self.Config.EXPERIMENT_TYPE == "dm_regression":
            # self.criterion = nn.MSELoss()   # aggregate by mean
            self.criterion = nn.MSELoss(size_average=False, reduce=True)   # aggregate by sum
        else:## LOSS_FUNCTION== default
            # self.criterion_prob = nn.Sigmoid()
            self.criterion_bce = nn.BCEWithLogitsLoss()
            self.criterion_mse = nn.MSELoss(size_average=False, reduce=True)
            self.criterion_kl = nn.KLDivLoss()


        NetworkClass = getattr(importlib.import_module("tractseg.models." + self.Config.MODEL.lower()),
                               self.Config.MODEL)
        ## do not use bn and use_dropout(the pool4)
        self.net = NetworkClass(embed_dim=self.Config.Embed_dim, n_input_channels=NR_OF_GRADIENTS, n_classes=self.Config.NR_OF_CLASSES,
                                n_filt=self.Config.UNET_NR_FILT, batchnorm=self.Config.BATCH_NORM,
                                dropout=self.Config.USE_DROPOUT, upsample=self.Config.UPSAMPLE_TYPE,)

        num_params = 0
        for name, param in self.net.named_parameters():
            num_params += param.numel()
        print('the param amount:',num_params)

        # MultiGPU setup
        # (Not really faster (max 10% speedup): GPU and CPU utility low)
        nr_gpus = torch.cuda.device_count()
        exp_utils.print_and_save(self.Config.EXP_PATH, "nr of gpus: {}".format(nr_gpus))
        self.net = nn.DataParallel(self.net)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        net = self.net.to(self.device)

        if self.Config.OPTIMIZER == "Adamax":
            self.optimizer = Adamax(net.parameters(), lr=self.Config.LEARNING_RATE,
                                    weight_decay=self.Config.WEIGHT_DECAY)
        elif self.Config.OPTIMIZER == "Adam":
            self.optimizer = Adam(net.parameters(), lr=self.Config.LEARNING_RATE,
                                  weight_decay=self.Config.WEIGHT_DECAY)
        else:
            raise ValueError("Optimizer not defined")

        if APEX_AVAILABLE and self.Config.FP16:
            # Use O0 to disable fp16 (might be a little faster on TitanX)
            self.net, self.optimizer = amp.initialize(self.net, self.optimizer, verbosity=0, opt_level="O1")
            if not inference:
                print("INFO: Using fp16 training")
        else:
            if not inference:
                print("INFO: Did not find APEX, defaulting to fp32 training")

        if self.Config.LR_SCHEDULE:
            ## adaptive lr: when a metric not change, reduce lr
            self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer,
                                                            mode=self.Config.LR_SCHEDULE_MODE,
                                                            patience=self.Config.LR_SCHEDULE_PATIENCE)

        if self.Config.LOAD_WEIGHTS:
            exp_utils.print_verbose(self.Config.VERBOSE, "Loading weights ... ({})".format(self.Config.WEIGHTS_PATH))
            self.load_model(self.Config.WEIGHTS_PATH)

        # Reset weights of last layer for transfer learning
        if self.Config.RESET_LAST_LAYER:
            self.net.conv_5 = nn.Conv2d(self.Config.UNET_NR_FILT, self.Config.NR_OF_CLASSES, kernel_size=1,
                                        stride=1, padding=0, bias=True).to(self.device)


    def calculate_lossweight(self, loss_seg, loss_rec):
        if loss_seg / loss_rec < 1:
            loss_rec_weight = 1e0
        elif loss_seg / loss_rec < 1e1:
            loss_rec_weight = 1e1
        elif loss_seg / loss_rec < 1e2:
            loss_rec_weight = 1e2
        elif loss_seg / loss_rec < 1e3:
            loss_rec_weight = 1e3
        elif loss_seg / loss_rec < 1e4:
            loss_rec_weight = 1e4
        elif loss_seg / loss_rec < 1e5:
            loss_rec_weight = 1e5
        elif loss_seg / loss_rec < 1e6:
            loss_rec_weight = 1e6
        elif loss_seg / loss_rec < 1e7:
            loss_rec_weight = 1e7
        elif loss_seg / loss_rec < 1e8:
            loss_rec_weight = 1e8
        elif loss_seg / loss_rec < 1e9:
            loss_rec_weight = 1e9
        elif loss_seg / loss_rec < 1e10:
            loss_rec_weight = 1e10
        elif loss_seg / loss_rec < 1e11:
            loss_rec_weight = 1e11
        elif loss_seg / loss_rec < 1e12:
            loss_rec_weight = 1e12
        elif loss_seg / loss_rec < 1e13:
            loss_rec_weight = 1e13
        elif loss_seg / loss_rec < 1e14:
            loss_rec_weight = 1e14
        elif loss_seg / loss_rec < 1e15:
            loss_rec_weight = 1e15
        elif loss_seg / loss_rec < 1e16:
            loss_rec_weight = 1e16
        elif loss_seg / loss_rec < 1e17:
            loss_rec_weight = 1e17
        elif loss_seg / loss_rec < 1e18:
            loss_rec_weight = 1e18
        elif loss_seg / loss_rec < 1e19:
            loss_rec_weight = 1e19
        elif loss_seg / loss_rec < 1e20:
            loss_rec_weight = 1e20
        elif loss_seg / loss_rec < 1e21:
            loss_rec_weight = 1e21
        elif loss_seg / loss_rec < 1e22:
            loss_rec_weight = 1e22
        elif loss_seg / loss_rec < 1e23:
            loss_rec_weight = 1e23
        elif loss_seg / loss_rec < 1e24:
            loss_rec_weight = 1e24
        else:
            loss_rec_weight = 1e25
        return loss_rec_weight

    def train(self, X, y, epoch_nr, lw_rec, lr_reset, first_bh, weight_factor=None):
        X = X.contiguous().cuda(non_blocking=True)  # (bs, features, x, y)
        y = y.contiguous().cuda(non_blocking=True)  # (bs, classes, x, y)
        self.net.train()
        self.optimizer.zero_grad()
        outputs, output_rec, encode_rec, encode_seg = self.net(X, y)  # (bs, classes, x, y)
        angle_err = None

        loss_rec_weight = lw_rec
        learnrate_reset = lr_reset
        first_batch = first_bh

        if weight_factor is not None:
            if len(y.shape) == 4:  # 2D
                weights = torch.ones((self.Config.BATCH_SIZE, self.Config.NR_OF_CLASSES,
                                      y.shape[2], y.shape[3])).cuda()
            else:  # 3D
                weights = torch.ones((self.Config.BATCH_SIZE, self.Config.NR_OF_CLASSES,
                                      y.shape[2], y.shape[3], y.shape[4])).cuda()
            bundle_mask = y > 0
            weights[bundle_mask.data] *= weight_factor  # 10

            if self.Config.EXPERIMENT_TYPE == "peak_regression":
                loss, angle_err = self.criterion(outputs, y, weights)
            else:# true
                loss = self.criterion_bce(outputs, y)

        else:
            if self.Config.LOSS_FUNCTION == "soft_sample_dice" or self.Config.LOSS_FUNCTION == "soft_batch_dice":
                loss = self.criterion(F.sigmoid(outputs), y)
            else:
                # adaptive loss weight of recovery
                loss_seg = self.criterion_bce(outputs, y)
                loss_rec = self.criterion_bce(output_rec, y)
                if loss_rec!=0 and (epoch_nr == 200 or epoch_nr==400 or epoch_nr==600 or epoch_nr==800) and first_batch == True:
                    learnrate_reset = True
                    loss_rec_weight = self.calculate_lossweight(loss_seg, loss_rec)
                    first_batch = False
                loss_rec = loss_rec * loss_rec_weight
                loss = loss_seg + loss_rec
                log_loss = 'loss seg {}, loss rec {}, lw_rec {}'.format(loss_seg, loss_rec, loss_rec_weight)


        if APEX_AVAILABLE and self.Config.FP16:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        self.optimizer.step()

        if self.Config.EXPERIMENT_TYPE == "peak_regression":
            f1 = metric_utils.calc_peak_length_dice_pytorch(self.Config.CLASSES, outputs.detach(), y.detach(),
                                                            max_angle_error=self.Config.PEAK_DICE_THR,
                                                            max_length_error=self.Config.PEAK_DICE_LEN_THR)
        elif self.Config.EXPERIMENT_TYPE == "dm_regression":
            f1 = pytorch_utils.f1_score_macro(y.detach() > self.Config.THRESHOLD, outputs.detach(),
                                              per_class=True, threshold=self.Config.THRESHOLD)
        else: # true
            f1 = pytorch_utils.f1_score_macro(y.detach(), F.sigmoid(outputs).detach(), per_class=True,
                                              threshold=self.Config.THRESHOLD)

        if self.Config.USE_VISLOGGER:
            probs = F.sigmoid(outputs)
        else:
            probs = None  # faster

        metrics = {}
        metrics["loss"] = loss.item()
        metrics["f1_macro"] = f1
        metrics["angle_err"] = angle_err if angle_err is not None else 0

        return probs, metrics, loss_rec_weight, learnrate_reset, first_batch, log_loss


    def test(self, X, y, lw_rec, weight_factor=None):
        with torch.no_grad():
            X = X.contiguous().cuda(non_blocking=True)
            y = y.contiguous().cuda(non_blocking=True)
        if self.Config.DROPOUT_SAMPLING:
            self.net.train()
        else:
            self.net.train(False)

        outputs, output_rec, encode_rec, encode_seg = self.net(X, y)  # (bs, classes, x, y)
        angle_err = None

        loss_rec_weight = lw_rec

        if weight_factor is not None: # false
            if len(y.shape) == 4:  # 2D
                weights = torch.ones((self.Config.BATCH_SIZE, self.Config.NR_OF_CLASSES,
                                      y.shape[2], y.shape[3])).cuda()
            else:  # 3D
                weights = torch.ones((self.Config.BATCH_SIZE, self.Config.NR_OF_CLASSES,
                                      y.shape[2], y.shape[3], y.shape[4])).cuda()
            bundle_mask = y > 0
            weights[bundle_mask.data] *= weight_factor
            if self.Config.EXPERIMENT_TYPE == "peak_regression":
                loss, angle_err = self.criterion(outputs, y, weights)
            else:
                loss = nn.BCEWithLogitsLoss(weight=weights)(outputs, y)

        else:
            if self.Config.LOSS_FUNCTION == "soft_sample_dice" or self.Config.LOSS_FUNCTION == "soft_batch_dice":
                loss = self.criterion(F.sigmoid(outputs), y)
            else: ## true
                loss_seg = self.criterion_bce(outputs, y)
                loss_rec = self.criterion_bce(output_rec, y) * loss_rec_weight
                loss = loss_seg + loss_rec
                log_loss = 'loss seg {}, loss rec {}, lw_rec {}'.format(loss_seg, loss_rec, loss_rec_weight)



        if self.Config.EXPERIMENT_TYPE == "peak_regression":
            f1 = metric_utils.calc_peak_length_dice_pytorch(self.Config.CLASSES, outputs.detach(), y.detach(),
                                                            max_angle_error=self.Config.PEAK_DICE_THR,
                                                            max_length_error=self.Config.PEAK_DICE_LEN_THR)
        elif self.Config.EXPERIMENT_TYPE == "dm_regression":
            f1 = pytorch_utils.f1_score_macro(y.detach() > self.Config.THRESHOLD, outputs.detach(),
                                              per_class=True, threshold=self.Config.THRESHOLD)
        else:
            f1 = pytorch_utils.f1_score_macro(y.detach(), F.sigmoid(outputs).detach(), per_class=True,
                                              threshold=self.Config.THRESHOLD)

        if self.Config.USE_VISLOGGER:
            probs = F.sigmoid(outputs)
        else:
            probs = None  # faster

        metrics = {}
        metrics["loss"] = loss.item()
        metrics["f1_macro"] = f1
        metrics["angle_err"] = angle_err if angle_err is not None else 0

        return probs, metrics, log_loss


    def predict(self, X):
        with torch.no_grad():
            X = torch.tensor(X, dtype=torch.float32).contiguous().to(self.device)

        if self.Config.DROPOUT_SAMPLING:
            self.net.train()
        else:
            self.net.train(False)

        outputs = self.net(X)
        # outputs = self.net(X, y)  # forward
        if self.Config.EXPERIMENT_TYPE == "peak_regression" or self.Config.EXPERIMENT_TYPE == "dm_regression":
            probs = outputs.detach().cpu().numpy()
        else:
            probs = F.sigmoid(outputs).detach().cpu().numpy()

        if self.Config.DIM == "2D":
            probs = probs.transpose(0, 2, 3, 1)  # (bs, x, y, classes)
        else:
            probs = probs.transpose(0, 2, 3, 4, 1)  # (bs, x, y, z, classes)
        return probs

    def save_model(self, metrics, epoch_nr, mode="f1"):
        print("  Saving weights...")
        # remove weights from previous epochs
        # for fl in glob.glob(join(self.Config.EXP_PATH, "best_weights_ep*")):
        #     os.remove(fl)
        try:
            # Actually is a pkl not a npz
            path = join(self.Config.EXP_PATH, "best_weights_ep" + str(epoch_nr) + ".npz")
            print('weights saving path is', path)

            pytorch_utils.save_checkpoint(
                join(self.Config.EXP_PATH, "best_weights_ep" + str(epoch_nr) + ".npz"),
                unet=self.net)
        except IOError:
            print("\nERROR: Could not save weights because of IO Error\n")
        self.Config.BEST_EPOCH = epoch_nr


    def load_model(self, path):
        if self.Config.RESET_LAST_LAYER:
            pytorch_utils.load_checkpoint_selectively(path, unet=self.net)
        else:
            pytorch_utils.load_checkpoint(path, unet=self.net)

    def print_current_lr(self):
        for param_group in self.optimizer.param_groups:
            exp_utils.print_and_save(self.Config.EXP_PATH, "current learning rate: {}".format(param_group['lr']))

    def reset_lr(self):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.Config.LEARNING_RATE
            # param_group['lr'] = 0.0001

