#!/usr/bin/env python

"""
This module is for training the model. See Readme.md for more details about training your own model.

Examples:
    Run local:
    $ ExpRunner --config=XXX

    Predicting with new config setup:
    $ ExpRunner --train=False --test=True --lw --config=XXX
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import warnings
import os
import importlib
import argparse
import pickle as pkl
from pprint import pprint
import distutils.util
from os.path import join
import os

import nibabel as nib
import numpy as np
import torch

from tractseg.libs import data_utils
from tractseg.libs import direction_merger
from tractseg.libs import exp_utils
from tractseg.libs import img_utils
from tractseg.libs import peak_utils
from tractseg.libs.system_config import SystemConfig as C
from tractseg.libs import trainer
from tractseg.data.data_loader_training import DataLoaderTraining as DataLoaderTraining2D
from tractseg.data.data_loader_training_3D import DataLoaderTraining as DataLoaderTraining3D
from tractseg.data.data_loader_inference import DataLoaderInference
from tractseg.data import dataset_specific_utils
from tractseg.models.base_model import BaseModel
from bin.utils_add import compute_dice_score

warnings.simplefilter("ignore", UserWarning)  # hide scipy warnings
warnings.simplefilter("ignore", FutureWarning)  # hide h5py warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")  # hide Cython benign warning
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")  # hide Cython benign warning


def main():
    parser = argparse.ArgumentParser(description="Train a network on your own data to segment white matter bundles.",
                                        epilog="Written by Jakob Wasserthal. Please reference 'Wasserthal et al. "
                                               "TractSeg - Fast and accurate white matter tract segmentation. "
                                               "https://doi.org/10.1016/j.neuroimage.2018.07.070)'")
    parser.add_argument("--config", metavar="name", help="Name of configuration to use",
                        default='my_custom_experiment')
    parser.add_argument("--train", metavar="True/False", help="Train network",
                        type=distutils.util.strtobool, default=False)
    parser.add_argument("--test", metavar="True/False", help="Test network",
                        type=distutils.util.strtobool, default=True)
    parser.add_argument("--seg", action="store_true", help="Create binary segmentation", default=True)
    parser.add_argument("--probs", action="store_true", help="Create probmap segmentation")
    parser.add_argument("--lw", action="store_true", help="Load weights of pretrained net", default=True)
    parser.add_argument("--only_val", action="store_true", help="only run validation")
    parser.add_argument("--en", metavar="name", help="Experiment name")
    parser.add_argument("--fold", metavar="N", help="Which fold to train when doing CrossValidation", type=int)
    parser.add_argument("--verbose", action="store_true", help="Show more intermediate output", default=True)
    args = parser.parse_args()


    Config = getattr(importlib.import_module("tractseg.experiments.base"), "Config")()
    if args.config:
        # Config.__dict__ does not work properly therefore use this approach
        Config = getattr(importlib.import_module("tractseg.experiments.custom." + args.config), "Config")()

    if args.en:
        Config.EXP_NAME = args.en ## my_custom_experiment

    Config.TRAIN = bool(args.train) ## True
    Config.TEST = bool(args.test) ## False
    Config.SEGMENT = args.seg ## False
    if args.probs:
        Config.GET_PROBS = True ## False
    if args.lw:
        Config.LOAD_WEIGHTS = args.lw ## False
    if args.fold:
        Config.CV_FOLD = args.fold ## 0
    if args.only_val:
        Config.ONLY_VAL = True ## False
    Config.VERBOSE = args.verbose ## True:show more intermedia output

    Config.MULTI_PARENT_PATH = join(C.EXP_PATH, Config.EXP_MULTI_NAME) ## '//home/wanliu/TractSeg/hcp_exp/'
    Config.EXP_PATH = join(C.EXP_PATH, Config.EXP_MULTI_NAME, Config.EXP_NAME) ## '/home/wanliu/TractSeg/hcp_exp/my_custom_experiment'

    ##### modify subject numbers
    Config.TRAIN_SUBJECTS, Config.VALIDATE_SUBJECTS, Config.TEST_SUBJECTS = dataset_specific_utils.get_cv_fold(Config.CV_FOLD,
                                                                                                  dataset=Config.DATASET)

    # if Config.WEIGHTS_PATH == "":
    #     Config.WEIGHTS_PATH = '/data3/wanliu/TractSeg_Run/hcp_exp/my_custom_experiment_x13/best_weights_ep336.npz'

    #Autoset input dimensions based on settings
    Config.INPUT_DIM = dataset_specific_utils.get_correct_input_dim(Config) ## (144,144)
    Config = dataset_specific_utils.get_labels_filename(Config) ## 'LABELS_FILENAME': 'bundle_masks_72'

    if Config.EXPERIMENT_TYPE == "peak_regression":
        Config.NR_OF_CLASSES = 3*len(dataset_specific_utils.get_bundle_names(Config.CLASSES)[1:])
    else:
        Config.NR_OF_CLASSES = len(dataset_specific_utils.get_bundle_names(Config.CLASSES)[1:]) ## 72

    if Config.TRAIN and not Config.ONLY_VAL:
        Config.EXP_PATH = exp_utils.create_experiment_folder(Config.EXP_NAME, Config.MULTI_PARENT_PATH, Config.TRAIN)

    Config = exp_utils.get_correct_labels_type(Config)



    if Config.TEST:
        Config.WEIGHTS_PATH = '/data4/wanliu/TractSeg_Run/hcp_exp3/my_custom_experiment_x36/best_weights_ep582.npz'
        Test_peak_dir = '/data3/wanliu/HCP_Wasser/HCP_for_training_COPY'
        #Test_peak_dir = '/data4/wanliu/HCP_Wasser_Clinical/HCP_for_training_COPY'
        save_path = '/data4/wanliu/TractSeg_Run/hcp_exp3/my_custom_experiment_x36'


        test_epoch = Config.WEIGHTS_PATH.split('/')[-1].split('_')[-1].split('.')[0]
        print("Loading weights in ", Config.WEIGHTS_PATH)
        model = BaseModel(Config, inference=True)
        model.load_model(Config.WEIGHTS_PATH)
        for subject in Config.TEST_SUBJECTS:
        #for subject in Config.VALIDATE_SUBJECTS:

            print("Get_segmentation subject {}".format(subject))
            peak_path = join(Test_peak_dir, subject, 'mrtrix_peaks.nii.gz')
            data_img = nib.load(peak_path)
            data_affine = data_img.affine
            data0 = data_img.get_fdata()
            data = np.nan_to_num(data0)
            data, _, bbox, original_shape = data_utils.crop_to_nonzero(data)

            data, transformation = data_utils.pad_and_scale_img_to_square_img(data, target_size=Config.INPUT_DIM[0],
                                                                              nr_cpus=-1)
            seg_xyz, _ = direction_merger.get_seg_single_img_3_directions(Config, model, data=data,
                                                                              scale_to_world_shape=False,
                                                                              only_prediction=True,
                                                                              batch_size=1)

            seg = direction_merger.mean_fusion(Config.THRESHOLD, seg_xyz, probs=False)
            seg = data_utils.cut_and_scale_img_back_to_original_img(seg, transformation, nr_cpus=-1)
            seg = data_utils.add_original_zero_padding_again(seg, bbox, original_shape, Config.NR_OF_CLASSES)

            ###
            # label_path = join(Test_peak_dir, subject, Config.LABELS_FILENAME + '.nii.gz')
            # label= nib.load(label_path).get_fdata()
            # print(label.shape)
            # print(seg.shape)
            # with open('/home/wanliu/TractSeg/prob1.txt','a') as f:
            #     f.write('This is subject {}!\n'.format(subject))
            #     for index in range(60,80):
            #         i = int(index)
            #         f.write('seg_predict_prob in voxel[{},{},{}] is \n{} \n'.format(i,i,i,seg[i, i, i, :]))
            #         f.write('corresponding label is \n{} \n'.format(label[i, i, i, :]))
            #         print('seg_predict_prob in voxel[{},{},{}] is \n{} \n'.format(i,i,i,seg[i, i, i, :]))
            #         print('corresponding label is \n{} \n'.format(label[i, i, i, :]))
            #         print('seg_predict_prob in voxel[{},{},{}] is \n{} \n'.format(i, i+20, i, seg[i, i, i, :]))
            #         print('corresponding label is \n{} \n'.format(label[i, i, i, :]))
            #

            print('save segmentation results')
            img_seg = nib.Nifti1Image(seg.astype(np.uint8), data_affine)
            output_all_bund = join(save_path, "segmentation" + '_' + test_epoch, 'all_bund_seg')
            exp_utils.make_dir(output_all_bund)
            print(output_all_bund)
            nib.save(img_seg, join(output_all_bund, subject + ".nii.gz"))

            # bundles = dataset_specific_utils.get_bundle_names(Config.CLASSES)[1:]
            # output_indiv_bund = join(save_path, "segmentation" + '_' + test_epoch, 'indiv_bund_seg', subject)
            # exp_utils.make_dir(output_indiv_bund)
            # print(output_indiv_bund)
            # for idx, bundle in enumerate(bundles):
            #     img_seg = nib.Nifti1Image(seg[:, :, :, idx], data_affine)
            #     nib.save(img_seg, join(output_indiv_bund, bundle + ".nii.gz"))


        ## -----------------------Compute mean Dice of each tract------------------------
        print('computing dice coeff')
        all_subjects = Config.TEST_SUBJECTS
        #all_subjects = Config.VALIDATE_SUBJECTS
        Seg_path = join(save_path, "segmentation" + '_' + test_epoch, 'all_bund_seg')
        Label_path = Test_peak_dir
        Dice_all = np.zeros([len(all_subjects), 72])
        print(Dice_all.shape)
        print(C.DATA_PATH)
        for subject_index in range(len(all_subjects)):
            subject = all_subjects[subject_index]
            print("Get_test subject {}".format(subject))
            seg_path = join(Seg_path, subject + ".nii.gz")
            label_path = join(Label_path, subject, Config.LABELS_FILENAME + '.nii.gz')
            print(seg_path)
            print(label_path)
            seg = nib.load(seg_path).get_fdata()
            label= nib.load(label_path).get_fdata()
            for tract_index in range(label.shape[-1]):
                dice = compute_dice_score(seg[:,:,:,tract_index], label[:,:,:,tract_index])
                Dice_all[subject_index, tract_index] = dice
            with open(join(save_path, "test_dice_"+str(test_epoch)+".txt"), 'a') as f:
                f.write('Dice of subject {} is \n {} \n'.format(subject, Dice_all[subject_index, :]))

        Dice_mean = np.mean(Dice_all, 0)
        Dice_average = np.mean(Dice_all)
        bundles = ['AF_left', 'AF_right', 'ATR_left', 'ATR_right', 'CA', 'CC_1', 'CC_2', 'CC_3', 'CC_4', 'CC_5', 'CC_6',
                   'CC_7', 'CG_left', 'CG_right', 'CST_left', 'CST_right', 'MLF_left', 'MLF_right', 'FPT_left',
                   'FPT_right', 'FX_left', 'FX_right', 'ICP_left', 'ICP_right', 'IFO_left', 'IFO_right', 'ILF_left',
                   'ILF_right', 'MCP', 'OR_left', 'OR_right', 'POPT_left', 'POPT_right', 'SCP_left', 'SCP_right',
                   'SLF_I_left', 'SLF_I_right', 'SLF_II_left', 'SLF_II_right', 'SLF_III_left', 'SLF_III_right',
                   'STR_left', 'STR_right', 'UF_left', 'UF_right', 'CC', 'T_PREF_left', 'T_PREF_right', 'T_PREM_left',
                   'T_PREM_right', 'T_PREC_left', 'T_PREC_right', 'T_POSTC_left', 'T_POSTC_right', 'T_PAR_left',
                   'T_PAR_right', 'T_OCC_left', 'T_OCC_right', 'ST_FO_left', 'ST_FO_right', 'ST_PREF_left',
                   'ST_PREF_right', 'ST_PREM_left', 'ST_PREM_right', 'ST_PREC_left', 'ST_PREC_right', 'ST_POSTC_left',
                   'ST_POSTC_right', 'ST_PAR_left', 'ST_PAR_right', 'ST_OCC_left', 'ST_OCC_right']

        with open(join(save_path, "test_dice_"+str(test_epoch)+".txt"),'a') as f:
            for index in range(72):
                log = '{}: {} \n'.format(bundles[index], Dice_mean[index])
                f.write(log)
            log = 'mean dice of all tract is:{}\n'.format(Dice_average)
            f.write(log)
            print(log)



####--------------Directly calculate dice by output.nii.gz file and labels-------------------------------
    # DICE = bool(1)
    # print(DICE)
    # if DICE:
    #     print('computing dice coeff')
    #     all_subjects = Config.TEST_SUBJECTS
    #     Seg_path = join(Config.EXP_PATH, "segmentation" + '_' + test_epoch, 'all_bund_seg')
    #     Label_path = '/data3/wanliu/HCP_Wasser/HCP_for_training_COPY'
    #     Dice_all = np.zeros([len(all_subjects), 72])
    #     print(Dice_all.shape)
    #     print(C.DATA_PATH)
    #     for subject_index in range(len(all_subjects)):
    #         subject = all_subjects[subject_index]
    #         print("Get_test subject {}".format(subject))
    #         seg_path = join(Seg_path, subject + ".nii.gz")
    #         label_path = join(Label_path, subject, Config.LABELS_FILENAME + '.nii.gz')
    #         # label_path = join(Config.EXP_PATH, subject + "_segmentation_label.nii.gz")
    #         print(seg_path)
    #         print(label_path)
    #         seg = nib.load(seg_path).get_fdata()
    #         label= nib.load(label_path).get_fdata()
    #         for tract_index in range(label.shape[-1]):
    #             dice = compute_dice_score(seg[:,:,:,tract_index], label[:,:,:,tract_index])
    #             Dice_all[subject_index, tract_index] = dice
    #         with open(join(Config.EXP_PATH, "test_dice.txt"), 'a') as f:
    #             f.write('Dice of subject {} is \n {} \n'.format(subject, Dice_all[subject_index, :]))
    #
    #     Dice_mean = np.mean(Dice_all, 0)
    #     bundles = ['AF_left', 'AF_right', 'ATR_left', 'ATR_right', 'CA', 'CC_1', 'CC_2', 'CC_3', 'CC_4', 'CC_5', 'CC_6',
    #                'CC_7', 'CG_left', 'CG_right', 'CST_left', 'CST_right', 'MLF_left', 'MLF_right', 'FPT_left',
    #                'FPT_right', 'FX_left', 'FX_right', 'ICP_left', 'ICP_right', 'IFO_left', 'IFO_right', 'ILF_left',
    #                'ILF_right', 'MCP', 'OR_left', 'OR_right', 'POPT_left', 'POPT_right', 'SCP_left', 'SCP_right',
    #                'SLF_I_left', 'SLF_I_right', 'SLF_II_left', 'SLF_II_right', 'SLF_III_left', 'SLF_III_right',
    #                'STR_left', 'STR_right', 'UF_left', 'UF_right', 'CC', 'T_PREF_left', 'T_PREF_right', 'T_PREM_left',
    #                'T_PREM_right', 'T_PREC_left', 'T_PREC_right', 'T_POSTC_left', 'T_POSTC_right', 'T_PAR_left',
    #                'T_PAR_right', 'T_OCC_left', 'T_OCC_right', 'ST_FO_left', 'ST_FO_right', 'ST_PREF_left',
    #                'ST_PREF_right', 'ST_PREM_left', 'ST_PREM_right', 'ST_PREC_left', 'ST_PREC_right', 'ST_POSTC_left',
    #                'ST_POSTC_right', 'ST_PAR_left', 'ST_PAR_right', 'ST_OCC_left', 'ST_OCC_right']
    #
    #     with open(join(Config.EXP_PATH, "test_dice.txt"),'a') as f:
    #         for index in range(72):
    #             log = '{}: {} \n'.format(bundles[index], Dice_mean[index])
    #             f.write(log)
    #             print(log)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    main()
