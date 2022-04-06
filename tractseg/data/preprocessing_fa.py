"""
Run this script to crop images + segmentations to brain area. Then save as nifti.
Reduces datasize and therefore IO by at least factor of 2.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from os.path import join

import nibabel as nib
import numpy as np
from joblib import Parallel, delayed

from tractseg.libs.system_config import SystemConfig as C
from tractseg.libs import data_utils
from tractseg.data.subjects import get_all_subjects
from tractseg.libs import exp_utils


#todo: adapt
dataset = "HCP_final"
DATASET_FOLDER = "HCP_for_training_COPY"  # source folder
DATASET_FOLDER_PREPROC = "HCP_preproc"  # target folder



def create_preprocessed_files():
    input_path = '/data3/wanliu/HCP_Wasser/HCP_for_training_COPY'
    output_path = '/data3/wanliu/HCP_Wasser/HCP_preproc'
    subjects = ["992774", "991267", "987983", "984472", "983773", "979984", "978578", "965771", "965367", "959574",
                "958976", "957974", "951457", "932554", "930449", "922854", "917255", "912447", "910241",
                "904044", "901442", "901139", "901038", "899885", "898176", "896879", "896778", "894673", "889579",
                "877269", "877168", "872764", "872158", "871964", "871762", "865363", "861456", "859671",
                "857263", "856766", "849971", "845458", "837964", "837560", "833249", "833148", "826454", "826353",
                "816653", "814649", "802844", "792766", "792564", "789373", "786569", "784565", "782561", "779370",
                "771354", "770352", "765056", "761957", "759869", "756055", "753251", "751348", "749361", "748662",
                "748258", "742549", "734045", "732243", "729557", "729254", "715647", "715041", "709551", "705341",
                "704238", "702133", "695768", "690152", "687163", "685058", "683256", "680957", "679568", "677968",
                "673455", "672756", "665254", "654754", "645551", "644044", "638049", "627549", "623844", "622236",
                "620434", "613538"]

    # subjects=['992774',"991267",]
    for subject in subjects:
        print('subject: {}'.format(subject))
        peak_path = join(input_path, subject, "mrtrix_peaks.nii.gz")
        data = nib.load(peak_path).get_fdata()
        _, _, bbox, _ = data_utils.crop_to_nonzero(np.nan_to_num(data))

        FA_path = join(input_path, subject, "FA.nii.gz")
        MD_path = join(input_path, subject, "MD.nii.gz")
        RGB_path = join(input_path, subject, "RGB.nii.gz")
        FA_nii = nib.load(FA_path)
        FA_affine = FA_nii.affine
        FA = np.expand_dims(FA_nii.get_fdata(), axis =-1)
        FA, _, _, _ = data_utils.crop_to_nonzero(FA, bbox=bbox)
        MD = np.expand_dims(nib.load(MD_path).get_fdata(), axis =-1)
        MD, _, _, _ = data_utils.crop_to_nonzero(MD, bbox=bbox)
        RGB = nib.load(RGB_path).get_fdata()
        RGB, _, _, _ = data_utils.crop_to_nonzero(RGB, bbox=bbox)


        nib.save(nib.Nifti1Image(FA, FA_affine), join(output_path, subject, "FA.nii.gz"))
        nib.save(nib.Nifti1Image(MD, FA_affine), join(output_path, subject, "MD.nii.gz"))
        nib.save(nib.Nifti1Image(RGB, FA_affine), join(output_path, subject, "RGB.nii.gz"))
    print('finish')


if __name__ == "__main__":
     create_preprocessed_files()
