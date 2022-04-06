                   # from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

import math
import matplotlib.pyplot as plt
import numpy as np
import openpyxl

import os
from os.path import join
import shutil
import nibabel as nib
import numpy as np

from dipy.core.gradients import gradient_table
import dipy.reconst.dti as dti
from dipy.reconst.dti import fractional_anisotropy,color_fa


##----------- remove the non-brain region of peaks and labels in the HCP_training_COPY file to HCP_preproc----------
def get_bbox_from_mask(mask, outside_value=0):
    mask_voxel_coords = np.where(mask != outside_value)
    minzidx = int(np.min(mask_voxel_coords[0]))
    maxzidx = int(np.max(mask_voxel_coords[0])) + 1
    minxidx = int(np.min(mask_voxel_coords[1]))
    maxxidx = int(np.max(mask_voxel_coords[1])) + 1
    minyidx = int(np.min(mask_voxel_coords[2]))
    maxyidx = int(np.max(mask_voxel_coords[2])) + 1
    return [[minzidx, maxzidx], [minxidx, maxxidx], [minyidx, maxyidx]]

def remove_nonbrain_area_HCP():
    ori_dir = 'Your_DataPath/HCP_for_training_COPY'
    new_dir = 'Your_DataPath/HCP_preproc'
    help_file = 'mrtrix_peaks.nii.gz'
    target_file_list = ['mrtrix_peaks.nii.gz', 'bundle_masks_72.nii.gz']
    subjects=os.listdir(ori_dir)
    for sub in subjects:
        print(sub)
        for target_file in target_file_list:
            ## extract crop area
            help_path = join(ori_dir, sub, help_file)
            help_data = np.nan_to_num(nib.load(help_path).get_fdata())
            bbox = get_bbox_from_mask(help_data)

            ## load the data
            ori_path = join(ori_dir, sub, target_file)
            new_path = join(new_dir, sub, target_file)
            large_nii = nib.load(ori_path)
            large_data = np.nan_to_num(large_nii.get_fdata())
            affine = large_nii.affine

            ## crop non-brain area
            data = large_data[bbox[0][0]:bbox[0][1], bbox[1][0]:bbox[1][1], bbox[2][0]:bbox[2][1],:]
            data_nii = nib.Nifti1Image(data.astype(np.float32), affine = affine)
            if os.path.exists(join(new_dir, sub))==0:
                os.makedirs(join(new_dir, sub))
            nib.save(data_nii, new_path)

if __name__ == "__main__":
    remove_nonbrain_area_HCP()

