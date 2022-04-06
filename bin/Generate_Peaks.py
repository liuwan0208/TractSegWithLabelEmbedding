#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import warnings
import argparse
import importlib
import os
from os.path import join
import sys
from pkg_resources import require
import nibabel as nib

from tractseg.libs.system_config import get_config_name
from tractseg.libs import exp_utils
from tractseg.libs import img_utils
from tractseg.libs import preprocessing
from tractseg.libs import plot_utils
from tractseg.libs import peak_utils
from tractseg.python_api import run_tractseg
from tractseg.libs.utils import bcolors
from tractseg.libs.system_config import SystemConfig as C
from tractseg.data import dataset_specific_utils

warnings.simplefilter("ignore", UserWarning)  # hide scipy warnings
warnings.simplefilter("ignore", FutureWarning)  # hide h5py warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")  # hide Cython benign warning
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")  # hide Cython benign warning

### ---- change the "-i" and "--csd_type"!!!!!----------
def main():
    parser = argparse.ArgumentParser(description="Segment white matter bundles in a Diffusion MRI image.",
                                        epilog="Written by Jakob Wasserthal. Please reference 'Wasserthal et al. "
                                               "TractSeg - Fast and accurate white matter tract segmentation'. "
                                               "https://doi.org/10.1016/j.neuroimage.2018.07.070'")

    parser.add_argument("-i", metavar="dir path", dest="input", default='Your_Init_DataPath')
    parser.add_argument("--make_bmask", default=True)

    parser.add_argument("--csd_type", metavar="csd|csd_msmt|csd_msmt_5tt", choices=["csd", "csd_msmt", "csd_msmt_5tt"],
                        help="Which MRtrix constrained spherical deconvolution (CSD) is used for peak generation.\n"
                             "'csd' [DEFAULT]: Standard CSD. Very fast.\n"
                             "'csd_msmt': Multi-shell multi-tissue CSD DHollander algorithm. Medium fast. Needs "
                             "more than one b-value shell.\n"
                             "'csd_msmt_5tt': Multi-shell multi-tissue CSD 5TT. Slow on large images. Needs more "
                             "than one b-value shell."
                             "Needs a T1 image with all non-brain area removed (a file "
                             "'T1w_acpc_dc_restore_brain.nii.gz' must be in the input directory).",
                        # default="csd_msmt") #----for multiple b value
                        default = "csd") #---- for single b value

    parser.add_argument("--raw_diffusion_input", action="store_true",
                        help="Provide a Diffusion nifti image as argument to -i. "
                             "Will calculate CSD and extract the mean peaks needed as input for TractSeg.",
                        default=True)

    parser.add_argument("--keep_intermediate_files", action="store_true",
                        help="Do not remove intermediate files like CSD output and peaks",
                        default=False)

    parser.add_argument("--preview", action="store_true", help="Save preview of some tracts as png. Requires VTK.",
                        default=False)

    parser.add_argument("--flip", action="store_true",
                        help="Flip output peaks of TOM along z axis to make compatible with MITK.",
                        default=False)

    parser.add_argument("--single_orientation", action="store_true",
                        help="Do not run model 3x along x/y/z orientation with subsequent mean fusion.",
                        default=False)

    parser.add_argument("--get_probabilities", action="store_true",
                        help="Output probability map instead of binary segmentation (without any postprocessing)",
                        default=False)

    parser.add_argument("--super_resolution", action="store_true",
                        help="Keep 1.25mm resolution of model instead of downsampling back to original resolution",
                        default=False)

    parser.add_argument("--uncertainty", action="store_true",
                        help="Create uncertainty map by monte carlo dropout (https://arxiv.org/abs/1506.02142)",
                        default=False)

    parser.add_argument("--no_postprocess", action="store_true",
                        help="Deactivate simple postprocessing of segmentations (removal of small blobs)",
                        default=False)

    parser.add_argument("--preprocess", action="store_true",
                        help="Move input image to MNI space (rigid registration of FA). "
                             "(Does not work together with csd_type=csd_msmt_5tt)",
                        default=False)

    parser.add_argument("--nr_cpus", metavar="n", type=int,
                        help="Number of CPUs to use. -1 means all available CPUs (default: -1)",
                        default=-1)

    parser.add_argument('--tract_segmentation_output_dir', metavar="folder_name",
                        help="name of bundle segmentations output folder (default: bundle_segmentations)",
                        default="bundle_segmentations")

    parser.add_argument('--TOM_output_dir', metavar="folder_name",
                        help="name of TOM output folder (default: TOM)",
                        default="TOM")

    parser.add_argument('--exp_name', metavar="folder_name", help="name of experiment - ONLY FOR TESTING",
                        default=None)

    parser.add_argument('--tract_definition', metavar="TractQuerier+|xtract", choices=["TractQuerier+", "xtract"],
                        help="Select which tract definitions to use. 'TractQuerier+' defines tracts mainly by their"
                             "cortical start and end region. 'xtract' defines tracts mainly by ROIs in white matter. "
                             "Both have their advantages and disadvantages. 'TractQuerier+' referes to the dataset "
                             "described the TractSeg NeuroImage paper. "
                             "NOTE 1: 'xtract' only works for output type 'tractseg_segmentation' and "
                             "'dm_regression'.",
                        default="TractQuerier+")

    parser.add_argument("--rescale_dm", action="store_true",
                        help="Rescale density map to [0,100] range. Original values can be very small and therefore "
                             "inconvenient to work with.",
                        default=False)

    parser.add_argument("--tract_segmentations_path", metavar="path",
                        help="Path to tract segmentations. Only needed for TOM. If empty will look for default "
                             "TractSeg output.",
                        default=None)

    parser.add_argument("--test", action="store_true",
                        help="Only needed for unittesting.",
                        default=False)

    parser.add_argument("--verbose", action="store_true", help="Show more intermediate output",
                        default=False)

    parser.add_argument('--version', action='version', version=require("TractSeg")[0].version)

    parser.add_argument("--output_type", metavar="tract_segmentation|endings_segmentation|TOM|dm_regression",
                        choices=["tract_segmentation", "endings_segmentation", "TOM", "dm_regression"],
                        help="TractSeg can segment not only bundles, but also the end regions of bundles. "
                             "Moreover it can create Tract Orientation Maps (TOM).\n"

                             "'tract_segmentation' [DEFAULT]: Segmentation of bundles (72 bundles).\n"
                             "'endings_segmentation': Segmentation of bundle end regions (72 bundles).\n"
                             "'TOM': Tract Orientation Maps (20 bundles).",
                        default="tract_segmentation")
    parser.add_argument("--single_output_file", action="store_true",
                        help="Output all bundles in one file (4D image)",
                        default=False)
    args = parser.parse_args()

    ####################################### Set more parameters #######################################

    input_type = "peaks"  # peaks|T1
    manual_exp_name = args.exp_name
    dropout_sampling = args.uncertainty
    input_path = args.input

    ####################################### Setup configuration #######################################

    if manual_exp_name is None:
        config_file = get_config_name(input_type, args.output_type, dropout_sampling=dropout_sampling,
                                      tract_definition=args.tract_definition)
        Config = getattr(importlib.import_module("tractseg.experiments.pretrained_models." +
                                                 config_file), "Config")()
    else:
        Config = exp_utils.load_config_from_txt(join(C.EXP_PATH,
                                                     exp_utils.get_manual_exp_name_peaks(manual_exp_name, "Part1"),
                                                     "Hyperparameters.txt"))

    Config = exp_utils.get_correct_labels_type(Config)
    Config.CSD_TYPE = args.csd_type
    Config.KEEP_INTERMEDIATE_FILES = args.keep_intermediate_files
    Config.VERBOSE = args.verbose
    Config.SINGLE_OUTPUT_FILE = args.single_output_file
    Config.FLIP_OUTPUT_PEAKS = args.flip
    Config.PREDICT_IMG = input_path is not None

    ####################################### Preprocessing #######################################
    subject_list = os.listdir(args.input)
    print(len(subject_list),':', subject_list)
    if args.raw_diffusion_input:
        for subject in subject_list:
            print(subject)
            Config.PREDICT_IMG_OUTPUT = os.path.join(args.input, subject, Config.TRACTSEG_DIR)
            print(Config.PREDICT_IMG_OUTPUT)
            exp_utils.make_dir(Config.PREDICT_IMG_OUTPUT)

            input = os.path.join(args.input, subject, 'data.nii.gz')
            bvecs = os.path.join(args.input, subject, 'bvecs')
            bvals = os.path.join(args.input, subject, 'bvals')

            if args.make_bmask == True:
                print('creat_ brain mask')
                brain_mask = preprocessing.create_brain_mask(input, os.path.join(args.input, subject))
            else:
                brain_mask = os.path.join(args.input, subject, 'nodif_brain_mask.nii.gz')

            if args.preprocess:
                if Config.EXPERIMENT_TYPE == "tract_segmentation" or Config.EXPERIMENT_TYPE == "dm_regression":
                    input, bvals, bvecs, brain_mask = preprocessing.move_to_MNI_space(input, bvals, bvecs, brain_mask,
                                                                                           Config.PREDICT_IMG_OUTPUT)
                    print('done1')
                else:
                    if not os.path.exists(join(Config.PREDICT_IMG_OUTPUT / "FA_2_MNI.mat")):
                        raise FileNotFoundError("Could not find file " + join(Config.PREDICT_IMG_OUTPUT / "FA_2_MNI.mat") +
                                                ". Run with options `--output_type tract_segmentation --preprocess` first.")
                    if not os.path.exists(join(Config.PREDICT_IMG_OUTPUT / "MNI_2_FA.mat")):
                        raise FileNotFoundError("Could not find file " + join(Config.PREDICT_IMG_OUTPUT / "MNI_2_FA.mat") +
                                                ". Run with options `--output_type tract_segmentation --preprocess` first.")

            print('----start----')
            preprocessing.create_fods(input, Config.PREDICT_IMG_OUTPUT, bvals, bvecs,
                                      brain_mask, Config.CSD_TYPE, nr_cpus=args.nr_cpus)
            print('----finish----')


if __name__ == '__main__':
    main()
