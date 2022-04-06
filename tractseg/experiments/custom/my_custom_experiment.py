#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from tractseg.experiments.tract_seg import Config as TractSegConfig


class Config(TractSegConfig):
    EXP_NAME = os.path.basename(__file__).split(".")[0]
    DATASET_FOLDER = "HCP_preproc"      # name of folder that contains all the preprocessed subjects (each subject has its own folder with the name of the subjectID)
    FEATURES_FILENAME = "mrtrix_peaks"
    LABELS_FILENAME = 'bundle_masks_72'
