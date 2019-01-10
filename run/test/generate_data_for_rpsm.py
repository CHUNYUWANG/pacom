# ------------------------------------------------------------------------------
# multiview.pose3d.pytorch
# Copyright (c) 2018-present Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Chunyu Wang (chnuwa@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pprint
import shutil
import pickle

import h5py
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms

import _init_paths
from core.config import config
from core.config import update_config
from core.config import update_dir
from core.config import get_model_name
from core.loss import JointsMSELoss
from core.function import train
from core.function import validate
from utils.utils import get_optimizer
from utils.utils import save_checkpoint, load_checkpoint
from utils.utils import create_logger
import dataset
import models


def parse_args():
    parser = argparse.ArgumentParser(description='Generate Data For RPSM')
    parser.add_argument(
        '--cfg', help='configuration file name', required=True, type=str)
    parser.add_argument(
        '--heatmap', help='heatmap file name', required=True, type=str)
    args, rest = parser.parse_known_args()
    update_config(args.cfg)
    return args


def main():
    args = parse_args()
    test_dataset = eval('dataset.' + config.DATASET.TEST_DATASET)(
        config, config.DATASET.TEST_SUBSET, False)
    grouping = test_dataset.grouping
    heatmaps = h5py.File(args.heatmap)['heatmaps']

    rpsm_db = []
    cnt = 0
    for items in grouping:
        for idx in items:
            datum = test_dataset.db[idx]
            hm = heatmaps[cnt]
            cnt += 1

            rpsm_datum = {
                'heatmap': hm,
                'cam_params': datum['camera'],
                'joints_3d_cam': datum['joints_3d'],
                'scale': datum['scale'],
                'center': datum['center']
            }
            rpsm_db.append(rpsm_datum)

    with open('data/testdata/rpsm_testdata.pkl', 'wb') as f:
        pickle.dump(rpsm_db, f)


if __name__ == '__main__':
    main()
