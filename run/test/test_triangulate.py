# ------------------------------------------------------------------------------
# pose.pytorch
# Copyright (c) 2018-present Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Chunyu Wang (chnuwa@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np

import _init_paths
import dataset
from core.config import config
from core.config import update_config
from multiviews.cameras import camera_to_world_frame
from multiviews.triangulate import triangulate_poses


def parse_args():
    parser = argparse.ArgumentParser(
        description='Test Recursive Pictorial Structure Model')
    parser.add_argument(
        '--cfg', help='experiment configure file name', required=True, type=str)
    args, rest = parser.parse_known_args()
    update_config(args.cfg)
    return args


def main():
    parse_args()
    test_dataset = eval('dataset.' + config.DATASET.TEST_DATASET)(
        config, config.DATASET.TEST_SUBSET, False)
    grouping = test_dataset.grouping

    items = grouping[0]
    cameras = []
    poses2d = []
    for item in items:
        cam = test_dataset.db[item]['camera']
        cameras.append(cam)
        poses2d.append(test_dataset.db[item]['joints_2d'])
        gt = test_dataset.db[item]['joints_3d']
        gt = camera_to_world_frame(gt, cam['R'], cam['T'])

    poses2d = np.array(poses2d)
    poses3d = np.squeeze(triangulate_poses(cameras, poses2d))
    print('Recovered Pose by Triangulation: ', poses3d)
    print('Ground truth pose: ', gt)


if __name__ == '__main__':
    main()
