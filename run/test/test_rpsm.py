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
import pickle
import numpy as np

import _init_paths
from core.config import config
from core.config import update_config
from multiviews.pictorial import rpsm
from multiviews.cameras import camera_to_world_frame
from multiviews.body import HumanBody


def parse_args():
    parser = argparse.ArgumentParser(
        description='Test Recursive Pictorial Structure Model')
    parser.add_argument(
        '--cfg', help='experiment configure file name', required=True, type=str)
    args, rest = parser.parse_known_args()
    update_config(args.cfg)
    return args


def compute_limb_length(body, pose):
    limb_length = {}
    skeleton = body.skeleton
    for node in skeleton:
        idx = node['idx']
        children = node['children']

        for child in children:
            length = np.linalg.norm(pose[idx] - pose[child])
            limb_length[(idx, child)] = length
    return limb_length


def load_rpsm_testdata(testdata):
    with open(testdata, 'rb') as f:
        db = pickle.load(f)

    heatmaps = []
    cameras = []
    boxes = []
    poses = []
    for i in range(4):
        heatmap = db[i]['heatmap']
        heatmaps.append(heatmap)

        camera = db[i]['cam_params']
        cameras.append(camera)

        pose_camera = db[i]['joints_3d_cam']
        pose_world = camera_to_world_frame(pose_camera, camera['R'],
                                           camera['T'])
        poses.append(pose_world)

        box = {}
        box['scale'] = np.array(db[i]['scale'])
        box['center'] = np.array(db[i]['center'])
        boxes.append(box)
    hms = np.array(heatmaps)

    grid_center = poses[0][0]
    body = HumanBody()
    limb_length = compute_limb_length(body, poses[0])

    return cameras, hms, boxes, grid_center, limb_length, poses[0]


def main():
    parse_args()
    test_data_file = 'data/testdata/rpsm_testdata.pkl'
    pairwise_file = 'data/testdata/pairwise.pkl'
    cameras, hms, boxes, grid_center, limb_length, gt = load_rpsm_testdata(
        test_data_file)
    with open(pairwise_file, 'rb') as f:
        pairwise = pickle.load(f)
        pairwise = pairwise['pairwise_constrain']
    pose = rpsm(cameras, hms, boxes, grid_center, limb_length, pairwise, config)

    print('GroundTruth Pose: ', gt)
    print('Recovered Pose by RPSM: ', pose)
    mpjpe = np.mean(np.sqrt(np.sum((pose - gt)**2, axis=1)))
    print('MPJPE: ', mpjpe)


if __name__ == '__main__':
    main()
