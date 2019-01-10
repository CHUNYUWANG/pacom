# ------------------------------------------------------------------------------
# multiview.pose3d.pytorch
# Copyright (c) 2018-present Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Chunyu Wang (chnuwa@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import torchvision.transforms as transforms

import _init_paths
import dataset
from core.config import config
from core.config import update_config
from utils.vis import save_batch_heatmaps


def parse_args():
    update_config('experiments/multiview_h36m/test/h36m_multiview_test.yaml')


def main():
    parse_args()
    train_dataset = eval('dataset.' + config.DATASET.TRAIN_DATASET)(
        config, config.DATASET.TRAIN_SUBSET, True,
        transforms.Compose([
            transforms.ToTensor(),
        ]))

    valid_dataset = eval('dataset.' + config.DATASET.TEST_DATASET)(
        config, config.DATASET.TEST_SUBSET, False,
        transforms.Compose([
            transforms.ToTensor(),
        ]))

    for i, t, w, m in train_dataset:
        i = i[0]
        t = t[0]
        w = w[0]
        m = m[0]
        i = i.view(1, 3, 256, 256)
        t = t.view(1, 20, 64, 64)
        save_batch_heatmaps(i, t, 'output/test/train_heatmaps.jpg')
        break

    for i, t, w, m in valid_dataset:
        i = i[0]
        t = t[0]
        w = w[0]
        m = m[0]
        i = i.view(1, 3, 256, 256)
        t = t.view(1, 20, 64, 64)
        save_batch_heatmaps(i, t, 'output/test/test_heatmaps.jpg')
        break

    # test the evaluate function
    valid_dataset = eval('dataset.' + config.DATASET.TEST_DATASET)(
        config, config.DATASET.TEST_SUBSET, False,
        transforms.Compose([
            transforms.ToTensor(),
        ]))

    predictions = []
    for i, t, w, m in valid_dataset:
        predictions.append(m[0]['joints_2d'])
        predictions.append(m[1]['joints_2d'])
        predictions.append(m[2]['joints_2d'])
        predictions.append(m[3]['joints_2d'])
    predictions = np.array(predictions)
    name_values, mean_pck = valid_dataset.evaluate(predictions)
    print(name_values)
    print(mean_pck)


if __name__ == '__main__':
    main()
