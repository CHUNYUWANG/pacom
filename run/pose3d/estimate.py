# ------------------------------------------------------------------------------
# pose.pytorch
# Copyright (c) 2018-present Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Chunyu Wang (chnuwa@microsoft.com)
# ------------------------------------------------------------------------------

import time
import parse
import pickle
import argparse
import numpy as np

import _init_paths
from core.config import config
from multiviews.body import HumanBody
from multiviews.tool import triangulate_points, procrustes
from pict_struct import Pict_Struct


import dataset


def parse_args():
    parser = argparse.ArgumentParser(description='Arguments for Triangulation')

    parser.add_argument('--pose2dPath',
                        help='path to the prediction file',
                        required=True,
                        type=str)

    parser.add_argument('--hmapPath',
                        help='path to the heatmap file',
                        required=True,
                        type=str)

    parser.add_argument('--constrainPath',
                        help='path to the pairwise constrain file',
                        required=True,
                        type=str)

    parser.add_argument('--camPath',
                        help='path to the camParams file',
                        default='cameras.h5',
                        type=str)

    parser.add_argument('--camNum',
                        help='num of cameras',
                        default=4,
                        type=int)

    parser.add_argument('--RTimes',
                        help='recursive infer times',
                        default=10,
                        type=int)

    parser.add_argument('--Procrustes',
                        help='indicate whether doing procrustes operation',
                        default='No',
                        type=str)

    parser.add_argument('--Mode',
                        help='triangulation, pict_struct or combination',
                        default='combination',
                        type=str)

    args = parser.parse_args()

    return args

def infer_meta_from_name(imgname):
    format_str = 's_{}_act_{}_subact_{}_ca_{}_{}'
    res = parse.parse(format_str, imgname)
    meta = {
        'subject': int(res[0]),
        'action': int(res[1]),
        'subaction': int(res[2]),
        'camID': int(res[3]),
        'videoID': int(res[4].split('.')[0])
    }
    return meta['subject'], meta['camID'], meta['action']


def break_limb_length(skel, pred, limb_length, thres=0.4):
    flag = False
    for node in skel:
        parent_idx = node['idx']
        children_idx = node['children']
        if len(children_idx) > 0:
            for child_idx in children_idx:
                expect_length = limb_length[(parent_idx, child_idx)]
                actual_length = np.linalg.norm(pred[parent_idx] - pred[child_idx])
                offset = np.abs(expect_length - actual_length)
                if offset > thres * expect_length:
                    flag = True
    return flag


def triangulation(cams_params, j2d_set, subj, cam_num):
    njoint = len(j2d_set) // cam_num
    point3d = []
    for k in range(njoint):
        p2d = j2d_set[k::njoint]
        p3d = triangulate_points(cams_params, p2d, subj=subj, cams_num=cam_num)
        point3d.append(p3d.tolist())
    point3d = np.asarray(point3d)
    return point3d


def error_computing_3d(point3d, j3d_set, R_set, T_set, Procrustes):

    errors = []
    for point3d_gt, R, T in zip(j3d_set, R_set, T_set):
        # translate p3d from world coordinates to camera coordinates
        point3d_pred_T = np.matmul(R, point3d.T - T)
        point3d_pred = point3d_pred_T.T

        if Procrustes == 'Yes':
            _, point3d_pred, _ = procrustes(point3d_gt, point3d_pred)

        mpjpe = np.linalg.norm(point3d_gt - point3d_pred, axis=1)
        errors.append(mpjpe)
    return errors

def load_gt_label():
    import torchvision.transforms as transforms

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    dataset = eval('dataset.' + config.DATASET.VAL_DATASET)(
        config,
        config.DATASET.TEST_SET,
        False,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    )
    grouping = dataset.grouping
    j3d_gt = []
    for m in grouping:
        for n in m:
            j3d_gt.append(dataset.db[n]['joints_3d_cam'])

    j3d_gt = np.array(j3d_gt)

    remains = [3, 2, 1, 4, 5, 6, 16, 15, 14, 11, 12 ,13]
    j3d_gt = j3d_gt[:, remains, :]

    return j3d_gt


def load_data(data_path):
    with open(data_path, 'rb') as f:
        dataset = pickle.load(f)

    joints2d = dataset['all_preds'][:, :, :2]
    joints3d = dataset['joints3d_gt']
    imgname = dataset['imgname']

    assert joints2d.shape[0] == joints3d.shape[0]

    return joints2d, joints3d, imgname


def load_constrain(file_path):
    with open(file_path, 'rb') as f:
        file = pickle.load(f)

    limb_length = file['limb_length']
    pairwise_constrain = file['pairwise_constrain']

    return limb_length, pairwise_constrain

def prepare_input(i, cam_num, j3d_gt, imgnames, cams_params, j2d_pred):
    j2d_set = []
    j3d_set = []
    R_set = []
    T_set = []

    for j in range(cam_num):
        index = cam_num * i + j

        j3d_set.append(j3d_gt[index])

        subject, camid, action_num = infer_meta_from_name(imgnames[index])

        R, T, f, c, k, p, name = cams_params[(subject, camid)]
        R_set.append(R)
        T_set.append(T)

        cam_name = 'subject{}_cam{}'.format(subject, camid)

        for k in range(j2d_pred[index].shape[0]):
            j2d_set.append((cam_name, j2d_pred[index][k]))

    return j2d_set, j3d_set, R_set, T_set, subject, action_num


def main():
    args = parse_args()
    print('Configuration: \n', args)

    cam_num = args.camNum
    cams_params = load_cameras(bpath=args.camPath)

    limb_length, pairwise_constrain = load_constrain(args.constrainPath)
    j2d_pred, j3d_gt, imgnames = load_data(args.pose2dPath)

    pict_struct = Pict_Struct(args.hmapPath, pairwise_constrain, limb_length, num_views=cam_num, times=args.RTimes)

    index_to_action_names = pict_struct.dataset.index_to_action_names()

    actions_hits = {}
    actions_wise_errors = {}

    errors = []
    compare_errors = dict(before=[], after=[])

    start = time.time()
    pict_struct_count = 0
    for i in range(len(imgnames) // cam_num):
        j2d_set, j3d_set, R_set, T_set, subject, action_num = prepare_input(i, cam_num, j3d_gt,
                                                                            imgnames, cams_params, j2d_pred)

        point3d = triangulation(cams_params, j2d_set, subject, cam_num)

        if args.Mode == 'pict_struct':
            point3d = pict_struct.infer(root_joint=point3d[config.DATASET.ROOTIDX], index=cam_num * i)

        elif args.Mode == 'combination':
            graph_humanbody = Graph_HumanBody()
            if break_limb_length(graph_humanbody.skel, point3d, limb_length):
                pose3d_pair = {'before': point3d}
                point3d = pict_struct.infer(root_joint=point3d[config.DATASET.ROOTIDX], index=cam_num * i)
                pose3d_pair['after'] = point3d

                # compute errors to compare before with after
                for key, points in pose3d_pair.items():
                    errs= error_computing_3d(points, j3d_set, R_set, T_set, args.Procrustes)
                    compare_errors[key].append(np.mean(errs))

                pict_struct_count += 1
        else:
            pass

        # compute all errors
        errs = error_computing_3d(point3d, j3d_set, R_set, T_set, args.Procrustes)
        errors += errs
        # print(i, np.mean(errors))

        # compute action wise errors
        action_name = index_to_action_names[action_num]
        if action_name not in actions_wise_errors:
            actions_wise_errors[action_name] = np.mean(errs)
            actions_hits[action_name] = 1
        else:
            actions_wise_errors[action_name] += np.mean(errs)
            actions_hits[action_name] += 1

    # for action wise mpjpe
    for k, _ in actions_wise_errors.items():
        actions_wise_errors[k] /= actions_hits[k]
    print('action wise errors \n', actions_wise_errors)

    # for compare before and after mpjpe
    if pict_struct_count > 0:
        for k, v in compare_errors.items():
            compare_errors[k] = np.mean(v)
        print(compare_errors)

    errors = np.array(errors).T
    print('config: {}, \n procrustes: {}, mode: {}, mean MPJPE: {}'.format(args.pose2dPath, args.Procrustes, args.Mode,
                                                              np.mean(errors)))
    print('joint-wise error: {}'.format(np.mean(errors, 1)))
    print('variance : {}'.format(np.var(np.mean(errors, 0))))

    end = time.time()
    print('do {} / {} times pictorial infer'.format(pict_struct_count, len(imgnames) // cam_num))
    print('time consuming: {} minutes'.format((end - start) / 60))
    print('----------------------------------------------------------------------------')


if __name__ == '__main__':
    main()
