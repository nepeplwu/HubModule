# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from collections import OrderedDict

import cv2
import numpy as np

__all__ = ['postprocess']


def get_max_preds(batch_heatmaps):
    """
    Get predictions from score maps.

    Args:
        batch_heatmaps (numpy.ndarray): output of the network, with shape [N, C, H, W]
    """
    assert isinstance(batch_heatmaps, np.ndarray), \
        'batch_heatmaps should be numpy.ndarray'
    assert batch_heatmaps.ndim == 4, 'batch_images should be 4-ndim'
    batch_size = batch_heatmaps.shape[0]
    num_joints = batch_heatmaps.shape[1]
    width = batch_heatmaps.shape[3]
    heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))
    idx = np.argmax(heatmaps_reshaped, 2)
    maxvals = np.amax(heatmaps_reshaped, 2)
    maxvals = maxvals.reshape((batch_size, num_joints, 1))
    idx = idx.reshape((batch_size, num_joints, 1))
    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)
    preds[:, :, 0] = (preds[:, :, 0]) % width
    preds[:, :, 1] = np.floor((preds[:, :, 1]) / width)
    pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2))
    pred_mask = pred_mask.astype(np.float32)
    preds *= pred_mask
    return preds, maxvals


def save_predict_results(batch_heatmaps):
    batch_size, num_joints, heatmap_height, heatmap_width = batch_heatmaps.shape
    preds, maxvals = get_max_preds(batch_heatmaps)
    return preds[0] * 4, num_joints


def postprocess(out_heatmaps, org_im, org_im_shape, org_im_path, output_dir,
                visualization):
    """
    Postprocess output of network. one image at a time.

    Args:
        out_heatmaps (numpy.ndarray): output of network.
        org_im (numpy.ndarray): original image.
        org_im_shape (list): shape pf original image.
        org_im_path (list): path of riginal image.
        output_dir (str): output directory to store image.
        visualization (bool): whether to save image or not.
    """
    preds, num_joints = save_predict_results(out_heatmaps)
    scale_horizon = org_im_shape[1] * 1.0 / 384
    scale_vertical = org_im_shape[0] * 1.0 / 384
    preds = np.multiply(preds, (scale_horizon, scale_vertical)).astype(int)
    if visualization:
        icolor = (255, 137, 0)
        ocolor = (138, 255, 0)
        rendered_im = org_im.copy()
        for j in range(num_joints):
            x, y = preds[j]
            cv2.circle(rendered_im, (x, y), 3, icolor, -1, 16)
            cv2.circle(rendered_im, (x, y), 6, ocolor, 1, 16)
        # check whether output_dir is existent or not
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        elif os.path.isfile(output_dir):
            os.remove(output_dir)
            os.makedirs(output_dir)
        # save image
        save_im_name = os.path.join(
            output_dir, 'rendered_{}.jpg'.format(
                os.path.splitext(os.path.basename(org_im_path))[0]))
        cv2.imwrite(save_im_name, rendered_im)
        print('image saved in {}'.format(save_im_name))

    # articulation
    articulation_points = OrderedDict()
    articulation_points['left_ankle'] = list(preds[0])
    articulation_points['left_knee'] = list(preds[1])
    articulation_points['left_hip'] = list(preds[2])
    articulation_points['right_hip'] = list(preds[3])
    articulation_points['right_knee'] = list(preds[4])
    articulation_points['right_ankle'] = list(preds[5])
    articulation_points['pelvis'] = list(preds[6])
    articulation_points['thorax'] = list(preds[7])
    articulation_points['upper neck'] = list(preds[8])
    articulation_points['head top'] = list(preds[9])
    articulation_points['right_wrist'] = list(preds[10])
    articulation_points['right_elbow'] = list(preds[11])
    articulation_points['right_shoulder'] = list(preds[12])
    articulation_points['left_shoulder'] = list(preds[13])
    articulation_points['left_elbow'] = list(preds[14])
    articulation_points['left_wrist'] = list(preds[15])
    return articulation_points
