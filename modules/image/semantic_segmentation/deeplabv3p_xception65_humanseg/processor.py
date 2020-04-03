# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
from collections import OrderedDict

import cv2
import numpy as np

__all__ = ['postprocess']


def postprocess(data_out,
                org_im,
                org_im_shape,
                org_im_path,
                output_dir,
                visualization,
                thresh=120):
    """
    Postprocess output of network. one image at a time.

    Args:
        data_out (numpy.ndarray): output of network.
        org_im (numpy.ndarray): original image.
        org_im_shape (list): shape pf original image.
        org_im_path (list): path of riginal image.
        output_dir (str): output directory to store image.
        visualization (bool): whether to save image or not.
        thresh (float): threshold.

    Returns:
        result (list[dict]): The data of processed image.
    """
    result = list()
    for logit in data_out:
        logit = logit[1] * 255
        logit = cv2.resize(logit, (org_im_shape[1], org_im_shape[0]))
        ret, logit = cv2.threshold(logit, thresh, 0, cv2.THRESH_TOZERO)
        logit = 255 * (logit - thresh) / (255 - thresh)
        rgba = np.concatenate((org_im, np.expand_dims(logit, axis=2)), axis=2)

        if visualization:
            _check_dir(output_dir)
            save_name = os.path.splitext(
                os.path.basename(org_im_path))[0] + '.png'
            save_path = os.path.join(output_dir, save_name)
            save_path = _check_duplicate_name(save_path)
            cv2.imwrite(save_path, rgba)
            result.append({'save_path': save_path, 'data': rgba})
        else:
            result.append({'data': rgba})
    return result


def _check_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    elif os.path.isfile(dir_path):
        os.remove(dir_path)
        os.makedirs(dir_path)


def _check_duplicate_name(file_path):
    if os.path.exists(file_path):
        file_path = file_path + '_time={}.png'.format(
            round(time.time(), 6) * 1e6)
    return file_path
