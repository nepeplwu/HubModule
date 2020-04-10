# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
from collections import OrderedDict

import base64
import cv2
import numpy as np
from PIL import Image

__all__ = ['base64_to_cv2', 'postprocess']


def base64_to_cv2(b64str):
    data = base64.b64decode(b64str.encode('utf8'))
    data = np.fromstring(data, np.uint8)
    data = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return data


def check_dir(dir_path):
    """
    Create directory to save processed image.

    Args:
        dir_path (str): directory path to save images.

    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    elif os.path.isfile(dir_path):
        os.remove(dir_path)
        os.makedirs(dir_path)


def get_save_image_name(org_im, org_im_path, output_dir):
    """
    Get save image name from source image path.
    """
    # name prefix of original image
    org_im_name = os.path.split(org_im_path)[-1]
    im_prefix = os.path.splitext(org_im_name)[0]
    # extension
    img = Image.fromarray(org_im[:, :, ::-1])
    if img.mode == 'RGBA':
        ext = '.png'
    elif img.mode == 'RGB':
        ext = '.jpg'
    # save image path
    save_im_path = os.path.join(output_dir, im_prefix + ext)
    if os.path.exists(save_im_path):
        save_im_path = os.path.join(
            output_dir, im_prefix + 'time={}'.format(int(time.time())) + ext)

    return save_im_path


def postprocess(label_score_list, org_im, org_im_path, output_dir,
                visualization, confs_threshold):
    """
    Postprocess output of network. one image at a time.

    Args:
        data_out (numpy.ndarray): output of network.
        orig_im (numpy.ndarray): original image.
        orig_im_path (list): path of riginal image.
        output_dir (str): output directory to store image.
        visualization (bool): whether to save image or not.
        shrink (float): parameter to control the resize scale in preprocess.
        confs_threshold (float): confidence threshold.

    Returns:
        output (dict): keys are 'data' and 'path', the correspoding values are:
            data (list[dict]): 5 keys, where
                'left', 'top', 'right', 'bottom' are the coordinate of detection bounding box,
                'confidence' is the confidence this bbox.
            path (str): The path of original image.
    """
    output = dict()
    output['data'] = list()
    output['path'] = org_im_path

    label_idx = np.argsort(label_score_list)

    if visualization:
        check_dir(output_dir)
        save_im_path = get_save_image_name(org_im, org_im_path, output_dir)
        im_out = org_im.copy()
        if len(output['data']) > 0:
            for bbox in output['data']:
                cv2.rectangle(im_out, (bbox['left'], bbox['top']),
                              (bbox['right'], bbox['bottom']), (255, 255, 0), 2)
        cv2.imwrite(save_im_path, im_out)

    return output
