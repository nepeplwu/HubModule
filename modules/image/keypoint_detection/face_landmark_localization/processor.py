# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import base64
import os
import time
from collections import OrderedDict

import cv2
import numpy as np

__all__ = ['check_dir', 'postprocess']


def base64_to_cv2(b64str):
    data = base64.b64decode(b64str.encode('utf8'))
    data = np.fromstring(data, np.uint8)
    data = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return data


def check_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    elif os.path.isfile(dir_path):
        os.remove(dir_path)
        os.makedirs(dir_path)


def get_image_ext(image):
    if image.shape[2] == 4:
        return ".png"
    return ".jpg"


def postprocess(res, output_dir, visualization):
    """
    postprocess ouput of network, one face at a time.
    """
    output = []
    _cur_id = -1
    for idx, _result in enumerate(res):
        if _result['id'] != _cur_id:
            _cur_id = _result['id']
            output.append({'data': []})
        output[-1]['data'].append(_result['points'])

    idx = -1
    if visualization:
        check_dir(output_dir)
        for sample in output:
            orig_im = res[idx + 1]['orig_im']
            for points in sample['data']:
                idx += 1
                coord_left = res[idx]['x1']
                coord_right = res[idx]['x2']
                coord_top = res[idx]['y1']
                coord_bottom = res[idx]['y2']
                for x, y in points:
                    x = x * (coord_right - coord_left) + coord_left
                    y = y * (coord_bottom - coord_top) + coord_top
                    cv2.circle(orig_im, (int(x), int(y)), 1, (0, 0, 255), 2)
            orig_im_path = res[idx]['orig_im_path']
            ext = os.path.splitext(orig_im_path) if orig_im_path else ''
            ext = ext if ext else get_image_ext(orig_im)
            org_im_path = orig_im_path if orig_im_path else 'ndarray_{}{}'.format(
                time.time(), ext)
            im_name = os.path.basename(org_im_path)
            im_save_path = os.path.join(output_dir, im_name)
            sample['save_path'] = im_save_path
            cv2.imwrite(im_save_path, orig_im)

    return output
