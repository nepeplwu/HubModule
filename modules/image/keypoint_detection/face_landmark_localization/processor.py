# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import cv2
import numpy as np
from collections import OrderedDict

__all__ = ['check_dir', 'postprocess']


def check_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    elif os.path.isfile(dir_path):
        os.remove(dir_path)
        os.makedirs(dir_path)


def postprocess(points, face_x1, face_y1, face_x2, face_y2, im_path,
                visualization):
    """
    postprocess ouput of network, one face at a time.
    """
    org_points = list()
    im = cv2.imread(im_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    for i in range(int(len(points) / 2)):
        x = points[2 * i] * (face_x2 - face_x1) + face_x1
        y = points[2 * i + 1] * (face_y2 - face_y1) + face_y1
        cv2.circle(im, (int(x), int(y)), 1, (0, 0, 255), 2)
        org_points.append([x, y])
    cv2.imwrite(im_path, im)
    return org_points
