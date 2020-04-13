# coding=utf-8
import os
import math
import time
from collections import OrderedDict

import cv2
import numpy as np
from PIL import Image

__all__ = ['reader']


def crop(image,
         pts,
         shift=0,
         scale=1.5,
         rotate=0,
         res_width=128,
         res_height=128):
    res = (res_width, res_height)
    idx1 = 0
    idx2 = 1
    # angle
    alpha = 0
    if pts[idx2, 0] != -1 and pts[idx2, 1] != -1 and pts[idx1, 0] != -1 and pts[
            idx1, 1] != -1:
        alpha = math.atan2(pts[idx2, 1] - pts[idx1, 1],
                           pts[idx2, 0] - pts[idx1, 0]) * 180 / math.pi
    pts[pts == -1] = np.inf
    coord_min = np.min(pts, 0)
    pts[pts == np.inf] = -1
    coord_max = np.max(pts, 0)
    # coordinates of center point
    c = np.array([
        coord_max[0] - (coord_max[0] - coord_min[0]) / 2,
        coord_max[1] - (coord_max[1] - coord_min[1]) / 2
    ])  # center
    max_wh = max((coord_max[0] - coord_min[0]) / 2,
                 (coord_max[1] - coord_min[1]) / 2)
    # Shift the center point, rot add eyes angle
    c = c + shift * max_wh
    rotate = rotate + alpha
    M = cv2.getRotationMatrix2D((c[0], c[1]), rotate,
                                res[0] / (2 * max_wh * scale))
    M[0, 2] = M[0, 2] - (c[0] - res[0] / 2.0)
    M[1, 2] = M[1, 2] - (c[1] - res[0] / 2.0)
    image_out = cv2.warpAffine(image, M, res)
    return image_out, M


def color_normalize(image, mean, std=None):
    if image.shape[-1] == 1:
        image = np.repeat(image, axis=2)
    h, w, c = image.shape
    image = np.transpose(image, (2, 0, 1))
    image = np.subtract(image.reshape(c, -1), mean[:, np.newaxis]).reshape(
        -1, h, w)
    image = np.transpose(image, (1, 2, 0))
    return image


def process_image(org_im, face):
    pts = np.array([
        face['left'], face['top'], face['right'], face['top'], face['left'],
        face['bottom'], face['right'], face['bottom']
    ]).reshape(4, 2).astype(np.float32)
    image_in, M = crop(org_im, pts)
    image_in = image_in / 256.0
    image_in = color_normalize(image_in, mean=np.array([0.5, 0.5, 0.5]))
    image_in = image_in.astype(np.float32).transpose([2, 0, 1]).reshape(
        -1, 3, 128, 128)
    return image_in


def reader(face_detector, shrink, confs_threshold, images, paths, use_gpu):
    """
    Preprocess to yield image.

    Args:
        images (list(numpy.ndarray)): images data, shape of each is [H, W, C], color space is BGR.
        paths (list[str]): paths to images.

    Yield:
        each (collections.OrderedDict): info of original image, preprocessed image.
    """
    component = list()
    if paths is not None:
        assert type(paths) is list, "paths should be a list."
        for im_path in paths:
            each = OrderedDict()
            assert os.path.isfile(
                im_path), "The {} isn't a valid file path.".format(im_path)
            im = cv2.imread(im_path)
            each['org_im'] = im
            each['org_im_path'] = im_path
            component.append(each)
    if images is not None:
        assert type(images) is list, "images should be a list."
        for im in images:
            each = OrderedDict()
            each['org_im'] = im
            each['org_im_path'] = 'ndarray_time={}'.format(
                round(time.time(), 6) * 1e6)
            component.append(each)

    for element in component:
        detect_faces = face_detector.face_detection(
            images=[element['org_im']],
            use_gpu=use_gpu,
            visualization=False,
            shrink=shrink,
            confs_threshold=confs_threshold)

        element['preprocessed'] = list()
        for face in detect_faces[0]['data']:
            handled = OrderedDict()
            handled['face'] = face
            handled['image'] = process_image(element['org_im'], face)
            element['preprocessed'].append(handled)

        yield element
