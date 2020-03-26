# coding=utf-8
import os
from collections import OrderedDict

import cv2
import numpy as np

__all__ = ['reader']


def preprocess(org_image):
    image = cv2.cvtColor(org_image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (320, 240))
    image_mean = np.array([127, 127, 127])
    image = (image - image_mean) / 128.0
    image = np.transpose(image, [2, 0, 1])
    return image


def reader(images=None, paths=None):
    """
    Preprocess to yield image.

    Args:
        images (list(numpy.ndarray)): images data, shape of each is [H, W, C]
        paths (list[str]): paths to images.

    Yield:
        each (collections.OrderedDict): info of original image, preprocessed image.
    """
    component = list()
    if paths:
        for im_path in paths:
            each = OrderedDict()
            assert os.path.isfile(
                im_path), "The {} isn't a valid file path.".format(im_path)
            im = cv2.imread(im_path).astype('float32')
            each['org_im'] = im
            each['org_im_shape'] = im.shape  # height, width, channel
            each['org_im_path'] = im_path
            component.append(each)
    if images is not None:
        assert type(images) is list, "images should be a list."
        for im in images:
            each = OrderedDict()
            each['org_im'] = im
            each['org_im_path'] = None
            each['org_im_shape'] = im.shape  # height, width, channel
            component.append(each)

    for element in component:
        element['image'] = preprocess(element['org_im'])
        yield element
