# coding=utf-8
import os
import time

import cv2
import numpy as np
from PIL import Image
from collections import OrderedDict

__all__ = ['reader']


def preprocess(orig_image):
    image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (640, 480))
    image_mean = np.array([127, 127, 127])
    image = (image - image_mean) / 128
    image = np.transpose(image, [2, 0, 1])
    image = image.astype(np.float32)
    return image


def reader(images=None, paths=None):
    """
    Preprocess to yield image.

    Args:
        images (numpy.ndarray): images data, with shape [N, H, W, C]
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
            org_im_name, _ = os.path.splitext(im_path.split('/')[-1])
            PIL_img = Image.open(im_path)
            if PIL_img.format == 'PNG':
                im_ext = '.png'
            elif PIL_img.format == 'JPEG':
                im_ext = '.jpg'
            elif PIL_img.format == 'BMP':
                im_ext = '.bmp'
            each['org_im_path'] = org_im_name + im_ext
            component.append(each)
    if images is not None:
        assert len(images.shape) == 4, "The dimension of images must be 4."
        for im in images:
            each = OrderedDict()
            each['org_im'] = im
            each['org_im_path'] = 'ndarray_time={}.jpg'.format(
                int(round(time.time(), 6) * 1e6))
            each['org_im_shape'] = im.shape  # height, width, channel
            component.append(each)

    for element in component:
        element['image'] = preprocess(element['org_im'])
        yield element
