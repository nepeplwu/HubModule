# coding=utf-8
import os
import time
from collections import OrderedDict

import cv2
import numpy as np
from PIL import Image

__all__ = ['reader']


def reader(face_detector, images=None, paths=None, use_gpu=False):
    """
    Preprocess to yield image.

    Args:
        images (list(numpy.ndarray)): images data, shape of each is [H, W, C].
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
            each['org_im_shape'] = im.shape
            # get im_path
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
        assert type(images) is list, "images should be a list."
        for im in images:
            each = OrderedDict()
            each['org_im'] = im
            each['org_im_path'] = 'ndarray_time={}.jpg'.format(
                round(time.time(), 6) * 1e6)
            each['org_im_shape'] = im.shape
            component.append(each)

    for element in component:
        im = element['org_im'].astype('float32').copy()
        face_detection_output = face_detector.face_detection(
            images=np.expand_dims(im, axis=0), use_gpu=use_gpu)

        element['image'] = list()
        for index, det in enumerate(face_detection_output[0]['data']):
            x1 = int(det['left'])
            x2 = int(det['right'])
            y1 = int(det['top'])
            y2 = int(det['bottom'])
            if x1 < 0:
                x1 = 0
            if y1 < 0:
                y1 = 0
            if x2 > element['org_im_shape'][1]:
                x2 = element['org_im_shape'][1]
            if y2 > element['org_im_shape'][0]:
                y2 = element['org_im_shape'][0]

            roi = im[y1:y2 + 1, x1:x2 + 1, ]
            gray_img = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
            gray_img = cv2.resize(
                gray_img, (60, 60), interpolation=cv2.INTER_CUBIC)
            mean, std_dev = cv2.meanStdDev(gray_img)
            gray_img = (gray_img - mean[0][0]) / (0.000001 + std_dev[0][0])
            gray_img = np.expand_dims(gray_img, axis=0)
            element['image'].append({
                'face': gray_img,
                'x1': x1,
                'y1': y1,
                'x2': x2,
                'y2': y2
            })
        yield element
