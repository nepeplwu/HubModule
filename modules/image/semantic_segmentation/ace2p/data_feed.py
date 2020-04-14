# coding=utf-8
import os
from collections import OrderedDict

import cv2
import numpy as np

__all__ = ['reader']


def _box2cs(box, aspect_ratio):
    x, y, w, h = box[:4]
    return _xywh2cs(x, y, w, h, aspect_ratio)


def _xywh2cs(x, y, w, h, aspect_ratio, pixel_std=200):
    center = np.zeros((2), dtype=np.float32)
    center[0] = x + w * 0.5
    center[1] = y + h * 0.5
    if w > aspect_ratio * h:
        h = w * 1.0 / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio
    scale = np.array([w * 1.0 / pixel_std, h * 1.0 / pixel_std],
                     dtype=np.float32)
    return center, scale


def get_direction(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)
    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs
    return src_result


def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_affine_transform(center,
                         scale,
                         rot,
                         output_size,
                         shift=np.array([0, 0], dtype=np.float32),
                         inv=0):
    if not isinstance(scale, np.ndarray) and not isinstance(
            scale, list) and not isinstance(scale, tuple):
        print(scale)
        scale = np.array([scale, scale])

    scale_tmp = scale * 200.0
    src_w = scale_tmp[0]
    dst_w = output_size[1]
    dst_h = output_size[0]
    rot_rad = np.pi * rot / 180
    src_direction = get_direction([0, src_w * -0.5], rot_rad)
    dst_direction = np.array([0, (dst_w - 1) * -0.5], np.float32)
    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_direction + scale_tmp * shift
    dst[0, :] = [(dst_w - 1) * 0.5, (dst_h - 1) * 0.5]
    dst[1, :] = np.array([(dst_w - 1) * 0.5, (dst_h - 1) * 0.5]) + dst_direction
    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))
    return trans


def preprocess(org_im, scale, rotation):
    image = org_im.copy()
    image_height, image_width, _ = image.shape

    aspect_ratio = scale[1] * 1.0 / scale[0]
    image_center, image_scale = _box2cs(
        [0, 0, image_width - 1, image_height - 1], aspect_ratio)

    trans = get_affine_transform(image_center, image_scale, rotation, scale)
    image = cv2.warpAffine(
        image,
        trans, (int(scale[1]), int(scale[0])),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0))

    img_mean = np.array([0.406, 0.456, 0.485]).reshape((1, 1, 3))
    img_std = np.array([0.225, 0.224, 0.229]).reshape((1, 1, 3))
    image = image.astype(np.float)
    image = (image / 255.0 - img_mean) / img_std
    image = image.transpose(2, 0, 1).astype(np.float32)

    image_info = {
        'image_center': image_center,
        'image_height': image_height,
        'image_width': image_width,
        'image_scale': image_scale,
        'rotation': rotation,
        'scale': scale
    }

    return image, image_info


def reader(images=None, paths=None, scale=(473, 473), rotation=0):
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
            im = cv2.imread(im_path)
            each['org_im'] = im
            each['org_im_path'] = im_path
            component.append(each)
    if images is not None:
        assert type(images) is list, "images should be a list."
        for im in images:
            each = OrderedDict()
            each['org_im'] = im
            each['org_im_path'] = None
            component.append(each)

    for element in component:
        element['image'], element['image_info'] = preprocess(
            element['org_im'], scale, rotation)
        yield element
