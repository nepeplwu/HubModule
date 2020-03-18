# coding=utf-8
import os
import time

import cv2
import numpy as np
from PIL import Image
from collections import OrderedDict

__all__ = ['reader']


def reader(images=None, paths=None):
    """
    Preprocess to get image data.

    Args:
        images (list): list of [content_arr, styles_arr_list, style_interpolation_weights],
            the first element is a numpy.ndarry with shape [H, W, C], content data.
            the second element is a list of numpy.ndarray with shape [H, W, C], styles data.
            the last element is a list (Optional), the interpolation weights correspond to styles.
        paths (list): list of [content_path, styles_path_list, style_interpolation_weights],
            the first element is a str, the path to content,
            the second element is a list, the path to styles,
            the last element is a list (Optional), the interpolation weights correspond to styles.
    Yield:
        im (numpy.ndarray): preprocessed data, with shape (1, 3, 512, 512).
    """
    pipeline_list = list()
    # images
    if images is not None:
        for ndarray_component in images:
            each_res = OrderedDict()
            if len(ndarray_component) > 1:
                # content_arr
                each_res['content_arr'] = _handle_single(
                    im_arr=ndarray_component[0])
                # styles_arr_list
                styles_arr_list = ndarray_component[1]
                styles_num = len(styles_arr_list)
                each_res['styles_arr_list'] = list()
                for i, style_arr in enumerate(styles_arr_list):
                    each_res['styles_arr_list'].append(
                        _handle_single(im_arr=style_arr))
                # style_interpolation_weights
                if len(ndarray_component) == 3:
                    assert len(
                        ndarray_component[2]
                    ) == styles_num, "The number of weights must be equal to the number of styles."
                    each_res['style_interpolation_weights'] = ndarray_component[
                        2]
                else:
                    each_res['style_interpolation_weights'] = np.ones(
                        styles_num)
                each_res['style_interpolation_weights'] = [
                    each_res['style_interpolation_weights'][j] / sum(
                        each_res['style_interpolation_weights'])
                    for j in range(styles_num)
                ]
                # save_im_name
                each_res[
                    'save_im_name'] = 'ndarray_interpolation' + '_time={}_'.format(
                        round(time.time(), 2))
                pipeline_list.append(each_res)
            else:
                raise ValueError(
                    'Each element is a list, whose length must >= 2.')
    # paths
    if paths:
        for path_component in paths:
            each_res = OrderedDict()
            if len(path_component) > 1:
                each_res['save_im_name'] = os.path.splitext(
                    os.path.basename(path_component[0]))[0]
                each_res['content_arr'] = _handle_single(
                    im_path=path_component[0])
                styles_path_list = path_component[1]
                styles_num = len(styles_path_list)
                if len(path_component) == 3:
                    assert len(
                        path_component[2]
                    ) == styles_num, "The number of weights must be equal to the number of styles."
                    each_res['style_interpolation_weights'] = path_component[2]
                else:
                    each_res['style_interpolation_weights'] = np.ones(
                        styles_num)
                each_res['style_interpolation_weights'] = [
                    each_res['style_interpolation_weights'][j] / sum(
                        each_res['style_interpolation_weights'])
                    for j in range(styles_num)
                ]

                each_res['styles_arr_list'] = list()
                for i, style_path in enumerate(styles_path_list):
                    each_res['styles_arr_list'].append(
                        _handle_single(im_path=style_path))
                    each_res['save_im_name'] += os.path.splitext(
                        os.path.basename(
                            style_path))[0] + '_w=%.2f_&' % each_res[
                                'style_interpolation_weights'][i]
                pipeline_list.append(each_res)
            else:
                raise ValueError(
                    'Each element is a list, whose length must >= 2.')
    # yield
    for element in pipeline_list:
        yield element


def _handle_single(im_path=None, im_arr=None):
    """
    Preprocess to get image data.
    Args:
        im_path (str): path to image.
        im_arr (numpy.ndarray): image data, with shape (H, W, 3).
    Returns:
        im (numpy.ndarray): preprocessed data, with shape (1, 3, 512, 512).
    """
    if im_path is not None:
        im = Image.open(im_path)
        im = im.resize((512, 512), resample=Image.BILINEAR)
        im = np.array(im).astype(np.float32)
    if im_arr is not None:
        im = im_arr.astype('float32')
        im = cv2.resize(im, (512, 512), interpolation=cv2.INTER_LINEAR)
    im = im.transpose((2, 0, 1))
    im = np.expand_dims(im, axis=0)
    im /= 255.0
    return im
