# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np

__all__ = ['postprocess']


def softmax(x):
    orig_shape = x.shape
    if len(x.shape) > 1:
        tmp = np.max(x, axis=1)
        x -= tmp.reshape((x.shape[0], 1))
        x = np.exp(x)
        tmp = np.sum(x, axis=1)
        x /= tmp.reshape((x.shape[0], 1))
    else:
        tmp = np.max(x)
        x -= tmp
        x = np.exp(x)
        tmp = np.sum(x)
        x /= tmp
    return x


def postprocess(data_out, label_list, top=1):
    """
    Postprocess output of network, one image at a time.

    Args:
        data_out (numpy.ndarray): output data of network.
        label_list (list): list of label.
        top (int): top of results.
    """
    output = list()
    for result in data_out:
        result_i = softmax(result)
        indexs = np.argsort(result_i)[::-1][0:top]
        for index in indexs:
            label = label_list[index]
            output.append({label: float(result_i[index])})
    return output
