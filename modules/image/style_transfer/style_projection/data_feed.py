# coding=utf-8
import cv2
import numpy as np
from PIL import Image

__all__ = ['reader']


def reader(im_path=None, im_arr=None):
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
