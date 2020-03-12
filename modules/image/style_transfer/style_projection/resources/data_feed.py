# coding=utf-8
import numpy as np
from PIL import Image

__all__ = ['reader']


def reader(path):
    im = Image.open(path)
    im = im.resize((512, 512), resample=Image.BILINEAR)
    im = np.array(im).astype(np.float32)
    im = im.transpose((2, 0, 1))
    im = np.expand_dims(im, axis=0)
    im /= 255.0
    return im
