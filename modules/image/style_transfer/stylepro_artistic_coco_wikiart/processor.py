# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
from PIL import Image

__all__ = ['postprocess', 'fr']


def postprocess(im, output_dir, save_im_name, visualization):
    im = np.multiply(im, 255.0) + 0.5
    im = np.clip(im, 0, 255)
    im = im.astype(np.uint8)
    im = im.transpose((1, 2, 0))
    if visualization:
        # create output directory
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        elif os.path.isfile(output_dir):
            os.remove(output_dir)
            os.makedirs(output_dir)
        # save image
        img = Image.fromarray(im)
        img.save(save_im_name)
        print('image saved in {}'.format(save_im_name))
    return im


def fr(content_feat, style_feat, alpha):
    content_feat = np.reshape(content_feat, (512, -1))
    style_feat = np.reshape(style_feat, (512, -1))

    content_feat_index = np.argsort(content_feat, axis=1)
    style_feat = np.sort(style_feat, axis=1)

    fr_feat = scatter_numpy(dim=1, index=content_feat_index, src=style_feat)
    fr_feat = fr_feat * alpha + content_feat * (1 - alpha)
    fr_feat = np.reshape(fr_feat, (1, 512, 64, 64))
    return fr_feat


def scatter_numpy(dim, index, src):
    """
    Writes all values from the Tensor src into dst at the indices specified in the index Tensor.

    :param dim: The axis along which to index
    :param index: The indices of elements to scatter
    :param src: The source element(s) to scatter
    :return: dst
    """
    dst = src.copy()
    idx_xsection_shape = index.shape[:dim] + index.shape[dim + 1:]
    dst_xsection_shape = dst.shape[:dim] + dst.shape[dim + 1:]
    if idx_xsection_shape != dst_xsection_shape:
        raise ValueError(
            "Except for dimension " + str(dim) +
            ", all dimensions of index and output should be the same size")
    if (index >= dst.shape[dim]).any() or (index < 0).any():
        raise IndexError("The values of index must be between 0 and {}.".format(
            dst.shape[dim] - 1))

    def make_slice(arr, dim, i):
        slc = [slice(None)] * arr.ndim
        slc[dim] = i
        return tuple(slc)

    # We use index and dim parameters to create idx
    # idx is in a form that can be used as a NumPy advanced index for scattering of src param.
    idx = [[
        *np.indices(idx_xsection_shape).reshape(index.ndim - 1, -1),
        index[make_slice(index, dim, i)].reshape(1, -1)[0]
    ] for i in range(index.shape[dim])]
    idx = list(np.concatenate(idx, axis=1))
    idx.insert(dim, idx.pop())

    if not np.isscalar(src):
        if index.shape[dim] > src.shape[dim]:
            raise IndexError("Dimension " + str(dim) +
                             "of index can not be bigger than that of src ")
        src_xsection_shape = src.shape[:dim] + src.shape[dim + 1:]
        if idx_xsection_shape != src_xsection_shape:
            raise ValueError(
                "Except for dimension " + str(dim) +
                ", all dimensions of index and src should be the same size")
        # src_idx is a NumPy advanced index for indexing of elements in the src
        src_idx = list(idx)
        src_idx.pop(dim)
        src_idx.insert(
            dim,
            np.repeat(np.arange(index.shape[dim]), np.prod(idx_xsection_shape)))
        dst[tuple(idx)] = src[tuple(src_idx)]
    else:
        dst[idx] = src
    return dst
