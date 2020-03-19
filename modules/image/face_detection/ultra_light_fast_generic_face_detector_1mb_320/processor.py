# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import cv2
import numpy as np
from collections import OrderedDict

__all__ = ['postprocess']


def area_of(left_top, right_bottom):
    hw = np.clip(right_bottom - left_top, 0.0, None)
    return hw[..., 0] * hw[..., 1]


def iou_of(boxes0, boxes1, eps=1e-5):
    overlap_left_top = np.maximum(boxes0[..., :2], boxes1[..., :2])
    overlap_right_bottom = np.minimum(boxes0[..., 2:], boxes1[..., 2:])
    overlap_area = area_of(overlap_left_top, overlap_right_bottom)
    area0 = area_of(boxes0[..., :2], boxes0[..., 2:])
    area1 = area_of(boxes1[..., :2], boxes1[..., 2:])
    return overlap_area / (area0 + area1 - overlap_area + eps)


def hard_nms(box_scores, iou_threshold, top_k=-1, candidate_size=200):
    scores = box_scores[:, -1]
    boxes = box_scores[:, :-1]
    picked = []
    # _, indexes = scores.sort(descending=True)
    indexes = np.argsort(scores)
    # indexes = indexes[:candidate_size]
    indexes = indexes[-candidate_size:]
    while len(indexes) > 0:
        # current = indexes[0]
        current = indexes[-1]
        picked.append(current)
        if 0 < top_k == len(picked) or len(indexes) == 1:
            break
        current_box = boxes[current, :]
        # indexes = indexes[1:]
        indexes = indexes[:-1]
        rest_boxes = boxes[indexes, :]
        iou = iou_of(rest_boxes, np.expand_dims(current_box, axis=0))
        indexes = indexes[iou <= iou_threshold]
    return box_scores[picked, :]


def check_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    elif os.path.isfile(dir_path):
        os.remove(dir_path)
        os.makedirs(dir_path)


def postprocess(confidences,
                boxes,
                org_im,
                org_im_shape,
                org_im_path,
                output_dir,
                visualization,
                confs_threshold=0.5,
                iou_threshold=0.5):
    """
    Postprocess output of network. one image at a time.

    Args:
        confidences (numpy.ndarray): confidences, with shape [num, 2]
        boxes (numpy.ndaray): boxes coordinate,  with shape [num, 4]
        org_im (numpy.ndarray): original image.
        org_im_shape (list): shape pf original image.
        org_im_path (list): path of riginal image.
        output_dir (str): output directory to store image.
        visualization (bool): whether to save image or not.
    """
    output = OrderedDict()
    output['data'] = list()
    output['path'] = org_im_path
    picked_box_probs = list()
    picked_labels = list()
    for class_index in range(1, confidences.shape[1]):
        probs = confidences[:, class_index]
        mask = probs > confs_threshold
        probs = probs[mask]
        if probs.shape[0] == 0:
            continue
        subset_boxes = boxes[mask, :]
        box_probs = np.concatenate([subset_boxes, probs.reshape(-1, 1)], axis=1)
        box_probs = hard_nms(box_probs, iou_threshold=iou_threshold, top_k=-1)
        picked_box_probs.append(box_probs)
        picked_labels.extend([class_index] * box_probs.shape[0])

    if not picked_box_probs:
        return output

    picked_box_probs = np.concatenate(picked_box_probs)
    picked_box_probs[:, 0] *= org_im_shape[1]
    picked_box_probs[:, 1] *= org_im_shape[0]
    picked_box_probs[:, 2] *= org_im_shape[1]
    picked_box_probs[:, 3] *= org_im_shape[0]

    for data in picked_box_probs:
        output['data'].append({
            'left': data[0],
            'right': data[2],
            'top': data[1],
            'bottom': data[3],
            'confidence': data[4]
        })

    picked_box_probs = picked_box_probs[:, :4].astype(np.int32)
    if visualization:
        for i in range(picked_box_probs.shape[0]):
            box = picked_box_probs[i]
            cv2.rectangle(org_im, (box[0], box[1]), (box[2], box[3]),
                          (255, 255, 0), 2)
        check_dir(output_dir)
        im_save_path = os.path.join(output_dir, org_im_path)
        cv2.imwrite(im_save_path, org_im)
        print("The image with bbox is saved as {}".format(im_save_path))
    return output
