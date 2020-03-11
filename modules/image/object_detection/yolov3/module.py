# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import OrderedDict

import paddle.fluid as fluid
import paddlehub as hub
from paddlehub.module.module import moduleinfo

from yolov3.data_feed import reader
from yolov3.processor import load_label_info, postprocess
from yolov3.yolo_head import YOLOv3Head



@moduleinfo(
    name="yolov3",
    version="1.0.0",
    type="cv/object_detection",
    summary="Baidu's YOLOv3 model for object detection.",
    author="paddlepaddle",
    author_email="paddle-dev@baidu.com")
class HubModule(hub.Module):
    def _initialize(self):
        self.reader = reader
        self.load_label_info = load_label_info
        self.postprocess = postprocess
        self.YOLOv3Head = YOLOv3Head

    def context(self,
                body_feats,
                yolo_head,
                image,
                trainable=True,
                param_prefix='',
                get_prediction=False):
        """Distill the Head Features, so as to perform transfer learning.

        :param body_feats: feature maps of backbone
        :type backbone: list
        :param yolo_head: yolo_head of YOLOv3
        :type yolo_head: <class 'YOLOv3Head' object>
        :param image: image tensor.
        :type image: <class 'paddle.fluid.framework.Variable'>
        :param trainable: whether to set parameters trainable.
        :type trainable: bool
        :param param_prefix: the prefix of parameters in yolo_head and backbone
        :type param_prefix: str
        :param get_prediction: whether to get prediction,
            if True, outputs is bbox_out,
            if False, outputs is head_features.
        :type get_prediction: bool
        """
        context_prog = image.block.program
        with fluid.program_guard(context_prog):
            im_size = fluid.layers.data(
                name='im_size', shape=[2], dtype='int32')
            head_features = yolo_head._get_outputs(
                body_feats, is_train=trainable)
            inputs = {'image': image, 'im_size': im_size}
            if get_prediction:
                bbox_out = yolo_head.get_prediction(head_features, im_size)
                outputs = {'bbox_out': bbox_out}
            else:
                outputs = {'head_features': head_features}

            place = fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(fluid.default_startup_program())

            if param_prefix:
                yolo_head.prefix_name = param_prefix

            for param in context_prog.global_block().iter_parameters():
                param.trainable = trainable
            return inputs, outputs, context_prog
