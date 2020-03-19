# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import OrderedDict

import paddle.fluid as fluid
import paddlehub as hub
from paddlehub.module.module import moduleinfo
from paddlehub.common.paddle_helper import add_vars_prefix

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
class YOLOv3(hub.Module):
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
                var_prefix=''):
        """Distill the Head Features, so as to perform transfer learning.

        :param body_feats: feature maps of backbone
        :type backbone: list
        :param yolo_head: yolo_head of YOLOv3
        :type yolo_head: <class 'YOLOv3Head' object>
        :param image: image tensor.
        :type image: <class 'paddle.fluid.framework.Variable'>
        :param trainable: whether to set parameters trainable.
        :type trainable: bool
        :param var_prefix: the prefix of variables in yolo_head and backbone
        :type var_prefix: str
        """
        context_prog = image.block.program
        with fluid.program_guard(context_prog, fluid.Program()):
            im_size = fluid.layers.data(
                name='im_size', shape=[2], dtype='int32')
            head_features = yolo_head._get_outputs(
                body_feats, is_train=trainable)
            inputs = {
                'image': var_prefix + image.name,
                'im_size': var_prefix + im_size.name
            }
            #             bbox_out = yolo_head.get_prediction(head_features, im_size)
            outputs = {
                'head_features':
                [var_prefix + var.name for var in head_features]
            }

            place = fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(fluid.default_startup_program())

            add_vars_prefix(context_prog, var_prefix)
            inputs = {
                key: context_prog.global_block().vars[value]
                for key, value in inputs.items()
            }
            outputs = {
                key: [
                    context_prog.global_block().vars[varname]
                    for varname in value
                ]
                for key, value in outputs.items()
            }

            for param in context_prog.global_block().iter_parameters():
                param.trainable = trainable
            return inputs, outputs, context_prog
