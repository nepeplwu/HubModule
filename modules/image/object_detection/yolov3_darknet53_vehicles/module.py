# coding=utf-8
from __future__ import absolute_import
from __future__ import division

import ast
import argparse
import os
from functools import partial

import numpy as np
import paddle.fluid as fluid
import paddlehub as hub
from paddle.fluid.core import PaddleTensor, AnalysisConfig, create_paddle_predictor
from paddlehub.module.module import moduleinfo, runnable, serving

from yolov3_darknet53_vehicles.darknet import DarkNet


@moduleinfo(
    name="yolov3_darknet53_vehicles",
    version="1.0.0",
    type="CV/object_detection",
    summary=
    "Baidu's YOLOv3 model for vehicles detection, with backbone DarkNet53.",
    author="paddlepaddle",
    author_email="paddle-dev@baidu.com")
class YOLOv3DarkNet53Vehicles(hub.Module):
    def _initialize(self):
        self.yolov3 = hub.Module(name="yolov3")
        self.default_pretrained_model_path = os.path.join(
            self.directory, "yolov3_darknet53_vehicles_model")
        self.label_names = self.yolov3.load_label_info(
            os.path.join(self.directory, "label_file.txt"))
        self._set_config()

    def _set_config(self):
        """
        predictor config setting.
        """
        cpu_config = AnalysisConfig(self.default_pretrained_model_path)
        cpu_config.disable_glog_info()
        cpu_config.disable_gpu()
        cpu_config.switch_ir_optim(False)
        self.cpu_predictor = create_paddle_predictor(cpu_config)

        try:
            _places = os.environ["CUDA_VISIBLE_DEVICES"]
            int(_places[0])
            use_gpu = True
        except:
            use_gpu = False
        if use_gpu:
            gpu_config = AnalysisConfig(self.default_pretrained_model_path)
            gpu_config.disable_glog_info()
            gpu_config.enable_use_gpu(memory_pool_init_size_mb=500, device_id=0)
            self.gpu_predictor = create_paddle_predictor(gpu_config)

    def context(self,
                trainable=True,
                pretrained=True,
                var_prefix='',
                get_prediction=False):
        """
        Distill the Head Features, so as to perform transfer learning.

        Args:
            trainable (bool): whether to set parameters trainable.
            pretrained (bool): whether to load default pretrained model.
            var_prefix (str): prefix to append to the varibles.
            get_prediction (bool): whether to get prediction.

        Returns:
             inputs(dict): the input variables.
             outputs(dict): the output variables.
             context_prog (Program): the program to execute transfer learning.
        """
        wrapped_prog = fluid.Program()
        startup_program = fluid.Program()
        with fluid.program_guard(wrapped_prog, startup_program):
            with fluid.unique_name.guard():
                # image
                image = fluid.layers.data(
                    name='image', shape=[3, 608, 608], dtype='float32')
                # yolo_head
                yolo_head = self.yolov3.YOLOv3Head(
                    anchor_masks=[[6, 7, 8], [3, 4, 5], [0, 1, 2]],
                    anchors=[[8, 9], [10, 23], [19, 15], [23, 33], [40, 25],
                             [54, 50], [101, 80], [139, 145], [253, 224]],
                    norm_decay=0.,
                    num_classes=6,
                    ignore_thresh=0.7,
                    label_smooth=False,
                    nms=self.yolov3.MultiClassNMS(
                        background_label=-1,
                        keep_top_k=100,
                        nms_threshold=0.45,
                        nms_top_k=400,
                        normalized=False,
                        score_threshold=0.005))
                backbone = DarkNet(norm_type='sync_bn', norm_decay=0., depth=53)
                body_feats = backbone(image)
                var_prefix = var_prefix if var_prefix else '@HUB_{}@'.format(
                    self.name)
                inputs, outputs, context_prog = self.yolov3.context(
                    body_feats=body_feats,
                    yolo_head=yolo_head,
                    image=image,
                    trainable=trainable,
                    var_prefix=var_prefix,
                    get_prediction=get_prediction)

                place = fluid.CPUPlace()
                exe = fluid.Executor(place)
                if pretrained:

                    def _if_exist(var):
                        return os.path.exists(
                            os.path.join(self.default_pretrained_model_path,
                                         var.name))

                    fluid.io.load_vars(
                        exe,
                        self.default_pretrained_model_path,
                        predicate=_if_exist)
                else:
                    exe.run(startup_program)
                return inputs, outputs, context_prog

    def object_detection(self,
                         paths=None,
                         images=None,
                         data=None,
                         batch_size=1,
                         use_gpu=False,
                         output_dir='yolov3_vehicles_detect_output',
                         score_thresh=0.2,
                         visualization=True):
        """API of Object Detection.

        Args:
            paths (list[str]): The paths of images.
            images (list(numpy.ndarray)): images data, shape of each is [H, W, C]
            batch_size (int): batch size.
            use_gpu (bool): Whether to use gpu.
            output_dir (str): The path to store output images.
            visualization (bool): Whether to save image or not.
            score_thresh (float): threshold for object detecion.

        Returns:
            res (list[dict]): The result of vehicles detecion. keys include 'data', 'save_path', the corresponding value is:
                data (dict): the result of object detection, keys include 'left', 'top', 'right', 'bottom', 'label', 'confidence', the corresponding value is:
                    left (float): The X coordinate of the upper left corner of the bounding box;
                    top (float): The Y coordinate of the upper left corner of the bounding box;
                    right (float): The X coordinate of the lower right corner of the bounding box;
                    bottom (float): The Y coordinate of the lower right corner of the bounding box;
                    label (str): The label of detection result;
                    confidence (float): The confidence of detection result.
                save_path (str): The path to save output images.
        """
        if data and 'image' in data:
            paths = data['image'] if not paths else paths + data['image']
        paths = paths if paths else list()
        data_reader = partial(self.yolov3.reader, paths, images)
        batch_reader = fluid.io.batch(data_reader, batch_size=batch_size)
        res = []
        for iter_id, feed_data in enumerate(batch_reader()):
            feed_data = np.array(feed_data)
            image_tensor = PaddleTensor(np.array(list(feed_data[:, 0])))
            im_size_tensor = PaddleTensor(np.array(list(feed_data[:, 1])))
            if use_gpu:
                data_out = self.gpu_predictor.run(
                    [image_tensor, im_size_tensor])
            else:
                data_out = self.cpu_predictor.run(
                    [image_tensor, im_size_tensor])

            output = self.yolov3.postprocess(
                paths=paths,
                images=images,
                data_out=data_out,
                score_thresh=score_thresh,
                label_names=self.label_names,
                output_dir=output_dir,
                handle_id=iter_id * batch_size,
                visualization=visualization)
            res.extend(output)
        return res
