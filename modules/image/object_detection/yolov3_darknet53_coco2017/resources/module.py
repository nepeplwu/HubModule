# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import paddle.fluid as fluid
import paddlehub as hub

from functools import partial
from paddlehub.module.module import moduleinfo


@moduleinfo(
    name="yolov3_darknet53_coco2017",
    version="2.0.0",
    type="cv/object_detection",
    summary="YOLOV3.",
    author="paddle",
    author_email="paddlepaddle@baidu.com")
class HubModule(hub.Module):
    def _initialize(self):
        self.yolov3 = hub.Module(name="yolov3")
        # default pretrained model of YOLOv3_DarkNet53, the shape of input image tensor is (3, 608, 608)
        self.default_pretrained_model_path = os.path.join(
            self.directory, "yolov3_darknet")
        self.label_names = self.yolov3.load_label_info(
            os.path.join(self.directory, "label_file.txt"))
        self.infer_prog = None
        self.image = None
        self.im_size = None
        self.bbox_out = None

    def context(self,
                yolo_head=None,
                input_image=None,
                trainable=True,
                pretrained=False,
                param_prefix='',
                get_prediction=False):
        """Distill the Head Features, so as to perform transfer learning.

        :param yolo_head: Head of YOLOv3.
        :type yolo_head: <class 'YOLOv3Head' object>
        :param input_image: image tensor.
        :type input_image: <class 'paddle.fluid.framework.Variable'>
        :param trainable: whether to set parameters trainable.
        :type trainable: bool
        :param pretrained: whether to load default pretrained model.
        :type pretrained: bool
        :param param_prefix: the prefix of parameters in yolo_head and backbone
        :type param_prefix: str
        :param get_prediction: whether to get prediction,
            if True, outputs is {'bbox_out': bbox_out},
            if False, outputs is {'head_features': head_features}.
        :type get_prediction: bool
        """
        wrapped_prog = input_image.block.program if input_image else fluid.Program(
        )
        with fluid.program_guard(wrapped_prog):
            with fluid.unique_name.guard():
                # image
                image = input_image if input_image else fluid.layers.data(
                    name='image', shape=[3, 608, 608], dtype='float32')
                # yolo_head
                if yolo_head is None:
                    yolo_head = self.yolov3.YOLOv3Head()
                darknet = hub.Module(name='darknet')
                _, _outputs, _ = darknet.context(input_image=image)
                body_feats = _outputs['body_feats']
                inputs, outputs, context_prog = self.yolov3.context(
                    body_feats=body_feats,
                    yolo_head=yolo_head,
                    image=image,
                    trainable=trainable,
                    param_prefix=param_prefix,
                    get_prediction=get_prediction)

            place = fluid.CPUPlace()
            exe = fluid.Executor(place)
            with fluid.program_guard(context_prog):
                if pretrained:

                    def _if_exist(var):
                        return os.path.exists(
                            os.path.join(self.default_pretrained_model_path,
                                         var.name))

                    load_default_pretrained_model = True
                    if param_prefix:
                        load_default_pretrained_model = False
                    elif input_image:
                        if input_image.shape != (-1, 3, 608, 608):
                            load_default_pretrained_model = False
                    if load_default_pretrained_model:
                        fluid.io.load_vars(
                            exe,
                            self.default_pretrained_model_path,
                            predicate=_if_exist)
                return inputs, outputs, context_prog

    def object_detection(self,
                         paths=None,
                         images=None,
                         use_gpu=False,
                         batch_size=1,
                         output_dir=None,
                         score_thresh=0.5):
        """API of Object Detection.

        :param paths: the path of images.
        :type paths: list, each element is correspond to the path of an image.
        :param images: data of images, [N, H, W, C]
        :type images: numpy.ndarray
        :param use_gpu: whether to use gpu or not.
        :type use_gpu: bool
        :param batch_size: bathc size.
        :type batch_size: int
        :param output_dir: the directory to store the detection result.
        :type output_dir: str
        :param score_thresh: the threshold of detection confidence.
        :type score_thresh: float
        """
        if self.infer_prog is None:
            inputs, outputs, self.infer_prog = self.context(
                trainable=False, pretrained=True, get_prediction=True)

            self.infer_prog = self.infer_prog.clone(for_test=True)
            self.image = inputs['image']
            self.im_size = inputs['im_size']
            self.bbox_out = outputs['bbox_out']

        place = fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()
        exe = fluid.Executor(place)

        paths = paths if paths else list()
        output_path = output_dir if output_dir else os.path.join(
            os.getcwd(), 'detection_result')
        feeder = fluid.DataFeeder([self.image, self.im_size], place)
        data_reader = partial(self.yolov3.reader, paths, images)
        batch_reader = fluid.io.batch(data_reader, batch_size=batch_size)

        res = []
        for iter_id, feed_data in enumerate(batch_reader()):
            feed_data = np.array(feed_data)
            data_out = exe.run(
                self.infer_prog,
                feed=feeder.feed(feed_data),
                fetch_list=[self.bbox_out],
                return_numpy=False)
            output = self.yolov3.postprocess(
                paths=paths,
                images=images,
                data_out=data_out,
                score_thresh=score_thresh,
                label_names=self.label_names,
                output_dir=output_path,
                handle_id=iter_id * batch_size,
                draw_bbox=True)
            res.append(output)
        return res
