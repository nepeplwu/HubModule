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
    name="ssd_vgg16_512_coco2017",
    version="1.0.0",
    type="cv/object_detection",
    summary="SSD with backbone VGG16, trained with dataset COCO.",
    author="paddle",
    author_email="paddlepaddle@baidu.com")
class HubModule(hub.Module):
    def _initialize(self):
        self.ssd = hub.Module(name="ssd")
        # default pretrained model of SSD, the shape of input image tensor is (3, 512, 512)
        self.default_pretrained_model_path = os.path.join(
            self.directory, "ssd_vgg16_512")
        self.label_names = self.ssd.load_label_info(
            os.path.join(self.directory, "label_file.txt"))
        self.infer_prog = None
        self.image = None
        self.bbox_out = None

    def context(self,
                multi_box_head=None,
                ssd_output_decoder=None,
                input_image=None,
                trainable=True,
                pretrained=False,
                param_prefix='',
                get_prediction=False):
        """Distill the Head Features, so as to perform transfer learning.

        :param multi_box_head: SSD head of MultiBoxHead.
        :type multi_box_head: <class 'MultiBoxHead' object>
        :param ssd_output_decoder: SSD output decoder
        :type ssd_output_decoder: <class 'SSDOutputDecoder' object>
        :param input_image: image tensor.
        :type input_image: <class 'paddle.fluid.framework.Variable'>
        :param trainable: whether to set parameters trainable.
        :type trainable: bool
        :param pretrained: whether to load default pretrained model.
        :type pretrained: bool
        :param param_prefix: the prefix of parameters in multi_box_head and backbone
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
                    name='image', shape=[3, 512, 512], dtype='float32')
                # backbone
                vgg = hub.Module(name='vgg16')
                _, _outputs, _ = vgg.context(
                    input_image=image,
                    normalizations=[20., -1, -1, -1, -1, -1, -1],
                    extra_block_filters=[[256, 512, 1, 2,
                                          3], [128, 256, 1, 2, 3],
                                         [128, 256, 1, 2,
                                          3], [128, 256, 1, 2, 3],
                                         [128, 256, 1, 1, 4]])
                body_feats = _outputs['body_feats']
                # multi_box_head
                if multi_box_head is None:
                    multi_box_head = self.ssd.MultiBoxHead(
                        base_size=512,
                        num_classes=81,
                        aspect_ratios=[[2.], [2., 3.], [2., 3.], [2., 3.],
                                       [2., 3.], [2.], [2.]],
                        min_ratio=15,
                        max_ratio=90,
                        min_sizes=[
                            20.0, 51.0, 133.0, 215.0, 296.0, 378.0, 460.0
                        ],
                        max_sizes=[
                            51.0, 133.0, 215.0, 296.0, 378.0, 460.0, 542.0
                        ],
                        steps=[8, 16, 32, 64, 128, 256, 512],
                        offset=0.5,
                        flip=True,
                        kernel_size=3,
                        pad=1)
                # ssd_output_decoder
                if ssd_output_decoder is None:
                    ssd_output_decoder = self.ssd.SSDOutputDecoder(
                        nms_threshold=0.45,
                        nms_top_k=400,
                        keep_top_k=200,
                        score_threshold=0.01,
                        nms_eta=1.0,
                        background_label=0)

                inputs, outputs, context_prog = self.ssd.context(
                    body_feats=body_feats,
                    multi_box_head=multi_box_head,
                    ssd_output_decoder=ssd_output_decoder,
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
                        if input_image.shape != (-1, 3, 512, 512):
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
            self.bbox_out = outputs['bbox_out']

        place = fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()
        exe = fluid.Executor(place)
        # create data feeder
        feeder = fluid.DataFeeder([self.image], place)
        resize_image = self.ssd.ResizeImage(
            target_size=300, interp=1, max_size=0, use_cv2=False)
        data_reader = partial(
            self.ssd.reader, paths, images, resize_image=resize_image)
        batch_reader = fluid.io.batch(data_reader, batch_size=batch_size)
        # execute program
        output_path = output_dir if output_dir else os.path.join(
            os.getcwd(), 'detection_result')
        paths = paths if paths else list()
        res = []
        for iter_id, feed_data in enumerate(batch_reader()):
            feed_data = np.array(feed_data)
            data_out = exe.run(
                self.infer_prog,
                feed=feeder.feed(feed_data),
                fetch_list=[self.bbox_out],
                return_numpy=False)
            output = self.ssd.postprocess(
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
