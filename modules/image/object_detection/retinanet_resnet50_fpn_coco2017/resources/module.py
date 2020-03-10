# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import paddle.fluid as fluid
import paddlehub as hub

from functools import partial
from .fpn import FPN
from .retina_head import AnchorGenerator, RetinaTargetAssign, RetinaOutputDecoder, RetinaHead
from .processor import load_label_info, postprocess
from .data_feed import test_reader, padding_minibatch

from paddlehub.module.module import moduleinfo


@moduleinfo(
    name="retinanet_resnet50_fpn_coco2017",
    version="1.1.0",
    type="cv/object_detection",
    summary=
    "Baidu's RetinaNet model for object detection, with backbone ResNet50 and FPN.",
    author="paddlepaddle",
    author_email="paddle-dev@baidu.com")
class HubModule(hub.Module):
    def _initialize(self):
        # default pretrained model of Retinanet_ResNet50_FPN, the shape of input image tensor is (3, 608, 608)
        self.default_pretrained_model_path = os.path.join(
            self.directory, "retinanet_r50_fpn_1x")
        self.label_names = load_label_info(
            os.path.join(self.directory, "label_file.txt"))
        self.infer_prog = None
        self.image = None
        self.im_info = None
        self.bbox_out = None

    def context(self,
                input_image=None,
                trainable=True,
                pretrained=False,
                param_prefix='',
                get_prediction=False):
        """Distill the Head Features, so as to perform transfer learning.

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
        context_prog = input_image.block.program if input_image else fluid.Program()
        startup_program = fluid.Program()
        with fluid.program_guard(context_prog, startup_program):
            # image
            image = input_image if input_image else fluid.layers.data(
                name='image',
                shape=[3, 800, 1333],
                dtype='float32',
                lod_level=0)
            # im_info
            im_info = fluid.layers.data(
                name='im_info', shape=[3], dtype='float32', lod_level=0)
            # backbone
            resnet = hub.Module(name='resnet_imagenet')
            _, _outputs, _ = resnet.context(
                input_image=image,
                depth=50,
                variant='b',
                norm_type='affine_channel',
                feature_maps=[3, 4, 5])
            body_feats = _outputs['body_feats']

            # retina_head
            retina_head = RetinaHead(
                anchor_generator=AnchorGenerator(
                    aspect_ratios=[1.0, 2.0, 0.5],
                    variance=[1.0, 1.0, 1.0, 1.0]),
                target_assign=RetinaTargetAssign(
                    positive_overlap=0.5, negative_overlap=0.4),
                output_decoder=RetinaOutputDecoder(
                    score_thresh=0.05,
                    nms_thresh=0.5,
                    pre_nms_top_n=1000,
                    detections_per_im=100,
                    nms_eta=1.0),
                num_convs_per_octave=4,
                num_chan=256,
                max_level=7,
                min_level=3,
                prior_prob=0.01,
                base_scale=4,
                num_scales_per_octave=3)
            # fpn
            fpn = FPN(
                max_level=7,
                min_level=3,
                num_chan=256,
                spatial_scale=[0.03125, 0.0625, 0.125],
                has_extra_convs=True)
            # body_feats
            body_feats, spatial_scale = fpn.get_output(body_feats)
            # inputs, outputs, context_prog
            inputs = {'image': image, 'im_info': im_info}
            if get_prediction:
                pred = retina_head.get_prediction(body_feats, spatial_scale,
                                                  im_info)
                outputs = {'bbox_out': pred}
            else:
                outputs = {'body_feats': body_feats}

                place = fluid.CPUPlace()
                exe = fluid.Executor(place)
                for param in context_prog.global_block().iter_parameters():
                    param.trainable = trainable
                if pretrained:

                    def _if_exist(var):
                        return os.path.exists(
                            os.path.join(self.default_pretrained_model_path,
                                         var.name))

                    if not param_prefix:
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
                         use_gpu=False,
                         batch_size=1,
                         output_dir=None,
                         score_thresh=0.5,
                         draw_bbox=True):
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
        :param draw_bbox: whether to draw bounding box and save images.
        :type draw_bbox: bool
        """
        if self.infer_prog is None:
            inputs, outputs, self.infer_prog = self.context(
                trainable=False, pretrained=True, get_prediction=True)
            self.infer_prog = self.infer_prog.clone(for_test=True)
            self.image = inputs['image']
            self.im_info = inputs['im_info']
            self.bbox_out = outputs['bbox_out']

        place = fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()
        exe = fluid.Executor(place)

        all_images = []
        paths = paths if paths else []
        for yield_data in test_reader(paths, images):
            all_images.append(yield_data)

        images_num = len(all_images)
        loop_num = int(np.ceil(images_num / batch_size))

        res = []
        for iter_id in range(loop_num):
            batch_data = []
            handle_id = iter_id * batch_size
            for image_id in range(batch_size):
                try:
                    batch_data.append(all_images[handle_id + image_id])
                except:
                    pass
            padding_image, padding_info = padding_minibatch(
                batch_data, coarsest_stride=32, use_padded_im_info=True)
            feed = {'image': padding_image, 'im_info': padding_info}
            data_out = exe.run(
                self.infer_prog,
                feed=feed,
                fetch_list=[self.bbox_out],
                return_numpy=False)
            output = postprocess(
                paths=paths,
                images=images,
                data_out=data_out,
                score_thresh=score_thresh,
                label_names=self.label_names,
                output_dir=output_dir,
                handle_id=handle_id,
                draw_bbox=draw_bbox)
            res.append(output)
        return res
