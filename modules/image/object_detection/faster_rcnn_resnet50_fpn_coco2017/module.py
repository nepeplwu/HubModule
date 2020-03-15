# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import ast
import argparse
from collections import OrderedDict
from functools import partial
from math import ceil

import numpy as np
import paddle.fluid as fluid
import paddlehub as hub
from paddlehub.module.module import moduleinfo, runnable
from paddle.fluid.core import PaddleTensor, AnalysisConfig, create_paddle_predictor

from faster_rcnn_resnet50_fpn_coco2017.fpn import FPN


@moduleinfo(
    name="faster_rcnn_resnet50_fpn_coco2017",
    version="1.0.0",
    type="cv/object_detection",
    summary=
    "Baidu's Faster-RCNN model for object detection, whose backbone is ResNet50, processed with Feature Pyramid Networks",
    author="paddlepaddle",
    author_email="paddle-dev@baidu.com")
class FasterRCNNResNet50RPN(hub.Module):
    def _initialize(self):
        self.faster_rcnn = hub.Module(name="faster_rcnn")
        # default pretrained model, Faster-RCNN with backbone ResNet50, shape of input tensor is [3, 800, 1333]
        self.default_pretrained_model_path = os.path.join(
            self.directory, "faster_rcnn_resnet50_fpn_model")
        self.label_names = self.faster_rcnn.load_label_info(
            os.path.join(self.directory, "label_file.txt"))
        self.infer_prog = None
        self.bbox_out = None
        self._set_config()

    def _set_config(self):
        """
        predictor config setting
        """
        cpu_config = AnalysisConfig(self.default_pretrained_model_path)
        cpu_config.disable_glog_info()
        cpu_config.disable_gpu()
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
                rpn_head=None,
                roi_extractor=None,
                bbox_head=None,
                bbox_assigner=None,
                input_image=None,
                trainable=True,
                pretrained=True,
                param_prefix='',
                phase='train'):
        """Distill the Head Features, so as to perform transfer learning.

        :param rpn_head: Head of Region Proposal Network
        :type rpn_head: <class 'RPNHead' object>
        :param bbox_head: Head of Bounding Box.
        :type bbox_head: <class 'BBoxHead' object>
        :param bbox_assigner: Parameters of fluid.layers.generate_proposal_labels.
        :type bbox_assigner: <class 'BBoxAssigner' object>
        :param input_image: image tensor.
        :type input_image: <class 'paddle.fluid.framework.Variable'>
        :param trainable: whether to set parameters trainable.
        :type trainable: bool
        :param pretrained: whether to load default pretrained model.
        :type pretrained: bool
        :param param_prefix: the prefix of parameters in neural network.
        :type param_prefix: str
        :param phase: Optional Choice: 'predict', 'train'
        :type phase: str
        """
        wrapped_prog = input_image.block.program if input_image else fluid.Program(
        )
        startup_program = fluid.Program()
        with fluid.program_guard(wrapped_prog, startup_program):
            with fluid.unique_name.guard():
                image = input_image if input_image else fluid.layers.data(
                    name='image', shape=[3, 800, 1333], dtype='float32')
                resnet = hub.Module(name='resnet50_v2_imagenet')
                _, _outputs, _ = resnet.context(input_image=image, variant='b',\
                                                 norm_type='affine_channel', feature_maps=[2, 3, 4, 5])
                body_feats = _outputs['body_feats']

                # fpn: FPN
                fpn = FPN(
                    max_level=6,
                    min_level=2,
                    num_chan=256,
                    spatial_scale=[0.03125, 0.0625, 0.125, 0.25])
                # rpn_head: FPNRPNHead
                if rpn_head is None:
                    rpn_head = self.faster_rcnn.FPNRPNHead(
                        anchor_generator=self.faster_rcnn.AnchorGenerator(
                            anchor_sizes=[32, 64, 128, 256, 512],
                            aspect_ratios=[0.5, 1.0, 2.0],
                            stride=[16.0, 16.0],
                            variance=[1.0, 1.0, 1.0, 1.0]),
                        rpn_target_assign=self.faster_rcnn.RPNTargetAssign(
                            rpn_batch_size_per_im=256,
                            rpn_fg_fraction=0.5,
                            rpn_negative_overlap=0.3,
                            rpn_positive_overlap=0.7,
                            rpn_straddle_thresh=0.0),
                        train_proposal=self.faster_rcnn.GenerateProposals(
                            min_size=0.0,
                            nms_thresh=0.7,
                            post_nms_top_n=2000,
                            pre_nms_top_n=2000),
                        test_proposal=self.faster_rcnn.GenerateProposals(
                            min_size=0.0,
                            nms_thresh=0.7,
                            post_nms_top_n=1000,
                            pre_nms_top_n=1000),
                        anchor_start_size=32,
                        num_chan=256,
                        min_level=2,
                        max_level=6)
                # roi_extractor: FPNRoIAlign
                if roi_extractor is None:
                    roi_extractor = self.faster_rcnn.FPNRoIAlign(
                        canconical_level=4,
                        canonical_size=224,
                        max_level=5,
                        min_level=2,
                        box_resolution=7,
                        sampling_ratio=2)
                # bbox_head: BBoxHead
                if bbox_head is None:
                    bbox_head = self.faster_rcnn.BBoxHead(
                        head=self.faster_rcnn.TwoFCHead(mlp_dim=1024),
                        nms=self.faster_rcnn.MultiClassNMS(
                            keep_top_k=100,
                            nms_threshold=0.5,
                            score_threshold=0.05))
                # bbox_assigner: BBoxAssigner
                if bbox_assigner is None:
                    bbox_assigner = self.faster_rcnn.BBoxAssigner(
                        batch_size_per_im=512,
                        bbox_reg_weights=[0.1, 0.1, 0.2, 0.2],
                        bg_thresh_hi=0.5,
                        bg_thresh_lo=0.0,
                        fg_fraction=0.25,
                        fg_thresh=0.5)
                # Base Class
                inputs, outputs, context_prog = self.faster_rcnn.context(
                    body_feats=body_feats,
                    fpn=fpn,
                    rpn_head=rpn_head,
                    roi_extractor=roi_extractor,
                    bbox_head=bbox_head,
                    bbox_assigner=bbox_assigner,
                    image=image,
                    trainable=trainable,
                    param_prefix=param_prefix,
                    phase=phase)

                place = fluid.CPUPlace()
                exe = fluid.Executor(place)
                if pretrained:

                    def _if_exist(var):
                        return os.path.exists(
                            os.path.join(self.default_pretrained_model_path,
                                         var.name))

                    load_default_pretrained_model = True
                    if param_prefix:
                        load_default_pretrained_model = False
                    elif input_image:
                        if input_image.shape != (-1, 3, 800, 1333):
                            load_default_pretrained_model = False
                    if load_default_pretrained_model:
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
                         visualization=True):
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
        :param visualization: whether to draw box and save images.
        :type visualization: bool
        """
        if self.infer_prog is None:
            inputs, outputs, self.infer_prog = self.context(
                trainable=False, pretrained=True, phase='predict')
            self.infer_prog = self.infer_prog.clone(for_test=True)
            self.bbox_out = outputs['bbox_out']

        place = fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()
        exe = fluid.Executor(place)
        output_path = output_dir if output_dir else os.path.join(
            os.getcwd(), 'detection_result')

        all_images = []
        paths = paths if paths else list()
        for yield_data in self.faster_rcnn.test_reader(paths, images):
            all_images.append(yield_data)
        fluid.io.save_inference_model(
            dirname="./faster_rcnn_resnet50_fpn_model",
            feeded_var_names=['image', 'im_info', 'im_shape'],
            target_vars=[self.bbox_out],
            executor=exe,
            main_program=self.infer_prog)
        images_num = len(all_images)
        loop_num = ceil(images_num / batch_size)
        res = []
        for iter_id in range(loop_num):
            batch_data = []
            handle_id = iter_id * batch_size
            for image_id in range(batch_size):
                try:
                    batch_data.append(all_images[handle_id + image_id])
                except:
                    pass
            padding_image, padding_info, padding_shape = self.faster_rcnn.padding_minibatch(
                batch_data, coarsest_stride=32, use_padded_im_info=True)
            padding_image_tensor = PaddleTensor(padding_image.copy())
            padding_info_tensor = PaddleTensor(padding_info.copy())
            padding_shape_tensor = PaddleTensor(padding_shape.copy())
            feed_list = [
                padding_image_tensor, padding_info_tensor, padding_shape_tensor
            ]
            if use_gpu:
                data_out = self.gpu_predictor.run(feed_list)
            else:
                data_out = self.cpu_predictor.run(feed_list)

            output = self.faster_rcnn.postprocess(
                paths=paths,
                images=images,
                data_out=data_out,
                score_thresh=score_thresh,
                label_names=self.label_names,
                output_dir=output_path,
                handle_id=handle_id,
                visualization=visualization)
            res.append(output)
        return res

    def add_module_config_arg(self):
        """
        Add the command config options
        """
        self.arg_config_group.add_argument(
            '--use_gpu',
            type=ast.literal_eval,
            default=False,
            help="whether use GPU or not")

        self.arg_config_group.add_argument(
            '--batch_size',
            type=int,
            default=1,
            help="batch size for prediction")
        self.arg_config_group.add_argument(
            '--user_dict',
            type=str,
            default=None,
            help=
            "customized dictionary for intervening the word segmentation result"
        )

    def add_module_input_arg(self):
        """
        Add the command input options
        """
        self.arg_input_group.add_argument(
            '--input_path',
            type=str,
            default=None,
            help="file contain input data")

    @runnable
    def run_cmd(self, argvs):
        self.parser = argparse.ArgumentParser(
            description="Run the faster_rcnn_resnet50_fpn_coco2017.",
            prog='hub run faster_rcnn_resnet50_fpn_coco2017',
            usage='%(prog)s',
            add_help=True)
        self.arg_input_group = self.parser.add_argument_group(
            title="Input options", description="Input data. Required")
        self.arg_config_group = self.parser.add_argument_group(
            title="Config options",
            description=
            "Run configuration for controlling module behavior, not required.")
        self.add_module_config_arg()

        self.add_module_input_arg()
        args = self.parser.parse_args(argvs)
        input_path = args.input_path
        if os.path.exists(input_path) == False:
            raise ValueError("input_path is not exit")
        return self.object_detection(
            paths=[input_path],
            use_gpu=args.use_gpu,
            batch_size=args.batch_size)
