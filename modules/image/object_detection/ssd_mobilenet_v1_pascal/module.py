# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from functools import partial

import numpy as np
import paddle.fluid as fluid
import paddlehub as hub
from paddlehub.module.module import moduleinfo
from paddle.fluid.core import PaddleTensor, AnalysisConfig, create_paddle_predictor


@moduleinfo(
    name="ssd_mobilenet_v1_pascal",
    version="1.1.0",
    type="cv/object_detection",
    summary="SSD with backbone MobileNet_V1, trained with dataset Pasecal VOC.",
    author="paddlepaddle",
    author_email="paddle-dev@baidu.com")
class SSDMobileNetv1(hub.Module):
    def _initialize(self):
        self.ssd = hub.Module(name="ssd")
        # default pretrained model of SSD_MobileNet_V1_VOC, the shape of image tensor is (3, 300, 300)
        self.default_pretrained_model_path = os.path.join(
            self.directory, "ssd_mobilenet_v1_model")
        self.label_names = self.ssd.load_label_info(
            os.path.join(self.directory, "label_file.txt"))
        self.infer_prog = None
        self.image = None
        self.bbox_out = None
        self._set_config()

    def _set_config(self):
        """
        predictor config setting
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
                multi_box_head=None,
                ssd_output_decoder=None,
                input_image=None,
                trainable=True,
                pretrained=True,
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
        startup_program = fluid.Program()
        with fluid.program_guard(wrapped_prog, startup_program):
            with fluid.unique_name.guard():
                # image
                image = input_image if input_image else fluid.layers.data(
                    name='image', shape=[3, 300, 300], dtype='float32')
                mobilenet_v1 = hub.Module(name='mobilenet_v1_imagenet')
                _, _outputs, _ = mobilenet_v1.context(input_image=image)
                body_feats = _outputs['body_feats']
                # multi_box_head
                if multi_box_head is None:
                    multi_box_head = self.ssd.MultiBoxHead(
                        base_size=300,
                        num_classes=21,
                        aspect_ratios=[[2.], [2., 3.], [2., 3.], [2., 3.],
                                       [2., 3.], [2., 3.]],
                        min_ratio=20,
                        max_ratio=90,
                        min_sizes=[60.0, 105.0, 150.0, 195.0, 240.0, 285.0],
                        max_sizes=[[], 150.0, 195.0, 240.0, 285.0, 300.0],
                        offset=0.5,
                        flip=True)
                # ssd_output_decoder
                if ssd_output_decoder is None:
                    ssd_output_decoder = self.ssd.SSDOutputDecoder(
                        nms_threshold=0.45,
                        nms_top_k=400,
                        keep_top_k=200,
                        score_threshold=0.01,
                        nms_eta=1.0,
                        background_label=0)
                # call ssd.context
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
                if pretrained:

                    def _if_exist(var):
                        return os.path.exists(
                            os.path.join(self.default_pretrained_model_path,
                                         var.name))

                    load_default_pretrained_model = True
                    if param_prefix:
                        load_default_pretrained_model = False
                    elif input_image:
                        if input_image.shape != (-1, 3, 300, 300):
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
        :param visualization: whether to draw bounding box and save images.
        :type visualization: bool
        """
        if self.infer_prog is None:
            inputs, outputs, self.infer_prog = self.context(
                trainable=False, pretrained=True, get_prediction=True)
            self.infer_prog = self.infer_prog.clone(for_test=True)
            self.image = inputs['image']
            self.bbox_out = outputs['bbox_out']

        place = fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()
        exe = fluid.Executor(place)
        # create reader
        feeder = fluid.DataFeeder([self.image], place)
        decode_image = self.ssd.DecodeImage(to_rgb=True, with_mixup=False)
        resize_image = self.ssd.ResizeImage(
            target_size=300, interp=1, max_size=0, use_cv2=False)
        permute_image = self.ssd.Permute()
        normalize_image = self.ssd.NormalizeImage(
            mean=[127.5, 127.5, 127.5],
            std=[127.502231, 127.502231, 127.502231],
            is_scale=False)
        data_reader = partial(
            self.ssd.reader,
            paths,
            images,
            decode_image=decode_image,
            resize_image=resize_image,
            permute_image=permute_image,
            normalize_image=normalize_image)
        batch_reader = fluid.io.batch(data_reader, batch_size=batch_size)
        # execute program
        output_path = output_dir if output_dir else os.path.join(
            os.getcwd(), 'detection_result')
        paths = paths if paths else list()
        res = []
        for iter_id, feed_data in enumerate(batch_reader()):
            np_data = np.array(feed_data).astype('float32')
            if np_data.shape == 1:
                np_data = np_data[0]
            else:
                np_data = np.squeeze(np_data, axis=1)
            data_tensor = PaddleTensor(np_data.copy())
            if use_gpu:
                data_out = self.gpu_predictor.run([data_tensor])
            else:
                data_out = self.cpu_predictor.run([data_tensor])
            output = self.ssd.postprocess(
                paths=paths,
                images=images,
                data_out=data_out,
                score_thresh=score_thresh,
                label_names=self.label_names,
                output_dir=output_path,
                handle_id=iter_id * batch_size,
                visualization=visualization)
            res.append(output)

        return res
