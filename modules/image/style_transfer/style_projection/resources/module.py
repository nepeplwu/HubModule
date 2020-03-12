# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import paddle.fluid as fluid
import paddlehub as hub

from paddle.fluid.core import PaddleTensor, AnalysisConfig, create_paddle_predictor
from paddlehub.module.module import moduleinfo

from .encoder_network import encoder_net
from .decoder_network import decoder_net
from .processor import postprocess, fr
from .data_feed import reader


@moduleinfo(
    name="style_projection_coco_wikiart",
    version="1.0.0",
    type="cv/style_transfer",
    summary=
    "Style Projection is an algorithm for Arbitrary image style, which is parameter-free, fast yet effective.",
    author="paddle",
    author_email="paddlepaddle@baidu.com")
class HubModule(hub.Module):
    def _initialize(self):
        self.pretrained_encoder_net = os.path.join(self.directory,
                                                   "style_projection_enc")
        self.pretrained_decoder_net = os.path.join(self.directory,
                                                   "style_projection_dec")
        self._set_config()

    def _set_config(self):
        """
        predictor config setting
        """
        # encoder
        cpu_config_enc = AnalysisConfig(self.pretrained_encoder_net)
        cpu_config_enc.disable_glog_info()
        cpu_config_enc.disable_gpu()
        self.cpu_predictor_enc = create_paddle_predictor(cpu_config_enc)
        # decoder
        cpu_config_dec = AnalysisConfig(self.pretrained_decoder_net)
        cpu_config_dec.disable_glog_info()
        cpu_config_dec.disable_gpu()
        self.cpu_predictor_dec = create_paddle_predictor(cpu_config_dec)

        try:
            _places = os.environ["CUDA_VISIBLE_DEVICES"]
            int(_places[0])
            use_gpu = True
        except:
            use_gpu = False
        if use_gpu:
            # encoder
            gpu_config_enc = AnalysisConfig(self.pretrained_encoder_net)
            gpu_config_enc.disable_glog_info()
            gpu_config_enc.enable_use_gpu(
                memory_pool_init_size_mb=1000, device_id=0)
            self.gpu_predictor_enc = create_paddle_predictor(gpu_config_enc)
            # decoder
            gpu_config_dec = AnalysisConfig(self.pretrained_decoder_net)
            gpu_config_dec.disable_glog_info()
            gpu_config_dec.enable_use_gpu(
                memory_pool_init_size_mb=1000, device_id=0)
            self.gpu_predictor_dec = create_paddle_predictor(gpu_config_dec)

    def encoder_context(self, trainable=False, pretrained=False):
        """encoder for transfer learning."""
        encoder_prog = fluid.Program()
        startup_prog = fluid.Program()
        with fluid.program_guard(encoder_prog, startup_prog):
            with fluid.unique_name.guard():
                enc_image, enc_outputs = encoder_net()
                place = fluid.CPUPlace()
                exe = fluid.Executor(place)
                # pretrained
                if pretrained:

                    def _if_exist(var):
                        b = os.path.exists(
                            os.path.join(self.pretrained_encoder_net, var.name))
                        return b

                    fluid.io.load_vars(
                        exe,
                        self.pretrained_encoder_net,
                        encoder_prog,
                        predicate=_if_exist)
                else:
                    exe.run(startup_prog)
                # trainable
                for param in encoder_prog.global_block().iter_parameters():
                    param.trainable = trainable
        return enc_image, enc_outputs, encoder_prog

    def decoder_context(self, trainable=False, pretrained=False):
        """decoder for transfer learning."""
        decoder_prog = fluid.Program()
        startup_prog = fluid.Program()
        with fluid.program_guard(decoder_prog, startup_prog):
            with fluid.unique_name.guard():
                dec_image, dec_outputs = decoder_net()
                place = fluid.CPUPlace()
                exe = fluid.Executor(place)
                # pretrained
                if pretrained:

                    def _if_exist(var):
                        b = os.path.exists(
                            os.path.join(self.pretrained_decoder_net, var.name))
                        return b

                    fluid.io.load_vars(
                        exe,
                        self.pretrained_decoder_net,
                        decoder_prog,
                        predicate=_if_exist)
                else:
                    exe.run(startup_prog)
                # trainable
                for param in decoder_prog.global_block().iter_parameters():
                    param.trainable = trainable
        return dec_image, dec_outputs, decoder_prog

    def style_transfer(self,
                       content_paths,
                       style_paths,
                       alpha,
                       use_gpu=False,
                       output_dir=None):
        """API for image style transfer.

        :param content_paths: file paths to the content images.
        :type content_paths: list of str
        :param style_paths: file paths to the style images.
        :type style_paths: list of str
        :param alpha: The weight that controls the degree of stylization. Should be between 0 and 1.
        :type alpha: float
        :param use_gpu: whether to use gpu.
        :type use_gpu: bool
        :param output_dir: the path to store output images.
        :type output_dir: str
        """
        output_dir = output_dir if output_dir else os.path.join(
            os.getcwd(), 'transfer_result')
        for content_path in content_paths:
            for style_path in style_paths:
                # preprocess to prepare data
                content = PaddleTensor(reader(content_path).copy())
                style = PaddleTensor(reader(style_path).copy())
                # encode
                if use_gpu:
                    content_feats = self.gpu_predictor_enc.run([content])
                    style_feats = self.gpu_predictor_enc.run([style])
                else:
                    content_feats = self.cpu_predictor_enc.run([content])
                    style_feats = self.cpu_predictor_enc.run([style])
                fr_feats = fr(content_feats[0].as_ndarray(),
                              style_feats[0].as_ndarray(), alpha)
                fr_feats = PaddleTensor(fr_feats.copy())
                # decode
                if use_gpu:
                    outputs = self.gpu_predictor_dec.run([fr_feats])
                else:
                    outputs = self.cpu_predictor_dec.run([fr_feats])
                output_data = outputs[0].as_ndarray()
                # postprocess
                postprocess(output_data[0], content_path, style_path,
                            output_dir)
