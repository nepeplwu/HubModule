# coding=utf-8
from __future__ import absolute_import
from __future__ import division

import time
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
    author="paddlepaddle",
    author_email="paddlepaddle@baidu.com")
class StyleProjection(hub.Module):
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

    def style_transfer(self,
                       images=None,
                       paths=None,
                       alpha=1,
                       use_gpu=False,
                       output_dir=None,
                       visualization=False):
        """
        API for image style transfer.

        Args:
            images (list): list of [content_arr, style_arrs_list, style_interpolation_weights],
                the first element is a numpy.ndarry with shape [H, W, C], content data.
                the second element is a list of numpy.ndarray with shape [H, W, C], styles data.
                the last element is a list (Optional), the interpolation weights correspond to styles.
            paths (list): list of [content_path, style_paths_list, style_interpolation_weights],
                the first element is a str, the path to content,
                the second element is a list, the path to styles,
                the last element is a list (Optional), the interpolation weights correspond to styles.
            alpha (float): The weight that controls the degree of stylization. Should be between 0 and 1.
            use_gpu (bool): whether to use gpu.
            output_dir (str): the path to store output images.
            visualization (bool): whether to save image or not.

        Returns:
            im_output (list of numpy.ndarray): list of output images.
        """
        # create output directory
        output_dir = output_dir if output_dir else os.path.join(
            os.getcwd(), 'transfer_result')
        im_output = list()
        # images
        if images:
            for arr_list in images:
                if len(arr_list) > 1:
                    content_arr = arr_list[0]
                    content = PaddleTensor(reader(im_arr=content_arr).copy())
                    styles_arr_list = arr_list[1]
                    if len(arr_list) == 3:
                        style_interpolation_weights = arr_list[2]
                    else:
                        style_interpolation_weights = np.ones(
                            len(styles_arr_list))
                    style_interpolation_weights = [
                        style_interpolation_weights[j] /
                        sum(style_interpolation_weights)
                        for j in range(len(style_interpolation_weights))
                    ]
                    accumulate = np.zeros((3, 512, 512))
                    save_im_name = output_dir + '/' + 'time={}'.format(
                        time.time()) + '_alpha={}_'.format(alpha)
                    for i, style_arr in enumerate(styles_arr_list):
                        style = PaddleTensor(reader(im_arr=style_arr).copy())
                        # encode
                        if use_gpu:
                            content_feats = self.gpu_predictor_enc.run(
                                [content])
                            style_feats = self.gpu_predictor_enc.run([style])
                        else:
                            content_feats = self.cpu_predictor_enc.run(
                                [content])
                            style_feats = self.cpu_predictor_enc.run([style])
                        fr_feats = fr(content_feats[0].as_ndarray(),
                                      style_feats[0].as_ndarray(), alpha)
                        fr_feats = PaddleTensor(fr_feats.copy())
                        # decode
                        if use_gpu:
                            predict_outputs = self.gpu_predictor_dec.run(
                                [fr_feats])
                        else:
                            predict_outputs = self.cpu_predictor_dec.run(
                                [fr_feats])
                        # interpolation
                        accumulate += predict_outputs[0].as_ndarray(
                        )[0] * style_interpolation_weights[i]
                    # postprocess
                    save_im_name += '{}_styles'.format(
                        len(styles_arr_list)) + '.jpg'
                    path_result = postprocess(accumulate, output_dir,
                                              save_im_name, visualization)
                    im_output.append(path_result)
                else:
                    raise ValueError(
                        'each element is a list, whose length must be larger than 1.'
                    )
        # paths
        if paths:
            for path in paths:
                if len(path) > 1:
                    content_path = path[0]
                    content = PaddleTensor(reader(im_path=content_path).copy())
                    style_paths = path[1]
                    if len(path) == 3:
                        style_interpolation_weights = path[2]
                    else:
                        style_interpolation_weights = np.ones(len(style_paths))
                    style_interpolation_weights = [
                        style_interpolation_weights[j] /
                        sum(style_interpolation_weights)
                        for j in range(len(style_interpolation_weights))
                    ]
                    accumulate = np.zeros((3, 512, 512))
                    save_im_name = output_dir + '/' + os.path.splitext(
                        os.path.basename(
                            content_path))[0] + '_alpha={}_'.format(alpha)
                    for i, style_path in enumerate(style_paths):
                        style = PaddleTensor(reader(im_path=style_path).copy())
                        # encode
                        if use_gpu:
                            content_feats = self.gpu_predictor_enc.run(
                                [content])
                            style_feats = self.gpu_predictor_enc.run([style])
                        else:
                            content_feats = self.cpu_predictor_enc.run(
                                [content])
                            style_feats = self.cpu_predictor_enc.run([style])
                        fr_feats = fr(content_feats[0].as_ndarray(),
                                      style_feats[0].as_ndarray(), alpha)
                        fr_feats = PaddleTensor(fr_feats.copy())
                        # decode
                        if use_gpu:
                            predict_outputs = self.gpu_predictor_dec.run(
                                [fr_feats])
                        else:
                            predict_outputs = self.cpu_predictor_dec.run(
                                [fr_feats])
                        # interpolation
                        accumulate += predict_outputs[0].as_ndarray(
                        )[0] * style_interpolation_weights[i]
                        save_im_name = save_im_name + os.path.splitext(
                            os.path.basename(style_path)
                        )[0] + '_w=%.2f_&' % style_interpolation_weights[i]
                    # postprocess
                    save_im_name += '.jpg'
                    path_result = postprocess(accumulate, output_dir,
                                              save_im_name, visualization)
                    im_output.append(path_result)
                else:
                    raise ValueError(
                        'path is a list, whose length must be larger than 1.')
        return im_output
