# coding=utf-8
from __future__ import absolute_import
from __future__ import division

import ast
import time
import os

import argparse
import numpy as np
import paddle.fluid as fluid
import paddlehub as hub
from paddle.fluid.core import PaddleTensor, AnalysisConfig, create_paddle_predictor
from paddlehub.module.module import moduleinfo, runnable, serving

from .encoder_network import encoder_net
from .decoder_network import decoder_net
from .processor import postprocess, fr
from .data_feed import reader


@moduleinfo(
    name="stylepro_artistic_coco_wikiart",
    version="1.0.0",
    type="cv/style_transfer",
    summary=
    "StylePro Artistic is an algorithm for Arbitrary image style, which is parameter-free, fast yet effective.",
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

    @serving
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
        for component in reader(images, paths):
            content = PaddleTensor(component['content_arr'].copy())
            content_feats = self.gpu_predictor_enc.run(
                [content]) if use_gpu else self.cpu_predictor_enc.run([content])
            accumulate = np.zeros((3, 512, 512))
            for i, style_arr in enumerate(component['styles_arr_list']):
                style = PaddleTensor(style_arr.copy())
                # encode
                style_feats = self.gpu_predictor_enc.run(
                    [style]) if use_gpu else self.cpu_predictor_enc.run([style])
                fr_feats = fr(content_feats[0].as_ndarray(),
                              style_feats[0].as_ndarray(), alpha)
                fr_feats = PaddleTensor(fr_feats.copy())
                # decode
                predict_outputs = self.gpu_predictor_dec.run([
                    fr_feats
                ]) if use_gpu else self.cpu_predictor_dec.run([fr_feats])
                # interpolation
                accumulate += predict_outputs[0].as_ndarray(
                )[0] * component['style_interpolation_weights'][i]
                # postprocess
                save_im_name = output_dir + '/' + component[
                    'save_im_name'] + '_alpha={}_'.format(alpha) + '.jpg'
                path_result = postprocess(accumulate, output_dir, save_im_name,
                                          visualization)
                im_output.append(path_result)
        return im_output

    @runnable
    def run_cmd(self, argvs):
        """
        Run as a command.
        """
        self.parser = argparse.ArgumentParser(
            description="Run the style_projection_coco_wikiart module.",
            prog='hub run style_projection_coco_wikiart',
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
        if args.style_interpolation_weights is None:
            paths = [[args.content_path, args.style_paths.split(',')]]
        else:
            paths = [[
                args.content_path,
                args.style_paths.split(','),
                list(args.style_interpolation_weights)
            ]]
        results = self.style_transfer(
            paths=paths,
            alpha=args.alpha,
            use_gpu=args.use_gpu,
            output_dir=args.output_dir,
            visualization=args.visualization)
        return results

    def add_module_config_arg(self):
        """
        Add the command config options.
        """
        self.arg_config_group.add_argument(
            '--use_gpu',
            type=ast.literal_eval,
            default=False,
            help="whether use GPU or not")
        self.arg_config_group.add_argument(
            '--output_dir',
            type=str,
            default=None,
            help="The directory to save output images.")
        self.arg_config_group.add_argument(
            '--visualization',
            type=ast.literal_eval,
            default=False,
            help="whether to save output as images.")

    def add_module_input_arg(self):
        """
        Add the command input options.
        """
        self.arg_input_group.add_argument(
            '--content_path', type=str, help="path to content.")
        self.arg_input_group.add_argument(
            '--style_paths', type=str, help="path to styles.")
        self.arg_input_group.add_argument(
            '--style_interpolation_weights',
            type=ast.literal_eval,
            default=None,
            help="interpolation weights of styles.")
        self.arg_config_group.add_argument(
            '--alpha',
            type=ast.literal_eval,
            default=1,
            help="The parameter to control the tranform degree.")
