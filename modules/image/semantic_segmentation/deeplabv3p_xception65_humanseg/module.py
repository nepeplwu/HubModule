# coding=utf-8
from __future__ import absolute_import
from __future__ import division

import ast
import os
import argparse

import numpy as np
import paddle.fluid as fluid
import paddlehub as hub
from paddle.fluid.core import PaddleTensor, AnalysisConfig, create_paddle_predictor
from paddlehub.module.module import moduleinfo, runnable, serving

from deeplabv3p_xception65_humanseg.processor import postprocess
from deeplabv3p_xception65_humanseg.data_feed import reader


@moduleinfo(
    name="deeplabv3p_xception65_humanseg",
    type="CV/semantic_segmentation",
    author="paddlepaddle",
    author_email="paddle-dev@baidu.comi",
    summary="DeepLabv3+ is a semantic segmentation model.",
    version="1.1.0")
class DeeplabV3pXception65HumanSeg(hub.Module):
    def _initialize(self):
        self.default_pretrained_model_path = os.path.join(
            self.directory, "deeplabv3p_xception65_humanseg_model")
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
            gpu_config.enable_use_gpu(
                memory_pool_init_size_mb=1000, device_id=0)
            self.gpu_predictor = create_paddle_predictor(gpu_config)

    @serving
    def segmentation(self,
                     images=None,
                     paths=None,
                     batch_size=1,
                     use_gpu=False,
                     visualization=False,
                     output_dir='human_seg_output'):
        """
        API for human segmentation.

        Args:
            images (list(numpy.ndarray)): images data, shape of each is [H, W, C], the color space is BGR.
            paths (list[str]): The paths of images.
            batch_size (int): batch size.
            use_gpu (bool): Whether to use gpu.
            visualization (bool): Whether to save image or not.
            output_dir (str): The path to store output images.

        Returns:
            res (list[dict]): each element in the list is a dict, the keys and values are:
                save_path (str, optional): the path to save images. (Exists only if visualization is True)
                data (numpy.ndarray): data of post processed image.
        """
        all_data = list()
        for yield_data in reader(images, paths):
            all_data.append(yield_data)

        total_num = len(all_data)
        loop_num = int(np.ceil(total_num / batch_size))

        res = list()
        for iter_id in range(loop_num):
            batch_data = list()
            handle_id = iter_id * batch_size
            for image_id in range(batch_size):
                try:
                    batch_data.append(all_data[handle_id + image_id])
                except:
                    pass
            # feed batch image
            batch_image = np.array([data['image'] for data in batch_data])
            batch_image = PaddleTensor(batch_image.copy())
            output = self.gpu_predictor.run([
                batch_image
            ]) if use_gpu else self.cpu_predictor.run([batch_image])
            output = np.expand_dims(output[0].as_ndarray(), axis=1)
            # postprocess one by one
            for i in range(len(batch_data)):
                out = postprocess(
                    data_out=output[i],
                    org_im=batch_data[i]['org_im'],
                    org_im_shape=batch_data[i]['org_im_shape'],
                    org_im_path=batch_data[i]['org_im_path'],
                    output_dir=output_dir,
                    visualization=visualization)
                res.append(out)
        return res

    @runnable
    def run_cmd(self, argvs):
        """
        Run as a command.
        """
        self.parser = argparse.ArgumentParser(
            description="Run the deeplabv3p_xception65_humanseg module.",
            prog='hub run deeplabv3p_xception65_humanseg',
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
        results = self.keypoint_detection(
            paths=[args.input_path],
            batch_size=args.batch_size,
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
            default='human_seg_output',
            help="The directory to save output images.")
        self.arg_config_group.add_argument(
            '--visualization',
            type=ast.literal_eval,
            default=False,
            help="whether to save output as images.")
        self.arg_config_group.add_argument(
            '--batch_size',
            type=ast.literal_eval,
            default=1,
            help="batch size.")

    def add_module_input_arg(self):
        """
        Add the command input options.
        """
        self.arg_input_group.add_argument(
            '--input_path', type=str, help="path to image.")
