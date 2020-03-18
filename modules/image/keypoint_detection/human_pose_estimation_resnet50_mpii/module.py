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

from .processor import postprocess
from .data_feed import reader
from .pose_resnet import ResNet


@moduleinfo(
    name="human_pose_estimation_resnet50_mpii",
    type="CV/keypoint_detection",
    author="paddlepaddle",
    author_email="paddle-dev@baidu.comi",
    summary=
    "Paddle implementation for the paper `Simple baselines for human pose estimation and tracking`, trained with the MPII dataset.",
    version="2.0.0")
class HumanPoseEstimation(hub.Module):
    def _initialize(self):
        self.default_pretrained_model_path = os.path.join(
            self.directory, "pose-resnet50-mpii-384x384")
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
    def keypoint_detection(self,
                           images=None,
                           paths=None,
                           batch_size=1,
                           use_gpu=False,
                           output_dir=None,
                           visualization=False):
        """
        API for human pose estimation and tracking.

        Args:
            images (numpy.ndarray): data of images, with shape [N, H, W, C].
            paths (list[str]): The paths of images.
            batch_size (int): batch size.
            use_gpu (bool): Whether to use gpu.
            output_dir (str): The path to store output images.
            visualization (bool): Whether to save image or not.

        Returns:
            res (list[collections.OrderedDict]): The key points of human pose.
        """
        # create output directory
        output_dir = output_dir if output_dir else os.path.join(
            os.getcwd(), 'keypoint_detection_result')

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
                    out_heatmaps=output[i],
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
            description="Run the human_pose_estimation_resnet50_mpii module.",
            prog='hub run human_pose_estimation_resnet50_mpii',
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
            default=None,
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
